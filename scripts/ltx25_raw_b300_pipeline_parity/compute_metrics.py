#!/usr/bin/env python3
"""Compute decoded-video SSIM, PSNR, and LPIPS for official/Omni pairs."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


FILENAME = re.compile(
    r"^(?P<mode>full_one_stage|distilled_one_stage|distilled_two_stage|full_two_stage)-"
    r"(?P<modality>t2v|i2v)-(?P<backend>official|omni)-seed-(?P<seed>\d+)\.mp4$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results-v2" / "videos",
    )
    parser.add_argument("--lpips-frames", type=int, default=24)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=24,
        help="Compare only the first N decoded frames; 24 frames is one second at 24 FPS.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute pairs already present in results-v2/videos/metrics.json.",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "full_one_stage",
            "distilled_one_stage",
            "distilled_two_stage",
            "full_two_stage",
        ),
    )
    parser.add_argument("--modality", choices=("t2v", "i2v"))
    parser.add_argument("--seeds", type=int, nargs="+")
    return parser.parse_args()


def video_metadata(capture: cv2.VideoCapture) -> dict[str, int | float]:
    return {
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }


def lpips_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32)
    return tensor.permute(2, 0, 1).unsqueeze(0).div(127.5).sub(1.0)


@torch.inference_mode()
def compare_pair(
    official_path: Path,
    omni_path: Path,
    *,
    lpips_model,
    lpips_frames: int,
    max_frames: int,
    device: torch.device,
) -> dict:
    official = cv2.VideoCapture(str(official_path))
    omni = cv2.VideoCapture(str(omni_path))
    if not official.isOpened() or not omni.isOpened():
        raise RuntimeError(f"Unable to open pair: {official_path}, {omni_path}")
    official_meta = video_metadata(official)
    omni_meta = video_metadata(omni)
    if official_meta != omni_meta:
        raise ValueError(
            f"Video metadata mismatch: official={official_meta}, omni={omni_meta}"
        )

    available_frames = int(official_meta["frame_count"])
    total_frames = min(available_frames, max_frames)
    sampled_indices = set(
        np.linspace(
            0, max(total_frames - 1, 0), min(lpips_frames, total_frames), dtype=int
        )
    )
    ssim_values: list[float] = []
    psnr_values: list[float] = []
    lpips_values: list[float] = []
    frame_index = 0
    while frame_index < total_frames:
        official_ok, official_frame = official.read()
        omni_ok, omni_frame = omni.read()
        if official_ok != omni_ok:
            raise ValueError(f"Frame count diverged at frame {frame_index}")
        if not official_ok:
            break
        ssim_values.append(
            float(
                structural_similarity(
                    official_frame, omni_frame, channel_axis=2, data_range=255
                )
            )
        )
        psnr_values.append(
            float(peak_signal_noise_ratio(official_frame, omni_frame, data_range=255))
        )
        if frame_index in sampled_indices:
            lpips_value = lpips_model(
                lpips_tensor(official_frame, device),
                lpips_tensor(omni_frame, device),
            )
            lpips_values.append(float(lpips_value.item()))
        frame_index += 1
    official.release()
    omni.release()
    if frame_index != total_frames:
        raise ValueError(
            f"Decoded {frame_index} frames, ffprobe reported {total_frames}"
        )

    finite_psnr = [value for value in psnr_values if math.isfinite(value)]
    return {
        **official_meta,
        "evaluated_frame_count": total_frames,
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_min": float(np.min(ssim_values)),
        "psnr_mean_db": float(np.mean(finite_psnr)) if finite_psnr else "inf",
        "psnr_min_db": float(np.min(finite_psnr)) if finite_psnr else "inf",
        "lpips_mean": float(np.mean(lpips_values)),
        "lpips_max": float(np.max(lpips_values)),
        "lpips_sampled_frames": [int(index) for index in sorted(sampled_indices)],
        "official": official_path.name,
        "omni": omni_path.name,
    }


def write_pair_result(
    metrics_path: Path, metric_contract: dict, pair_id: str, pair_result: dict
) -> dict:
    lock_path = metrics_path.with_suffix(metrics_path.suffix + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = {"metric_contract": metric_contract, "pairs": {}}
        if metrics_path.exists():
            current = json.loads(metrics_path.read_text())
            current["metric_contract"] = metric_contract
            current.setdefault("pairs", {})
        current["pairs"][pair_id] = pair_result
        temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        temporary.write_text(json.dumps(current, indent=2) + "\n")
        temporary.replace(metrics_path)
        return current


def main() -> None:
    args = parse_args()
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)

    import lpips

    lpips_model = lpips.LPIPS(net="alex").eval().to(device)
    grouped: dict[tuple[str, str, int], dict[str, Path]] = {}
    for path in sorted(args.results_dir.glob("*.mp4")):
        match = FILENAME.match(path.name)
        if match is None:
            continue
        seed = int(match["seed"])
        if args.mode is not None and match["mode"] != args.mode:
            continue
        if args.modality is not None and match["modality"] != args.modality:
            continue
        if args.seeds is not None and seed not in args.seeds:
            continue
        key = (match["mode"], match["modality"], seed)
        grouped.setdefault(key, {})[match["backend"]] = path

    metrics_path = args.results_dir / "metrics.json"
    previous_pairs = {}
    if metrics_path.exists() and not args.force:
        previous_pairs = json.loads(metrics_path.read_text()).get("pairs", {})

    metric_contract = {
        "ssim": f"first {args.max_frames} decoded BGR frames (channel permutation invariant)",
        "psnr": f"finite per-frame values across the first {args.max_frames} decoded frames",
        "lpips": f"AlexNet, {args.lpips_frames} uniformly sampled decoded frames",
        "comparison": (
            "official raw split artifacts vs the corresponding Diffusers "
            "materialization; same prompt, seed, shape, schedule, ConvVAE, and CRF 18"
        ),
    }
    results = {"metric_contract": metric_contract, "pairs": previous_pairs}
    for (mode, modality, seed), paths in grouped.items():
        if set(paths) != {"official", "omni"}:
            continue
        pair_id = f"{mode}-{modality}-seed-{seed}"
        previous = results["pairs"].get(pair_id)
        if (
            not args.force
            and previous is not None
            and previous.get("official") == paths["official"].name
            and previous.get("omni") == paths["omni"].name
        ):
            print(f"[{pair_id}] already computed", flush=True)
            continue
        print(f"[{pair_id}] computing", flush=True)
        pair_result = compare_pair(
            paths["official"],
            paths["omni"],
            lpips_model=lpips_model,
            lpips_frames=args.lpips_frames,
            max_frames=args.max_frames,
            device=device,
        )
        results = write_pair_result(metrics_path, metric_contract, pair_id, pair_result)

    if not results["pairs"]:
        raise SystemExit(f"No complete official/Omni pairs found in {args.results_dir}")


if __name__ == "__main__":
    main()

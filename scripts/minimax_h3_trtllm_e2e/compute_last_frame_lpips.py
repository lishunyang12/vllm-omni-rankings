#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import lpips
import torch


def read_last_frame(path: Path) -> tuple[torch.Tensor, int]:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"No frames found in {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode the last frame of {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    return tensor.div(127.5).sub(1).unsqueeze(0), frame_count


def selected_video(summary_path: Path) -> Path:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for run in summary["runs"]:
        if not run.get("warmup", False) and run.get("mp4"):
            path = Path(run["mp4"])
            if path.is_file():
                return path
    raise RuntimeError(f"No measured video found in {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-mode", default="trtllm_dense")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    run_config = json.loads(
        (args.session_root / "run_config.json").read_text(encoding="utf-8")
    )
    modes = list(run_config["modes"])
    if args.reference_mode not in modes:
        raise ValueError(f"Reference mode {args.reference_mode!r} is not in {modes}")

    videos = {
        mode: selected_video(args.session_root / mode / "summary.json")
        for mode in modes
    }
    torch.set_num_threads(args.threads)
    metric = lpips.LPIPS(net="alex").eval()
    reference, reference_frames = read_last_frame(videos[args.reference_mode])

    results: dict[str, dict[str, object]] = {}
    for mode in modes:
        candidate, frame_count = read_last_frame(videos[mode])
        if frame_count != reference_frames:
            raise RuntimeError(
                f"Frame count mismatch for {mode}: {frame_count} != {reference_frames}"
            )
        with torch.inference_mode():
            score = float(metric(reference, candidate).item())
        results[mode] = {
            "video": str(videos[mode]),
            "last_frame_index": frame_count - 1,
            "lpips": score,
        }

    payload = {
        "metric": "LPIPS",
        "network": "alex",
        "input": "last decoded RGB frame, normalized to [-1, 1]",
        "reference_mode": args.reference_mode,
        "reference_video": str(videos[args.reference_mode]),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({mode: value["lpips"] for mode, value in results.items()}))


if __name__ == "__main__":
    main()

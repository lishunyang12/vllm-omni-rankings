#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch


def _video_info(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    result = (
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    return result


def _read_batch(
    capture: cv2.VideoCapture, batch_size: int
) -> tuple[torch.Tensor, np.ndarray] | None:
    frames = []
    while len(frames) < batch_size:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not frames:
        return None
    array = np.stack(frames)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float().div(127.5).sub(1)
    return tensor, array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    reference_info = _video_info(args.reference)
    candidate_info = _video_info(args.candidate)
    if reference_info != candidate_info:
        raise RuntimeError(
            f"Video metadata differs: {reference_info} != {candidate_info}"
        )

    device = torch.device(args.device)
    metric = lpips.LPIPS(net="alex").eval().to(device)
    reference_capture = cv2.VideoCapture(str(args.reference))
    candidate_capture = cv2.VideoCapture(str(args.candidate))
    lpips_scores: list[float] = []
    mean_absolute_errors: list[float] = []
    psnr_scores: list[float] = []
    try:
        while True:
            reference_batch = _read_batch(reference_capture, args.batch_size)
            candidate_batch = _read_batch(candidate_capture, args.batch_size)
            if reference_batch is None or candidate_batch is None:
                if reference_batch is not None or candidate_batch is not None:
                    raise RuntimeError("Videos ended at different frames")
                break
            reference_tensor, reference_rgb = reference_batch
            candidate_tensor, candidate_rgb = candidate_batch
            if reference_tensor.shape != candidate_tensor.shape:
                raise RuntimeError(
                    f"Decoded batch shapes differ: {reference_tensor.shape} != {candidate_tensor.shape}"
                )
            with torch.inference_mode():
                scores = metric(
                    reference_tensor.to(device),
                    candidate_tensor.to(device),
                )
            lpips_scores.extend(float(score) for score in scores.flatten().cpu())

            difference = reference_rgb.astype(np.float32) - candidate_rgb.astype(
                np.float32
            )
            mean_absolute_errors.extend(
                np.abs(difference).mean(axis=(1, 2, 3)).tolist()
            )
            for mse in np.square(difference).mean(axis=(1, 2, 3)):
                psnr_scores.append(
                    math.inf if mse == 0 else 10 * math.log10(255**2 / float(mse))
                )
    finally:
        reference_capture.release()
        candidate_capture.release()

    if len(lpips_scores) != reference_info[0]:
        raise RuntimeError(
            f"Decoded {len(lpips_scores)} frames, expected {reference_info[0]}"
        )

    payload = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "frames": reference_info[0],
        "fps": reference_info[1],
        "width": reference_info[2],
        "height": reference_info[3],
        "lpips_alex": {
            "mean": statistics.fmean(lpips_scores),
            "median": statistics.median(lpips_scores),
            "p95": float(np.percentile(lpips_scores, 95)),
            "max": max(lpips_scores),
            "last": lpips_scores[-1],
        },
        "decoded_rgb": {
            "mean_absolute_error": statistics.fmean(mean_absolute_errors),
            "mean_psnr_db": statistics.fmean(psnr_scores),
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

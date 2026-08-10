#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DIFFUSE_KEY = "MiniMaxH3Pipeline.diffuse"


def summarize(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    measured = [run for run in payload["runs"] if not run["warmup"]]
    values = [float(run["stage_durations"][DIFFUSE_KEY]) for run in measured]
    mean = statistics.fmean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "summary": str(path),
        "async_ulysses": bool(payload["async_ulysses"]),
        "values_s": values,
        "median_s": median,
        "mean_s": mean,
        "stdev_s": stdev,
        "cv": stdev / mean,
        "span_over_median": (max(values) - min(values)) / median,
        "stable": len(values) >= 5
        and stdev / mean <= 0.02
        and (max(values) - min(values)) / median <= 0.05,
        "peak_memory_mb": max(
            float(run["worker_peak_memory_mb"]) for run in payload["runs"]
        ),
        "warmup_frame_hash": payload["runs"][0]["frames_sha256"],
        "measured_frame_hashes": sorted({run["frames_sha256"] for run in measured}),
        "measured_audio_hashes": sorted({run["audio_sha256"] for run in measured}),
        "video": payload["runs"][1]["mp4"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dense = summarize(args.session_root / "dense" / "summary.json")
    async_result = summarize(args.session_root / "async" / "summary.json")
    result = {
        "dense": dense,
        "async": async_result,
        "speedup": float(dense["median_s"]) / float(async_result["median_s"]),
        "both_stable": bool(dense["stable"] and async_result["stable"]),
        "measured_frame_hash_match": dense["measured_frame_hashes"]
        == async_result["measured_frame_hashes"],
        "measured_audio_hash_match": dense["measured_audio_hashes"]
        == async_result["measured_audio_hashes"],
    }
    output = args.output or args.session_root / "results.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

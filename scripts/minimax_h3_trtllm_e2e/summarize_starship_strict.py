#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DIFFUSE_KEY = "MiniMaxH3Pipeline.diffuse"
MODES = (
    "trtllm_dense",
    "sage_fp8",
    "skip_softmax_005_gate097",
    "sage_fp8_skip_005_gate097",
)


def timing_stats(path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    values = [
        float(run["stage_durations"][DIFFUSE_KEY])
        for run in summary["runs"]
        if not run.get("warmup", False)
    ]
    mean = statistics.fmean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = stdev / mean
    span = (max(values) - min(values)) / median
    deterministic = bool(summary["steady_output_deterministic"])
    return {
        "summary": str(path),
        "values_s": values,
        "count": len(values),
        "mean_s": mean,
        "median_s": median,
        "cv": cv,
        "span_over_median": span,
        "deterministic": deterministic,
        "stable": len(values) >= 5 and cv <= 0.02 and span <= 0.05 and deterministic,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_root", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    results = {
        mode: timing_stats(args.session_root / mode / "summary.json") for mode in MODES
    }
    dense_median = float(results["trtllm_dense"]["median_s"])
    for result in results.values():
        result["speedup"] = dense_median / float(result["median_s"])

    thermal = json.loads(
        (args.session_root / "thermal_audit.json").read_text(encoding="utf-8")
    )
    accepted = all(bool(result["stable"]) for result in results.values()) and bool(
        thermal["accepted"]
    )
    payload = {
        "criteria": {
            "minimum_measured_runs": 5,
            "maximum_cv": 0.02,
            "maximum_span_over_median": 0.05,
            "deterministic_output": True,
            "thermal_audit": "pass",
        },
        "thermal_accepted": bool(thermal["accepted"]),
        "accepted": accepted,
        "results": results,
    }
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    labels = {
        "trtllm_dense": "TRTLLM dense",
        "sage_fp8": "FP8 SAGE",
        "skip_softmax_005_gate097": "Skip-Softmax 0.05/0.97",
        "sage_fp8_skip_005_gate097": "FP8 SAGE + Skip-Softmax 0.05/0.97",
    }
    lines = [
        "| Mode | Median diffuse | Speedup | CV | Span/median |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        result = results[mode]
        lines.append(
            f"| {labels[mode]} "
            f"| {float(result['median_s']):.3f} s "
            f"| {float(result['speedup']):.3f}× "
            f"| {100 * float(result['cv']):.2f}% "
            f"| {100 * float(result['span_over_median']):.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Strict timing qualification: {'pass' if accepted else 'fail'}.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("pass" if accepted else "fail")


if __name__ == "__main__":
    main()

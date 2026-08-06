#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DIFFUSE_KEY = "MiniMaxH3Pipeline.diffuse"
MAX_ISOLATED_MEDIAN_S = 85.0


def timing_stats(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [
        float(run["stage_durations"][DIFFUSE_KEY])
        for run in payload["runs"]
        if not run.get("warmup", False)
    ]
    mean = statistics.fmean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    result: dict[str, object] = {
        "summary": str(path),
        "mode": payload["mode"],
        "values_s": values,
        "count": len(values),
        "min_s": min(values),
        "max_s": max(values),
        "mean_s": mean,
        "median_s": median,
        "stdev_s": stdev,
        "cv": stdev / mean,
        "span_over_median": (max(values) - min(values)) / median,
    }
    result["stable"] = (
        result["count"] >= 5
        and result["cv"] <= 0.02
        and result["span_over_median"] <= 0.05
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isolated-g0", type=Path, required=True)
    parser.add_argument("--isolated-g1", type=Path, required=True)
    parser.add_argument("--concurrent-g0", type=Path, required=True)
    parser.add_argument("--concurrent-g1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {
        "isolated_g0": timing_stats(args.isolated_g0),
        "isolated_g1": timing_stats(args.isolated_g1),
        "concurrent_g0": timing_stats(args.concurrent_g0),
        "concurrent_g1": timing_stats(args.concurrent_g1),
    }
    med = {name: float(stats["median_s"]) for name, stats in runs.items()}
    group_delta = abs(med["isolated_g0"] - med["isolated_g1"]) / statistics.fmean(
        [med["isolated_g0"], med["isolated_g1"]]
    )
    concurrent_delta = {
        group: med[f"concurrent_{group}"] / med[f"isolated_{group}"] - 1.0
        for group in ("g0", "g1")
    }
    isolated_stable = {
        group: bool(runs[f"isolated_{group}"]["stable"])
        and med[f"isolated_{group}"] <= MAX_ISOLATED_MEDIAN_S
        for group in ("g0", "g1")
    }
    all_stable = all(bool(stats["stable"]) for stats in runs.values())
    parallel_ok = (
        all_stable
        and all(isolated_stable.values())
        and group_delta <= 0.02
        and all(abs(delta) <= 0.02 for delta in concurrent_delta.values())
    )
    if parallel_ok:
        decision = "parallel"
    else:
        stable_groups = [group for group, stable in isolated_stable.items() if stable]
        decision = (
            f"serial_{min(stable_groups, key=lambda group: med[f'isolated_{group}'])}"
            if stable_groups
            else "abort"
        )

    result = {
        "criteria": {
            "minimum_measured_runs": 5,
            "maximum_cv": 0.02,
            "maximum_span_over_median": 0.05,
            "maximum_group_delta": 0.02,
            "maximum_concurrent_delta": 0.02,
            "maximum_isolated_median_s": MAX_ISOLATED_MEDIAN_S,
            "serial_group_selection": "lowest isolated median among stable groups",
        },
        "runs": runs,
        "isolated_group_delta": group_delta,
        "concurrent_delta": concurrent_delta,
        "decision": decision,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(decision)


if __name__ == "__main__":
    main()

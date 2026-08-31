#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_timing import MAX_ISOLATED_MEDIAN_S, timing_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("g0", "g1"), required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stats = timing_stats(args.summary)
    accepted = (
        bool(stats["stable"]) and float(stats["median_s"]) <= MAX_ISOLATED_MEDIAN_S
    )
    decision = f"serial_{args.group}" if accepted else "abort"
    result = {
        "criteria": {
            "minimum_measured_runs": 5,
            "maximum_cv": 0.02,
            "maximum_span_over_median": 0.05,
            "maximum_isolated_median_s": MAX_ISOLATED_MEDIAN_S,
        },
        "runs": {f"isolated_{args.group}": stats},
        "isolated_group_delta": None,
        "concurrent_delta": {},
        "decision": decision,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(decision)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DIFFUSE_KEY = "MiniMaxH3Pipeline.diffuse"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("g0", "g1"), required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--maximum", type=float, default=85.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    values = [
        float(run["stage_durations"][DIFFUSE_KEY])
        for run in payload["runs"]
        if not run.get("warmup", False)
    ]
    if len(values) != 1:
        raise RuntimeError(
            f"Expected one measured run in {args.summary}, found {len(values)}"
        )

    accepted = values[0] <= args.maximum
    result = {
        "maximum_measured_diffuse_s": args.maximum,
        "measured_diffuse_s": values[0],
        "group": args.group,
        "accepted": accepted,
        "selection": args.group if accepted else "none",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["selection"])


if __name__ == "__main__":
    main()

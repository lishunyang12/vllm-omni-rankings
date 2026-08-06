#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DIFFUSE_KEY = "MiniMaxH3Pipeline.diffuse"


def measured_diffuse(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [
        float(run["stage_durations"][DIFFUSE_KEY])
        for run in payload["runs"]
        if not run.get("warmup", False)
    ]
    if len(values) != 1:
        raise RuntimeError(f"Expected one measured run in {path}, found {len(values)}")
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--maximum-median", type=float, default=85.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    values = {"g0": measured_diffuse(args.g0), "g1": measured_diffuse(args.g1)}
    viable = [group for group, value in values.items() if value <= args.maximum_median]
    selection = "both" if len(viable) == 2 else viable[0] if viable else "none"
    result = {
        "maximum_measured_diffuse_s": args.maximum_median,
        "measured_diffuse_s": values,
        "viable_groups": viable,
        "selection": selection,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(selection)


if __name__ == "__main__":
    main()

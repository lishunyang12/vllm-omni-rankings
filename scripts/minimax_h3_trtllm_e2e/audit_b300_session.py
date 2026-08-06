#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DIFFUSE_KEY = "MiniMaxH3Pipeline.diffuse"
THRESHOLDS = ("005", "010", "03", "05")
GATES = ("090", "095", "099")
REQUIRED_MODES = (
    "recipe_flash",
    "trtllm_dense",
    "sage_fp8",
    *(
        f"{prefix}_{threshold}_gate{gate}"
        for prefix in ("skip_softmax", "sage_fp8_skip")
        for gate in GATES
        for threshold in THRESHOLDS
    ),
)


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
    return {
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
        "stable": len(values) >= 5
        and stdev / mean <= 0.02
        and (max(values) - min(values)) / median <= 0.05,
    }


def candidate_priority(
    candidate: dict[str, object], preferred_group: str
) -> tuple[bool, int, float]:
    path = str(candidate["summary"])
    if "/reruns/" in path:
        source = 4
    elif "/final/" in path:
        source = 3
    elif f"/qualification/isolated_{preferred_group}/" in path:
        source = 2
    elif "/qualification/isolated_" in path:
        source = 1
    else:
        source = 0
    return bool(candidate["stable"]), source, -float(candidate["cv"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates: dict[str, list[dict[str, object]]] = {}
    errors: list[dict[str, str]] = []
    for path in sorted(args.session_root.rglob("summary.json")):
        path_text = str(path)
        if ".incomplete." in path_text or "/kernel_profiles/" in path_text:
            continue
        try:
            stats = timing_stats(path)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append({"summary": path_text, "error": str(exc)})
            continue
        candidates.setdefault(str(stats["mode"]), []).append(stats)

    qualification = json.loads(
        (args.session_root / "qualification.json").read_text(encoding="utf-8")
    )
    decision = qualification["decision"]
    preferred_group = "g1" if decision == "serial_g1" else "g0"

    modes: dict[str, dict[str, object]] = {}
    needs_rerun: list[str] = []
    for mode in REQUIRED_MODES:
        mode_candidates = candidates.get(mode, [])
        selected = (
            max(
                mode_candidates,
                key=lambda candidate: candidate_priority(candidate, preferred_group),
            )
            if mode_candidates
            else None
        )
        modes[mode] = {
            "selected": selected,
            "candidates": mode_candidates,
        }
        if selected is None or not selected["stable"]:
            needs_rerun.append(mode)

    result = {
        "criteria": {
            "minimum_measured_runs": 5,
            "maximum_cv": 0.02,
            "maximum_span_over_median": 0.05,
        },
        "required_modes": list(REQUIRED_MODES),
        "qualification_decision": decision,
        "preferred_group": preferred_group,
        "modes": modes,
        "needs_rerun": needs_rerun,
        "complete_and_stable": not needs_rerun,
        "errors": errors,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"needs_rerun": needs_rerun, "output": str(args.output)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested = {int(value) for value in args.gpus.split(",")}
    samples: dict[int, list[dict[str, object]]] = {index: [] for index in requested}
    with args.telemetry.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) != 8:
                continue
            index = int(row[1])
            if index not in requested:
                continue
            samples[index].append(
                {
                    "timestamp": row[0].strip(),
                    "temperature_c": int(row[2]),
                    "utilization_percent": int(row[3]),
                    "sm_clock_mhz": int(row[4]),
                    "power_w": float(row[5]),
                    "thermal_slowdown_active": row[6].strip() == "Active",
                    "thermal_slowdown_counter_us": int(row[7]),
                }
            )

    missing = sorted(index for index, values in samples.items() if not values)
    if missing:
        raise RuntimeError(f"No telemetry samples for GPU(s): {missing}")

    gpu_results: dict[str, dict[str, object]] = {}
    accepted = True
    for index, values in sorted(samples.items()):
        counters = [int(value["thermal_slowdown_counter_us"]) for value in values]
        loaded = [value for value in values if int(value["utilization_percent"]) >= 90]
        counter_delta = max(counters) - min(counters)
        active_samples = sum(bool(value["thermal_slowdown_active"]) for value in values)
        gpu_accepted = counter_delta == 0 and active_samples == 0 and bool(loaded)
        accepted &= gpu_accepted
        gpu_results[str(index)] = {
            "sample_count": len(values),
            "loaded_sample_count": len(loaded),
            "maximum_temperature_c": max(
                int(value["temperature_c"]) for value in values
            ),
            "minimum_loaded_sm_clock_mhz": min(
                (int(value["sm_clock_mhz"]) for value in loaded), default=None
            ),
            "thermal_slowdown_active_samples": active_samples,
            "thermal_slowdown_counter_delta_us": counter_delta,
            "accepted": gpu_accepted,
        }

    result = {
        "criteria": {
            "thermal_slowdown_counter_delta_us": 0,
            "thermal_slowdown_active_samples": 0,
            "minimum_loaded_samples": 1,
        },
        "gpus": gpu_results,
        "accepted": accepted,
        "decision": "pass" if accepted else "fail",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["decision"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DIFFUSE_KEY = "MiniMaxH3Pipeline.diffuse"
MODE_NAMES = ("sync_ulysses", "async_ulysses")
COPY_NAMES_FOR_OVERLAP = {
    "[CUDA memcpy Device-to-Device]",
    "[CUDA memcpy Peer-to-Peer]",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing {key!r} in CSV row: {row}")
    return float(value.replace(",", ""))


def _int(row: dict[str, str], key: str) -> int:
    return int(_float(row, key))


def _diffuse_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [
        float(run["stage_durations"][DIFFUSE_KEY])
        for run in payload["runs"]
        if not run["warmup"]
    ]
    if not values:
        raise ValueError(f"No measured diffuse run in {path}")
    return {
        "values_s": values,
        "median_s": statistics.median(values),
    }


def _nccl_summary(path: Path) -> dict[str, object]:
    rows = [row for row in _read_csv(path) if "nccl" in row.get("Name", "").lower()]
    return {
        "count": sum(_int(row, "Instances") for row in rows),
        "time_ms": sum(_int(row, "Total Time (ns)") for row in rows) / 1e6,
        "kernel_types": len(rows),
    }


def _named_kernel_summary(path: Path, name_fragment: str) -> dict[str, object]:
    rows = [row for row in _read_csv(path) if name_fragment in row.get("Name", "")]
    return {
        "count": sum(_int(row, "Instances") for row in rows),
        "time_ms": sum(_int(row, "Total Time (ns)") for row in rows) / 1e6,
    }


def _memcpy_summary(time_path: Path, size_path: Path) -> dict[str, object]:
    time_rows = {
        row["Operation"]: row
        for row in _read_csv(time_path)
        if "CUDA memcpy" in row.get("Operation", "")
    }
    size_rows = {
        row["Operation"]: row
        for row in _read_csv(size_path)
        if "CUDA memcpy" in row.get("Operation", "")
    }

    operations = sorted(time_rows.keys() | size_rows.keys())
    breakdown: dict[str, dict[str, object]] = {}
    for operation in operations:
        time_row = time_rows.get(operation)
        size_row = size_rows.get(operation)
        time_count = _int(time_row, "Count") if time_row else 0
        size_count = _int(size_row, "Count") if size_row else 0
        if time_row and size_row and time_count != size_count:
            raise ValueError(
                f"Memcpy count mismatch for {operation}: "
                f"time={time_count}, size={size_count}"
            )
        breakdown[operation] = {
            "count": time_count or size_count,
            "time_ms": (_int(time_row, "Total Time (ns)") / 1e6 if time_row else 0.0),
            # Nsight exports transfer volume in decimal MB rounded to 0.001 MB.
            "bytes_mb": _float(size_row, "Total (MB)") if size_row else 0.0,
        }

    return {
        "count": sum(int(item["count"]) for item in breakdown.values()),
        "time_ms": sum(float(item["time_ms"]) for item in breakdown.values()),
        "bytes_mb": sum(float(item["bytes_mb"]) for item in breakdown.values()),
        "breakdown": breakdown,
    }


Interval = tuple[int, int]


def _merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _duration(intervals: Iterable[Interval]) -> int:
    return sum(end - start for start, end in intervals)


def _intersection_duration(left: list[Interval], right: list[Interval]) -> int:
    total = 0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _copy_compute_overlap(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"available": False, "reason": f"missing {path.name}"}

    rows = _read_csv(path)
    required = {"Start (ns)", "Duration (ns)", "Device", "GrdX", "Name"}
    if not rows:
        return {"available": False, "reason": "empty cuda_gpu_trace"}
    missing = required - rows[0].keys()
    if missing:
        return {
            "available": False,
            "reason": f"cuda_gpu_trace missing columns: {sorted(missing)}",
        }

    copies: dict[str, dict[str, list[Interval]]] = defaultdict(
        lambda: defaultdict(list)
    )
    compute: dict[str, list[Interval]] = defaultdict(list)
    for row in rows:
        device = row["Device"]
        if not device or not row["Start (ns)"] or not row["Duration (ns)"]:
            continue
        start = _int(row, "Start (ns)")
        end = start + _int(row, "Duration (ns)")
        name = row["Name"]
        if name in COPY_NAMES_FOR_OVERLAP:
            copies[device][name].append((start, end))
        elif row["GrdX"] and "nccl" not in name.lower():
            # A populated launch grid distinguishes kernels from memory ops.
            compute[device].append((start, end))

    per_device: dict[str, dict[str, float]] = {}
    total_copy_ns = 0
    total_compute_ns = 0
    total_overlap_ns = 0
    for device in sorted(copies.keys() | compute.keys()):
        copy_intervals = _merge_intervals(
            interval
            for operation_intervals in copies[device].values()
            for interval in operation_intervals
        )
        compute_intervals = _merge_intervals(compute[device])
        copy_ns = _duration(copy_intervals)
        compute_ns = _duration(compute_intervals)
        overlap_ns = _intersection_duration(copy_intervals, compute_intervals)
        total_copy_ns += copy_ns
        total_compute_ns += compute_ns
        total_overlap_ns += overlap_ns
        per_device[device] = {
            "copy_busy_ms": copy_ns / 1e6,
            "non_nccl_compute_busy_ms": compute_ns / 1e6,
            "overlap_ms": overlap_ns / 1e6,
            "copy_hidden_pct": 100.0 * overlap_ns / copy_ns if copy_ns else 0.0,
        }

    if not total_copy_ns:
        return {
            "available": False,
            "reason": "no DtoD or peer-to-peer copy intervals in cuda_gpu_trace",
        }
    by_operation: dict[str, dict[str, float]] = {}
    for operation in sorted(COPY_NAMES_FOR_OVERLAP):
        operation_copy_ns = 0
        operation_overlap_ns = 0
        for device in sorted(copies.keys() | compute.keys()):
            operation_intervals = _merge_intervals(copies[device][operation])
            compute_intervals = _merge_intervals(compute[device])
            operation_copy_ns += _duration(operation_intervals)
            operation_overlap_ns += _intersection_duration(
                operation_intervals, compute_intervals
            )
        if operation_copy_ns:
            by_operation[operation] = {
                "copy_busy_ms": operation_copy_ns / 1e6,
                "overlap_ms": operation_overlap_ns / 1e6,
                "copy_hidden_pct": 100.0 * operation_overlap_ns / operation_copy_ns,
            }

    return {
        "available": True,
        "method": (
            "union of DtoD/P2P copy intervals intersected with the union of "
            "non-NCCL kernel intervals on each Nsight Device"
        ),
        "aggregate_device_copy_busy_ms": total_copy_ns / 1e6,
        "aggregate_device_non_nccl_compute_busy_ms": total_compute_ns / 1e6,
        "aggregate_device_overlap_ms": total_overlap_ns / 1e6,
        "copy_hidden_pct": 100.0 * total_overlap_ns / total_copy_ns,
        "by_operation": by_operation,
        "per_device": per_device,
    }


def _mode_summary(root: Path, mode: str) -> dict[str, object]:
    mode_root = root / mode
    kernel_summary = mode_root / "stats_cuda_gpu_kern_sum.csv"
    return {
        "diffuse": _diffuse_summary(mode_root / "benchmark" / "summary.json"),
        "nccl_kernels": _nccl_summary(kernel_summary),
        "symmetric_memory_barriers": _named_kernel_summary(
            kernel_summary, "c10d::symmetric_memory::barrier_kernel"
        ),
        "cuda_memcpy": _memcpy_summary(
            mode_root / "stats_cuda_gpu_mem_time_sum.csv",
            mode_root / "stats_cuda_gpu_mem_size_sum.csv",
        ),
        "copy_compute_overlap": _copy_compute_overlap(
            mode_root / "stats_cuda_gpu_trace.csv"
        ),
    }


def _pct_change(before: float, after: float) -> float | None:
    return 100.0 * (after - before) / before if before else None


def summarize(root: Path) -> dict[str, object]:
    modes = {mode: _mode_summary(root, mode) for mode in MODE_NAMES}
    sync = modes["sync_ulysses"]
    async_result = modes["async_ulysses"]
    sync_diffuse = float(sync["diffuse"]["median_s"])
    async_diffuse = float(async_result["diffuse"]["median_s"])
    comparison: dict[str, object] = {
        "diffuse_speedup_x": sync_diffuse / async_diffuse,
        "diffuse_time_change_pct": _pct_change(sync_diffuse, async_diffuse),
        "nccl_kernel_count_change": (
            int(async_result["nccl_kernels"]["count"])
            - int(sync["nccl_kernels"]["count"])
        ),
        "nccl_kernel_time_change_pct": _pct_change(
            float(sync["nccl_kernels"]["time_ms"]),
            float(async_result["nccl_kernels"]["time_ms"]),
        ),
        "symmetric_memory_barrier_count_change": (
            int(async_result["symmetric_memory_barriers"]["count"])
            - int(sync["symmetric_memory_barriers"]["count"])
        ),
        "cuda_memcpy_time_change_pct": _pct_change(
            float(sync["cuda_memcpy"]["time_ms"]),
            float(async_result["cuda_memcpy"]["time_ms"]),
        ),
        "cuda_memcpy_bytes_change_pct": _pct_change(
            float(sync["cuda_memcpy"]["bytes_mb"]),
            float(async_result["cuda_memcpy"]["bytes_mb"]),
        ),
    }
    sync_overlap = sync["copy_compute_overlap"]
    async_overlap = async_result["copy_compute_overlap"]
    if sync_overlap["available"] and async_overlap["available"]:
        comparison["copy_hidden_change_percentage_points"] = float(
            async_overlap["copy_hidden_pct"]
        ) - float(sync_overlap["copy_hidden_pct"])

    return {
        "session_root": str(root.resolve()),
        "modes": modes,
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sync and async Ulysses Nsight Systems captures."
    )
    parser.add_argument("session_root", type=Path)
    parser.add_argument("--output", type=Path, help="Also write the JSON result here")
    args = parser.parse_args()

    result = summarize(args.session_root)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

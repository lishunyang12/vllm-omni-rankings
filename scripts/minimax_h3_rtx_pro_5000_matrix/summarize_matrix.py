#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert MiniMax-H3 case artifacts into the requested comparison matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

COLUMNS = (
    "GPU Model",
    "Node / Cluster",
    "GPU Count",
    "Workload",
    "Precision",
    "Parallelism",
    "Offload",
    "Denoise Steps",
    "E2E (s)",
    "Text Encode (s)",
    "Visual Encode (s)",
    "Audio Encode (s)",
    "Latent (s)",
    "Denoise (s)",
    "VAE Decode (s)",
    "Per Step (ms)",
    "Peak Memory (GiB)",
    "Peak Memory Scope",
    "SM Clock (MHz)",
    "Status",
    "Notes",
)

TASKS = {
    "t2va": "t2va",
    "fl2va_first_frame": "fl2va",
    "ref2va_image_audio": "ref2va",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def rounded(value: float | None, digits: int = 2) -> float | str:
    return "n/a" if value is None else round(value, digits)


def stage_value(
    stages: dict[str, Any],
    *,
    exact: tuple[str, ...] = (),
    contains: tuple[str, ...] = (),
) -> float | None:
    for key in exact:
        value = finite_float(stages.get(key))
        if value is not None:
            return value
    matches = []
    for key, raw_value in stages.items():
        lowered = key.lower()
        if any(token in lowered for token in contains):
            value = finite_float(raw_value)
            if value is not None:
                matches.append(value)
    return sum(matches) if matches else None


def telemetry(
    case_dir: Path, selected_gpu_ids: set[int]
) -> tuple[str | None, float | None, float | None]:
    path = case_dir / "gpu_telemetry.csv"
    if not path.is_file():
        return None, None, None
    gpu_model = None
    peak_memory_mib = None
    max_sm_clock = None
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        for row in csv.reader(stream):
            if len(row) < 7:
                continue
            try:
                gpu_index = int(row[1].strip())
            except ValueError:
                continue
            if gpu_index not in selected_gpu_ids:
                continue
            gpu_model = gpu_model or row[2].strip()
            memory = finite_float(row[4].strip())
            clock = finite_float(row[6].strip())
            if memory is not None:
                peak_memory_mib = (
                    memory if peak_memory_mib is None else max(peak_memory_mib, memory)
                )
            if clock is not None:
                max_sm_clock = (
                    clock if max_sm_clock is None else max(max_sm_clock, clock)
                )
    return gpu_model, peak_memory_mib, max_sm_clock


def precision_label(mode: str) -> str:
    return {
        "bf16": "BF16",
        "fp8": "FP8 (online DiT linear)",
        "fp8_sm120_attn": "FP8 (linear + SM120 attention)",
    }.get(mode, mode)


def status_for(case_dir: Path) -> str:
    path = case_dir / "status.txt"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "Incomplete"


def task_metrics(record: dict[str, Any], denoise_steps: int) -> dict[str, float | None]:
    stages = record.get("stage_durations") or {}
    denoise = stage_value(stages, exact=("MiniMaxH3Pipeline.diffuse",))
    return {
        "e2e": finite_float(record.get("wall_time_s")),
        "text": stage_value(stages, exact=("MiniMaxH3Pipeline.encode_prompt",)),
        "visual": stage_value(
            stages,
            contains=(
                "encode_video",
                "encode_visual",
                "encode_image",
                "video_audio_conditions",
            ),
        ),
        "audio": stage_value(stages, contains=("encode_audio", "audio_encode")),
        "latent": stage_value(
            stages, contains=("prepare_latent", "latent_prepare", "create_latent")
        ),
        "denoise": denoise,
        "decode": stage_value(
            stages, exact=("MiniMaxH3Pipeline.decode",), contains=("vae_decode",)
        ),
        "per_step": None
        if denoise is None or denoise_steps <= 0
        else denoise * 1000.0 / denoise_steps,
    }


def rows_for_case(case_dir: Path, matrix_env: dict[str, str]) -> list[dict[str, Any]]:
    case_env = read_env(case_dir / "case.env")
    if not case_env:
        return []
    status = status_for(case_dir)
    gpu_count = int(case_env["gpu_count"])
    selected_gpu_ids = {int(value) for value in case_env["gpu_ids"].split(",")}
    gpu_model, peak_mib, sm_clock = telemetry(case_dir, selected_gpu_ids)
    mode = case_env["mode"]
    requested_steps = int(matrix_env.get("steps_requested", "50"))
    # MiniMax-H3's scheduler performs one fewer denoise update than the requested value.
    denoise_steps = max(requested_steps - 1, 1)

    summary_path = case_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    records = {record.get("task_id"): record for record in summary.get("tasks", [])}
    if gpu_model is None:
        hardware = summary.get("hardware") or []
        gpu_model = hardware[0].get("name") if hardware else None
    if not gpu_model or gpu_model == "NVIDIA Graphics Device":
        gpu_model = case_env.get("gpu_model_label", "RTX PRO 5000 Blackwell")

    common_notes = [
        f"requested {requested_steps} / executed {denoise_steps} denoise updates",
        f"attention={summary.get('attention_backend', case_env.get('attention_mode'))}",
        f"repeat={case_env.get('repeat', '1')}",
        f"commit={matrix_env.get('code_commit', 'unknown')[:12]}",
    ]
    if case_env.get("attention_mode") == "sm120_prims":
        common_notes.append("experimental Tom-Zheng/flashinfer@4a2345906256")

    rows = []
    for task_id, workload in TASKS.items():
        record = records.get(task_id)
        metrics = (
            task_metrics(record, denoise_steps)
            if record is not None
            else defaultdict(lambda: None)
        )
        if (
            record is None
            and workload == "ref2va"
            and matrix_env.get("run_ref2va") == "0"
        ):
            row_status = "Skipped"
        elif record is None:
            row_status = status
        else:
            row_status = "Passed" if status == "Passed" else status
        notes = list(common_notes)
        if record is None:
            notes.append(f"missing task record {task_id}")
        if (
            workload == "fl2va"
            and metrics["visual"] is not None
            and metrics["audio"] is None
        ):
            notes.append(
                "visual column contains combined video/audio condition encode when the profiler does not split it"
            )
        rows.append(
            {
                "GPU Model": gpu_model,
                "Node / Cluster": case_env.get("node_label", "n/a"),
                "GPU Count": gpu_count,
                "Workload": workload,
                "Precision": precision_label(mode),
                "Parallelism": case_env["parallelism"],
                "Offload": case_env["offload"],
                "Denoise Steps": requested_steps,
                "E2E (s)": rounded(metrics["e2e"]),
                "Text Encode (s)": rounded(metrics["text"]),
                "Visual Encode (s)": rounded(metrics["visual"]),
                "Audio Encode (s)": rounded(metrics["audio"]),
                "Latent (s)": rounded(metrics["latent"]),
                "Denoise (s)": rounded(metrics["denoise"]),
                "VAE Decode (s)": rounded(metrics["decode"]),
                "Per Step (ms)": rounded(metrics["per_step"], 0),
                "Peak Memory (GiB)": rounded(
                    None if peak_mib is None else peak_mib / 1024.0, 2
                ),
                "Peak Memory Scope": "per-GPU max (external nvidia-smi)",
                "SM Clock (MHz)": rounded(sm_clock, 0),
                "Status": row_status,
                "Notes": "; ".join(notes),
            }
        )
    return rows


def numeric_median(rows: list[dict[str, Any]], column: str) -> float | str:
    values = [value for row in rows if (value := finite_float(row[column])) is not None]
    return "n/a" if not values else round(statistics.median(values), 2)


def aggregate_repeats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    identity_columns = (
        "GPU Model",
        "Node / Cluster",
        "GPU Count",
        "Workload",
        "Precision",
        "Parallelism",
        "Offload",
        "Denoise Steps",
    )
    for row in rows:
        groups[tuple(row[column] for column in identity_columns)].append(row)

    output = []
    latency_columns = (
        "E2E (s)",
        "Text Encode (s)",
        "Visual Encode (s)",
        "Audio Encode (s)",
        "Latent (s)",
        "Denoise (s)",
        "VAE Decode (s)",
        "Per Step (ms)",
    )
    for group_rows in groups.values():
        passed = [row for row in group_rows if row["Status"] == "Passed"]
        source = passed or group_rows
        result = {column: source[0][column] for column in identity_columns}
        for column in latency_columns:
            result[column] = numeric_median(source, column)
        peak_values = [
            value
            for row in source
            if (value := finite_float(row["Peak Memory (GiB)"])) is not None
        ]
        clock_values = [
            value
            for row in source
            if (value := finite_float(row["SM Clock (MHz)"])) is not None
        ]
        result["Peak Memory (GiB)"] = (
            "n/a" if not peak_values else round(max(peak_values), 2)
        )
        result["Peak Memory Scope"] = source[0]["Peak Memory Scope"]
        result["SM Clock (MHz)"] = (
            "n/a" if not clock_values else round(max(clock_values))
        )
        if passed:
            result["Status"] = "Passed"
        elif any(row["Status"] == "OOM" for row in group_rows):
            result["Status"] = "OOM"
        else:
            result["Status"] = group_rows[0]["Status"]
        result["Notes"] = (
            f"median latency across {len(passed)} passed repeat(s); max peak memory/clock; "
            + source[0]["Notes"]
        )
        output.append(result)
    return sorted(
        output,
        key=lambda row: (int(row["GPU Count"]), row["Precision"], row["Workload"]),
    )


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    matrix_env = read_env(result_root / "matrix.env")
    rows = []
    pattern = re.compile(r"g[1248]-.+-tp\d+-u\d+-run\d+$")
    for case_dir in sorted(
        path
        for path in result_root.iterdir()
        if path.is_dir() and pattern.fullmatch(path.name)
    ):
        rows.extend(rows_for_case(case_dir, matrix_env))
    rows = aggregate_repeats(rows)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    args.json_output.write_text(
        json.dumps({"metadata": matrix_env, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {args.csv_output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate MiniMax-H3 online-FP8 timing, memory, and fidelity results."""

from __future__ import annotations

import argparse
import array
import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


CASES = {
    "bf16": ("BF16", "BF16", 0),
    "dit_fp8": ("FP8", "BF16", 0),
    "te_mlp_l0_9": ("BF16", "MLP layers 0-9", 20),
    "te_mlp_l0_24": ("BF16", "MLP layers 0-24", 50),
    "te_mlp_all": ("BF16", "MLP layers 0-49", 100),
    "te_mlp_o_all": ("BF16", "MLP + attention O layers 0-49", 150),
    "te_all_linear": ("BF16", "all decoder linear layers", 200),
    "dit_te_mlp_all": ("FP8", "MLP layers 0-49", 100),
    "all_linear_fp8": ("FP8", "all decoder linear layers", 200),
}
TASKS = ("t2va", "i2va", "ref2va")


def run(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "command failed")
    return result.stdout + result.stderr


def probe(video: Path) -> dict[str, Any]:
    data = json.loads(
        run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=codec_type,width,height,r_frame_rate,sample_rate,channels",
            "-of", "json", str(video),
        ])
    )
    video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio_stream = next(
        (s for s in data["streams"] if s["codec_type"] == "audio"), {}
    )
    numerator, denominator = map(int, video_stream["r_frame_rate"].split("/"))
    return {
        "duration_s": float(data["format"]["duration"]),
        "size_bytes": int(data["format"]["size"]),
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": numerator / denominator,
        "audio_sample_rate": int(audio_stream.get("sample_rate", 0)),
        "audio_channels": int(audio_stream.get("channels", 0)),
    }


def decoded_audio(video: Path) -> array.array[float]:
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "2", "-ar", "32000", "-",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    samples = array.array("f")
    samples.frombytes(result.stdout)
    return samples


def audio_fidelity(reference: Path, candidate: Path) -> tuple[float | None, float | None]:
    try:
        left = decoded_audio(reference)
        right = decoded_audio(candidate)
        count = min(len(left), len(right))
        if not count:
            return None, None
        left_mean = sum(left[:count]) / count
        right_mean = sum(right[:count]) / count
        signal_power = 0.0
        noise_power = 0.0
        covariance = 0.0
        left_variance = 0.0
        right_variance = 0.0
        for left_value, right_value in zip(left[:count], right[:count]):
            signal_power += left_value * left_value
            delta = left_value - right_value
            noise_power += delta * delta
            left_centered = left_value - left_mean
            right_centered = right_value - right_mean
            covariance += left_centered * right_centered
            left_variance += left_centered * left_centered
            right_variance += right_centered * right_centered
        snr = (
            10.0 * math.log10(signal_power / noise_power)
            if noise_power > 0.0 else float("inf")
        )
        denominator = math.sqrt(left_variance * right_variance)
        correlation = covariance / denominator if denominator > 0.0 else None
        return snr, correlation
    except (RuntimeError, ValueError):
        return None, None


def fidelity(
    reference: Path,
    candidate: Path,
) -> tuple[float | None, float | None, float | None, float | None]:
    if reference == candidate:
        return None, 1.0, None, 1.0
    try:
        psnr_output = run([
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(reference), "-i",
            str(candidate), "-lavfi", "[0:v][1:v]psnr", "-an", "-f", "null", "-",
        ])
        ssim_output = run([
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(reference), "-i",
            str(candidate), "-lavfi", "[0:v][1:v]ssim", "-an", "-f", "null", "-",
        ])
        psnr_match = re.search(r"average:([0-9.]+|inf)", psnr_output)
        ssim_match = re.search(r"All:([0-9.]+)", ssim_output)
        psnr = float(psnr_match.group(1)) if psnr_match else None
        ssim = float(ssim_match.group(1)) if ssim_match else None
        audio_snr, audio_correlation = audio_fidelity(reference, candidate)
        return psnr, ssim, audio_snr, audio_correlation
    except (RuntimeError, ValueError):
        return None, None, None, None


def peak_memory(path: Path, gpu_ids: set[int]) -> float | None:
    if not path.exists():
        return None
    peaks: dict[int, float] = {}
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            try:
                gpu_id = int(row[1].strip())
                memory_mib = float(row[2].strip())
            except (IndexError, ValueError):
                continue
            if gpu_id in gpu_ids:
                peaks[gpu_id] = max(peaks.get(gpu_id, 0.0), memory_mib)
    return max(peaks.values()) if peaks else None


def finite_mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None and value != float("inf")]
    return sum(filtered) / len(filtered) if filtered else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if value == float("inf"):
        return "inf"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_unit(value: Any, suffix: str, digits: int = 3) -> str:
    return "-" if value is None else f"{fmt(value, digits)}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--skip-fidelity", action="store_true")
    args = parser.parse_args()
    cases_root = args.output_root / "cases"
    report_dir = args.report_dir or args.output_root / "aggregate"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for case, (dit, encoder, module_count) in CASES.items():
        case_root = cases_root / case
        gpu_file = case_root / "gpus.txt"
        gpu_ids = {
            int(value) for value in gpu_file.read_text().strip().split(",")
        } if gpu_file.exists() else set(range(8))
        for task in TASKS:
            video = case_root / f"{task}.mp4"
            timing_file = case_root / f"{task}.time.json"
            if not video.exists() or not timing_file.exists():
                continue
            timing = json.loads(timing_file.read_text())
            media = probe(video)
            (case_root / f"{task}.media.json").write_text(
                json.dumps(media, indent=2) + "\n"
            )
            reference = cases_root / "bf16" / f"{task}.mp4"
            psnr, ssim, audio_snr, audio_correlation = (None, None, None, None)
            if not args.skip_fidelity and reference.exists():
                psnr, ssim, audio_snr, audio_correlation = fidelity(reference, video)
            row = {
                "case": case,
                "task": task,
                "dit": dit,
                "text_encoder": encoder,
                "encoder_fp8_linear_modules_per_rank": module_count,
                "wall_time_s": float(timing["wall_time_s"]),
                "peak_gpu_memory_mib": peak_memory(
                    case_root / f"{task}.gpu.csv", gpu_ids
                ),
                "psnr_avg_db_vs_bf16": psnr,
                "ssim_vs_bf16": ssim,
                "audio_snr_db_vs_bf16": audio_snr,
                "audio_corr_vs_bf16": audio_correlation,
                **media,
            }
            rows.append(row)

    baseline = {
        row["task"]: row for row in rows if row["case"] == "bf16"
    }
    for row in rows:
        reference = baseline.get(row["task"])
        row["speedup_vs_bf16"] = (
            reference["wall_time_s"] / row["wall_time_s"] if reference else None
        )
        base_memory = reference["peak_gpu_memory_mib"] if reference else None
        row["memory_reduction_vs_bf16_pct"] = (
            100.0 * (base_memory - row["peak_gpu_memory_mib"]) / base_memory
            if base_memory and row["peak_gpu_memory_mib"] is not None else None
        )

    fieldnames = [
        "case", "task", "dit", "text_encoder",
        "encoder_fp8_linear_modules_per_rank", "wall_time_s",
        "speedup_vs_bf16", "peak_gpu_memory_mib",
        "memory_reduction_vs_bf16_pct", "psnr_avg_db_vs_bf16",
        "ssim_vs_bf16", "audio_snr_db_vs_bf16", "audio_corr_vs_bf16",
        "duration_s", "width", "height", "fps",
        "audio_sample_rate", "audio_channels", "size_bytes",
    ]
    with (report_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=True) + "\n")

    summary: list[dict[str, Any]] = []
    for case in CASES:
        case_rows = [row for row in rows if row["case"] == case]
        if not case_rows:
            continue
        summary.append({
            "case": case,
            "tasks": len(case_rows),
            "mean_speedup_vs_bf16": finite_mean(
                [row["speedup_vs_bf16"] for row in case_rows]
            ),
            "max_peak_gpu_memory_mib": max(
                row["peak_gpu_memory_mib"] for row in case_rows
                if row["peak_gpu_memory_mib"] is not None
            ),
            "mean_memory_reduction_vs_bf16_pct": finite_mean(
                [row["memory_reduction_vs_bf16_pct"] for row in case_rows]
            ),
            "mean_psnr_avg_db_vs_bf16": finite_mean(
                [row["psnr_avg_db_vs_bf16"] for row in case_rows]
            ),
            "mean_ssim_vs_bf16": finite_mean(
                [row["ssim_vs_bf16"] for row in case_rows]
            ),
            "mean_audio_snr_db_vs_bf16": finite_mean(
                [row["audio_snr_db_vs_bf16"] for row in case_rows]
            ),
            "mean_audio_corr_vs_bf16": finite_mean(
                [row["audio_corr_vs_bf16"] for row in case_rows]
            ),
        })
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n"
    )

    lines = [
        "# Measured results", "",
        "All numbers are descriptive; no quality threshold is applied. PSNR and SSIM",
        "compare decoded video frames with the matching BF16 output at seed 0.", "",
        "## Interpretation", "",
        "- `all_linear_fp8` is the throughput-oriented choice and the simplest global",
        "  configuration. It has the highest mean speedup in this matrix.",
        "- `dit_fp8` is the peak-memory-oriented choice. It has the largest mean memory",
        "  reduction and the lowest maximum peak memory among the FP8 DiT cases.",
        "- `te_mlp_l0_9` is the conservative encoder-only starting point. Encoder-only",
        "  quantization preserves higher similarity here, but provides no stable",
        "  end-to-end speed or peak-memory benefit on this workload.",
        "- Quantization quality is not monotonic for a single diffusion seed. Inspect the",
        "  linked videos rather than treating any one fidelity metric as a hard gate.", "",
        "## Aggregate results", "",
        "| Case | Tasks | Mean speedup | Max peak GPU MiB | Mean memory saved | Mean PSNR dB | Mean SSIM | Mean audio SNR dB | Mean audio corr. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| `{item['case']}` | {item['tasks']} | "
            f"{fmt_unit(item['mean_speedup_vs_bf16'], 'x')} | "
            f"{fmt(item['max_peak_gpu_memory_mib'], 0)} | "
            f"{fmt_unit(item['mean_memory_reduction_vs_bf16_pct'], '%', 1)} | "
            f"{fmt(item['mean_psnr_avg_db_vs_bf16'])} | "
            f"{fmt(item['mean_ssim_vs_bf16'], 4)} | "
            f"{fmt(item['mean_audio_snr_db_vs_bf16'])} | "
            f"{fmt(item['mean_audio_corr_vs_bf16'], 4)} |"
        )
    lines.extend([
        "", "## Per-task results", "",
        "| Case | Task | Video | Wall s | Speedup | Peak GPU MiB | Memory saved | PSNR dB | SSIM | Audio SNR dB | Audio corr. |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        video_link = f"[MP4](videos/{row['case']}/{row['task']}.mp4)"
        lines.append(
            f"| `{row['case']}` | `{row['task']}` | {video_link} | "
            f"{fmt(row['wall_time_s'], 2)} | "
            f"{fmt_unit(row['speedup_vs_bf16'], 'x')} | "
            f"{fmt(row['peak_gpu_memory_mib'], 0)} | "
            f"{fmt_unit(row['memory_reduction_vs_bf16_pct'], '%', 1)} | "
            f"{fmt(row['psnr_avg_db_vs_bf16'])} | "
            f"{fmt(row['ssim_vs_bf16'], 4)} | "
            f"{fmt(row['audio_snr_db_vs_bf16'])} | "
            f"{fmt(row['audio_corr_vs_bf16'], 4)} |"
        )
    (report_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

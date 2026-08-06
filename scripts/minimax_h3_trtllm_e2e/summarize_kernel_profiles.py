#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROFILE_MODES = (
    "trtllm_dense",
    "sage_fp8",
    "skip_softmax_05_gate099",
    "sage_fp8_skip_05_gate099",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for mode in PROFILE_MODES:
        mode_root = args.profile_root / mode
        exit_code = (mode_root / "exit_code").read_text(encoding="utf-8").strip()
        if exit_code != "0":
            raise RuntimeError(f"Kernel profile failed for {mode}: exit={exit_code}")

        nvtx_rows = read_csv(mode_root / "stats_nvtx_gpu_proj_sum.csv")
        steady = next(row for row in nvtx_rows if row["Range"] == ":steady_diffusion")
        kernel_rows = read_csv(mode_root / "stats_cuda_gpu_kern_sum.csv")
        attention_rows = [
            row
            for row in kernel_rows
            if "fmha" in row["Name"].lower() or "sageQuantQkvKernel" in row["Name"]
        ]
        if not attention_rows:
            raise RuntimeError(f"No attention kernels found for {mode}")

        kernels = []
        for row in attention_rows:
            kernels.append(
                {
                    "kind": "SAGE quantization"
                    if "sageQuantQkvKernel" in row["Name"]
                    else "FMHA",
                    "name": row["Name"],
                    "instances": int(row["Instances"]),
                    "total_gpu_s": int(row["Total Time (ns)"]) / 1e9,
                    "average_gpu_ms": float(row["Avg (ns)"]) / 1e6,
                    "gpu_time_percent": float(row["Time (%)"]),
                }
            )
        primary_fmha = max(
            (kernel for kernel in kernels if kernel["kind"] == "FMHA"),
            key=lambda kernel: float(kernel["total_gpu_s"]),
        )
        results.append(
            {
                "mode": mode,
                "artifacts": {
                    "cuda_kernel_summary": f"kernel_profiles/{mode}/cuda_gpu_kern_sum.csv",
                    "nvtx_gpu_projection": f"kernel_profiles/{mode}/nvtx_gpu_proj_sum.csv",
                },
                "rank0_projected_diffuse_gpu_s": int(steady["Total Proj Time (ns)"])
                / 1e9,
                "attention_gpu_s_four_gpu_total": sum(
                    float(kernel["total_gpu_s"]) for kernel in kernels
                ),
                "attention_gpu_time_percent": sum(
                    float(kernel["gpu_time_percent"]) for kernel in kernels
                ),
                "primary_fmha_average_gpu_ms": primary_fmha["average_gpu_ms"],
                "primary_fmha_instances": primary_fmha["instances"],
                "kernels": kernels,
            }
        )

    dense_attention = float(results[0]["attention_gpu_s_four_gpu_total"])
    for result in results:
        result["attention_speedup_vs_dense"] = dense_attention / float(
            result["attention_gpu_s_four_gpu_total"]
        )

    payload = {
        "profile_scope": "second diffuse request after one warmup",
        "notes": [
            "Rank-0 projected diffuse time comes from the steady_diffusion NVTX range.",
            "Attention totals aggregate CUDA kernel time across all four GPU processes.",
            "Nsight profiling is separate from the uninstrumented E2E timing matrix.",
        ],
        "results": results,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "| Mode | Rank-0 projected diffuse | Attention GPU time (4-GPU total) | Attention share | Attention speedup vs dense | Primary FMHA average |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| `{result['mode']}` "
            f"| {float(result['rank0_projected_diffuse_gpu_s']):.3f} s "
            f"| {float(result['attention_gpu_s_four_gpu_total']):.3f} s "
            f"| {float(result['attention_gpu_time_percent']):.1f}% "
            f"| {float(result['attention_speedup_vs_dense']):.3f}× "
            f"| {float(result['primary_fmha_average_gpu_ms']):.3f} ms |"
        )
    lines.extend(
        [
            "",
            "Attention GPU time includes every FMHA kernel and SAGE quantization kernel in the capture. "
            "The total is summed across four GPUs; the projected diffuse time is the rank-0 wall-clock GPU projection.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

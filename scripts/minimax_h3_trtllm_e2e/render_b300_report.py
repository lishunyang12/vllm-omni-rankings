#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


THRESHOLD_LABELS = {"005": "0.05", "010": "0.10", "03": "0.30", "05": "0.50"}
GATE_LABELS = {"090": ("0.90", 21), "095": ("0.95", 30), "099": ("0.99", 43)}


def describe_mode(mode: str) -> tuple[str, str, str]:
    if mode == "recipe_flash":
        return "FLASH_ATTN", "—", "—"
    if mode == "trtllm_dense":
        return "TRTLLM_ATTN", "—", "—"
    if mode == "sage_fp8":
        return "TRTLLM_ATTN", "FP8", "—"

    sage = mode.startswith("sage_fp8_skip_")
    prefix = "sage_fp8_skip_" if sage else "skip_softmax_"
    suffix = mode.removeprefix(prefix)
    threshold_name, gate_name = suffix.split("_gate")
    gate, enabled_steps = GATE_LABELS[gate_name]
    skip = (
        f"threshold {THRESHOLD_LABELS[threshold_name]}, "
        f"disabled_until_timestep={gate} ({enabled_steps}/49 steps)"
    )
    return "TRTLLM_ATTN", "FP8" if sage else "—", skip


def selected_video(summary_path: Path) -> str | None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for run in payload["runs"]:
        value = run.get("mp4")
        if value:
            return str(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--link-prefix")
    args = parser.parse_args()

    if (args.artifact_dir is None) != (args.link_prefix is None):
        raise ValueError("--artifact-dir and --link-prefix must be used together")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["complete_and_stable"]:
        raise RuntimeError(f"Timing audit is incomplete: {audit['needs_rerun']}")

    dense = audit["modes"]["trtllm_dense"]["selected"]
    dense_median = float(dense["median_s"])
    rows = []
    for mode in audit["required_modes"]:
        stats = audit["modes"][mode]["selected"]
        summary = Path(stats["summary"])
        video = selected_video(summary)
        if video is None:
            raise RuntimeError(f"No video recorded for {mode}: {summary}")
        if args.artifact_dir is not None:
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
            summary_output = args.artifact_dir / f"{mode}.json"
            video_output = args.artifact_dir / f"{mode}.mp4"
            shutil.copy2(summary, summary_output)
            shutil.copy2(video, video_output)
            link_prefix = args.link_prefix.rstrip("/")
            summary_link = f"{link_prefix}/{summary_output.name}"
            video_link = f"{link_prefix}/{video_output.name}"
        else:
            summary_link = str(summary)
            video_link = video
        backend, sage, skip = describe_mode(mode)
        median = float(stats["median_s"])
        rows.append(
            {
                "mode": mode,
                "backend": backend,
                "sage": sage,
                "skip_softmax": skip,
                "median_diffuse_s": f"{median:.3f}",
                "cv_percent": f"{100 * float(stats['cv']):.2f}",
                "speedup_vs_dense": f"{dense_median / median:.3f}",
                "video": video_link,
                "summary": summary_link,
            }
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "measurement": "median of five requests after one warmup",
                    "baseline": "trtllm_dense",
                    "results": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    lines = [
        "| Mode | Backend | SAGE | Skip-Softmax | Median diffuse | CV | Speedup vs dense | Video | Raw record |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['mode']}` | `{row['backend']}` | {row['sage']} | {row['skip_softmax']} "
            f"| {row['median_diffuse_s']} s | {row['cv_percent']}% | {row['speedup_vs_dense']}× "
            f"| [MP4]({row['video']}) | [JSON]({row['summary']}) |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

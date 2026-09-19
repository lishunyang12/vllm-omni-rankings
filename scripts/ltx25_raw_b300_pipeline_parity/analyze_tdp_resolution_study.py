#!/usr/bin/env python3
"""Validate, score, and publish the LTX-2.5 TDP resolution study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


VAE_SPATIAL_COMPRESSION_RATIO = 32
MODES = ("global_sp", "stage2_tdp")
_SSIM_RE = re.compile(r"All:(?P<score>-?[0-9.]+)")
_PSNR_RE = re.compile(r"average:(?P<score>inf|[0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=Path(__file__).parent / "tdp-resolution-study",
    )
    parser.add_argument("--half-width", type=int, default=640)
    parser.add_argument("--crf", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def split_dimension(size: int, tile_count: int, overlap: int) -> list[tuple[int, int]]:
    total = size + overlap * (tile_count - 1)
    tile_size, remainder = divmod(total, tile_count)
    if tile_size <= overlap:
        raise ValueError(
            f"Cannot split {size} cells into {tile_count} tiles with overlap {overlap}"
        )
    intervals: list[tuple[int, int]] = []
    start = 0
    for index in range(tile_count):
        end = start + tile_size + int(index < remainder)
        intervals.append((start, end))
        start = end - overlap
    if intervals[-1][1] != size:
        raise RuntimeError("Internal tile split error")
    return intervals


def overlap_bands(
    *,
    target_width: int,
    target_height: int,
    internal_width: int,
    internal_height: int,
    overlap: int,
) -> list[dict[str, int | str]]:
    latent_height = internal_height // VAE_SPATIAL_COMPRESSION_RATIO
    latent_width = internal_width // VAE_SPATIAL_COMPRESSION_RATIO
    height_intervals = split_dimension(latent_height, 2, overlap)
    width_intervals = split_dimension(latent_width, 2, overlap)
    y_start = height_intervals[1][0] * VAE_SPATIAL_COMPRESSION_RATIO
    y_end = min(height_intervals[0][1] * VAE_SPATIAL_COMPRESSION_RATIO, target_height)
    x_start = width_intervals[1][0] * VAE_SPATIAL_COMPRESSION_RATIO
    x_end = min(width_intervals[0][1] * VAE_SPATIAL_COMPRESSION_RATIO, target_width)
    return [
        {
            "axis": "horizontal",
            "x": 0,
            "y": y_start,
            "width": target_width,
            "height": y_end - y_start,
        },
        {
            "axis": "vertical",
            "x": x_start,
            "y": 0,
            "width": x_end - x_start,
            "height": target_height,
        },
    ]


def probe_video(path: Path) -> dict[str, int | float]:
    completed = run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "fps": int(numerator) / int(denominator),
    }


def similarity(
    baseline: Path,
    tiled: Path,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> dict[str, float]:
    crop = f"crop={width}:{height}:{x}:{y}"
    graph = (
        f"[0:v]{crop},split=2[a1][a2];"
        f"[1:v]{crop},split=2[b1][b2];"
        "[a1][b1]ssim[ssimout];[a2][b2]psnr[psnrout]"
    )
    completed = run_checked(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(baseline),
            "-i",
            str(tiled),
            "-filter_complex",
            graph,
            "-map",
            "[ssimout]",
            "-map",
            "[psnrout]",
            "-f",
            "null",
            "-",
        ]
    )
    ssim_match = _SSIM_RE.search(completed.stderr)
    psnr_match = _PSNR_RE.search(completed.stderr)
    if ssim_match is None or psnr_match is None:
        raise ValueError(f"Could not parse SSIM/PSNR for {baseline} and {tiled}")
    psnr_text = psnr_match.group("score")
    return {
        "ssim_mean": float(ssim_match.group("score")),
        "psnr_mean_db": math.inf if psnr_text == "inf" else float(psnr_text),
    }


def create_side_by_side(
    baseline: Path,
    tiled: Path,
    output: Path,
    *,
    label: str,
    seed: int,
    target_width: int,
    target_height: int,
    half_width: int,
    crf: int,
) -> None:
    common = f"crop={target_width}:{target_height}:0:0,scale={half_width}:-2"
    graph = (
        f"[0:v]{common},drawbox=x=0:y=0:w=iw:h=50:color=black@0.68:t=fill,"
        f"drawtext=text='Global SP | {label} | seed {seed}':x=18:y=13:fontsize=22:fontcolor=white[left];"
        f"[1:v]{common},drawbox=x=0:y=0:w=iw:h=50:color=black@0.68:t=fill,"
        "drawtext=text='Stage 2 TDP | overlap 5':x=18:y=13:fontsize=22:fontcolor=white[right];"
        "[left][right]hstack=inputs=2[review]"
    )
    run_checked(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(baseline),
            "-i",
            str(tiled),
            "-filter_complex",
            graph,
            "-map",
            "[review]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def create_preview(video: Path, output: Path) -> None:
    run_checked(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostdin",
            "-ss",
            "2.0",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output),
        ]
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values),
    }


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    global_latency = [case["latency_seconds"]["global_sp"] for case in cases]
    tdp_latency = [case["latency_seconds"]["stage2_tdp"] for case in cases]
    ssim = [case["quality"]["full_frame"]["ssim_mean"] for case in cases]
    psnr = [case["quality"]["full_frame"]["psnr_mean_db"] for case in cases]
    horizontal_ssim = [
        case["quality"]["overlap_bands"][0]["ssim_mean"] for case in cases
    ]
    vertical_ssim = [case["quality"]["overlap_bands"][1]["ssim_mean"] for case in cases]
    speedups = [
        left / right for left, right in zip(global_latency, tdp_latency, strict=True)
    ]
    return {
        "latency_seconds": {
            "global_sp": summary(global_latency),
            "stage2_tdp": summary(tdp_latency),
            "speedup_global_over_tdp": summary(speedups),
        },
        "quality": {
            "full_frame_ssim": summary(ssim),
            "full_frame_psnr_db": summary(psnr),
            "horizontal_overlap_ssim": summary(horizontal_ssim),
            "vertical_overlap_ssim": summary(vertical_ssim),
        },
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    args.publish_dir = args.publish_dir.resolve()
    source_manifest_path = args.results_dir / "results.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise FileNotFoundError(binary)
    if args.half_width < 2 or args.half_width % 2:
        raise ValueError("--half-width must be a positive even integer")

    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    seeds = source["contract"]["seeds"]
    resolutions = {item["label"]: item for item in source["contract"]["resolutions"]}
    expected_count = len(seeds) * len(resolutions) * len(MODES)
    if len(source["requests"]) != expected_count:
        raise ValueError(
            f"Expected {expected_count} measured requests, found {len(source['requests'])}"
        )

    args.publish_dir.mkdir(parents=True, exist_ok=True)
    public_result_path = args.publish_dir / "results.json"
    public = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "title": "LTX-2.5 Global SP versus Stage-2 TDP resolution study",
        "contract": source["contract"],
        "provenance": {
            "model": source["model"],
            "model_class": source["model_class"],
            "vllm_omni_revision": source["vllm_omni_revision"],
            "software_versions": source["software_versions"],
            "server_command": source["server_command"],
            "gpus": source["gpus"],
            "attention_backend": source["attention_backend"],
            "latency_scope": source["contract"]["latency_scope"],
        },
        "resolutions": {},
    }
    if public_result_path.is_file() and not args.force:
        prior = json.loads(public_result_path.read_text(encoding="utf-8"))
        public["resolutions"] = prior.get("resolutions", {})

    for label, resolution in resolutions.items():
        target_width = resolution["width"]
        target_height = resolution["height"]
        internal_width = math.ceil(target_width / 64) * 64
        internal_height = math.ceil(target_height / 64) * 64
        bands = overlap_bands(
            target_width=target_width,
            target_height=target_height,
            internal_width=internal_width,
            internal_height=internal_height,
            overlap=source["contract"]["tdp_overlap_latent_cells"],
        )
        resolution_entry = public["resolutions"].setdefault(
            label,
            {
                "label": label,
                "target": {"width": target_width, "height": target_height},
                "internal": {"width": internal_width, "height": internal_height},
                "overlap_bands": bands,
                "cases": [],
            },
        )
        existing = {case["seed"]: case for case in resolution_entry.get("cases", [])}
        for seed in seeds:
            output_rel = Path("videos") / label / f"seed-{seed}.mp4"
            preview_rel = Path("previews") / label / f"seed-{seed}.jpg"
            output = args.publish_dir / output_rel
            preview = args.publish_dir / preview_rel
            if (
                seed in existing
                and output.is_file()
                and preview.is_file()
                and not args.force
            ):
                print(f"[{label} seed {seed}] already analyzed; skipping", flush=True)
                continue

            baseline_record = source["requests"][f"{label}/global_sp/seed-{seed}"]
            tiled_record = source["requests"][f"{label}/stage2_tdp/seed-{seed}"]
            baseline = args.results_dir / baseline_record["video"]
            tiled = args.results_dir / tiled_record["video"]
            if probe_video(baseline) != baseline_record["metadata"]:
                raise ValueError(
                    f"Baseline decode validation changed for {label} seed {seed}"
                )
            if probe_video(tiled) != tiled_record["metadata"]:
                raise ValueError(
                    f"TDP decode validation changed for {label} seed {seed}"
                )

            print(f"[{label} seed {seed}] computing full-frame metrics", flush=True)
            full_frame = similarity(
                baseline,
                tiled,
                x=0,
                y=0,
                width=target_width,
                height=target_height,
            )
            scored_bands = []
            for band in bands:
                scored_bands.append(
                    {
                        **band,
                        **similarity(
                            baseline,
                            tiled,
                            x=int(band["x"]),
                            y=int(band["y"]),
                            width=int(band["width"]),
                            height=int(band["height"]),
                        ),
                    }
                )

            output.parent.mkdir(parents=True, exist_ok=True)
            preview.parent.mkdir(parents=True, exist_ok=True)
            create_side_by_side(
                baseline,
                tiled,
                output,
                label=label,
                seed=seed,
                target_width=target_width,
                target_height=target_height,
                half_width=args.half_width,
                crf=args.crf,
            )
            published_metadata = probe_video(output)
            if (
                published_metadata["width"] != 2 * args.half_width
                or published_metadata["frames"] != baseline_record["metadata"]["frames"]
                or published_metadata["fps"] != baseline_record["metadata"]["fps"]
            ):
                raise ValueError(
                    f"Unexpected published metadata for {label} seed {seed}: {published_metadata}"
                )
            create_preview(output, preview)
            case = {
                "seed": seed,
                "latency_seconds": {
                    "global_sp": baseline_record["elapsed_seconds"],
                    "stage2_tdp": tiled_record["elapsed_seconds"],
                    "speedup_global_over_tdp": baseline_record["elapsed_seconds"]
                    / tiled_record["elapsed_seconds"],
                },
                "peak_used_mib": {
                    "global_sp": baseline_record["memory"]["max_peak_used_mib"],
                    "stage2_tdp": tiled_record["memory"]["max_peak_used_mib"],
                },
                "quality": {"full_frame": full_frame, "overlap_bands": scored_bands},
                "video": output_rel.as_posix(),
                "preview": preview_rel.as_posix(),
                "video_bytes": output.stat().st_size,
                "video_sha256": hash_file(output),
                "video_metadata": published_metadata,
            }
            existing[seed] = case
            resolution_entry["cases"] = [existing[item] for item in sorted(existing)]
            resolution_entry["summary"] = aggregate(resolution_entry["cases"])
            write_json_atomic(public_result_path, public)

    checksum_paths = [public_result_path]
    checksum_paths.extend(sorted((args.publish_dir / "videos").rglob("*.mp4")))
    checksum_paths.extend(sorted((args.publish_dir / "previews").rglob("*.jpg")))
    checksum_lines = [
        f"{hash_file(path)}  {path.relative_to(args.publish_dir).as_posix()}"
        for path in checksum_paths
    ]
    (args.publish_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(f"Published analysis: {public_result_path}", flush=True)


if __name__ == "__main__":
    main()

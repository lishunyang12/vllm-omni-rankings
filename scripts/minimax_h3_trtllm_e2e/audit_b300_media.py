#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_video(mode: str, summary_path: Path) -> dict[str, object]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    video_path = next(
        (Path(run["mp4"]) for run in summary["runs"] if run.get("mp4")),
        None,
    )
    if video_path is None or not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty video for {mode}: {video_path}")

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(probe.stdout)
    video_stream = next(
        stream for stream in metadata["streams"] if stream["codec_type"] == "video"
    )
    audio_stream = next(
        stream for stream in metadata["streams"] if stream["codec_type"] == "audio"
    )
    expected = {
        "width": int(summary["width"]),
        "height": int(summary["height"]),
        "fps": int(summary["runs"][0]["fps"]),
        "audio_sample_rate": int(summary["runs"][0]["audio_sample_rate"]),
    }
    observed = {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": video_stream["avg_frame_rate"],
        "video_frames": int(video_stream["nb_frames"]),
        "audio_sample_rate": int(audio_stream["sample_rate"]),
        "audio_channels": int(audio_stream["channels"]),
        "duration_s": float(metadata["format"]["duration"]),
    }
    if (
        observed["width"] != expected["width"]
        or observed["height"] != expected["height"]
    ):
        raise RuntimeError(f"Unexpected video dimensions for {mode}: {observed}")
    if observed["fps"] != f"{expected['fps']}/1" or observed["video_frames"] != 209:
        raise RuntimeError(f"Unexpected frame metadata for {mode}: {observed}")
    if (
        observed["audio_sample_rate"] != expected["audio_sample_rate"]
        or observed["audio_channels"] != 2
    ):
        raise RuntimeError(f"Unexpected audio metadata for {mode}: {observed}")

    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return {
        "mode": mode,
        "path": str(video_path),
        "size_bytes": video_path.stat().st_size,
        "sha256": file_sha256(video_path),
        **observed,
        "full_decode": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("timing_audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    timing = json.loads(args.timing_audit.read_text(encoding="utf-8"))
    inputs = [
        (
            mode,
            Path(timing["modes"][mode]["selected"]["summary"]),
        )
        for mode in timing["required_modes"]
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        videos = list(executor.map(lambda values: audit_video(*values), inputs))

    payload = {
        "criteria": {
            "required_videos": len(timing["required_modes"]),
            "video_dimensions": "1248x768",
            "video_frames": 209,
            "fps": 24,
            "audio_sample_rate": 32000,
            "audio_channels": 2,
            "full_ffmpeg_decode": True,
        },
        "videos": videos,
        "accepted": len(videos) == len(timing["required_modes"]),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("pass" if payload["accepted"] else "fail")


if __name__ == "__main__":
    main()

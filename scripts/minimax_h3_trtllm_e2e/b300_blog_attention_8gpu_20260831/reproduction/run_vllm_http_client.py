from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import av
import numpy as np


EXPECTED_PROMPT_SHA256 = "98f36b879692095e099ae824c18d9e93e7006a490e082fd474a5f531769dcf06"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:18081")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=2)
    parser.add_argument("--max-start-temperature-c", type=int)
    parser.add_argument(
        "--gpu-indices",
        help="Comma-separated physical GPU indices used by the server.",
    )
    parser.add_argument("--cooldown-timeout-s", type=float, default=900.0)
    parser.add_argument("--inter-request-delay-s", type=float, default=0.0)
    return parser.parse_args()


def validate_media(path: Path) -> dict[str, object]:
    with av.open(str(path)) as container:
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        metadata = {
            "video_codec": video_stream.codec_context.name,
            "audio_codec": audio_stream.codec_context.name,
            "fps": float(video_stream.average_rate),
            "audio_sample_rate": audio_stream.codec_context.sample_rate,
            "audio_channels": len(audio_stream.codec_context.layout.channels),
        }
        frame_means = []
        frame_variances = []
        for frame in container.decode(video=0):
            pixels = frame.to_ndarray(format="rgb24")
            frame_means.append(float(pixels.mean()))
            frame_variances.append(float(pixels.var()))

    audio_square_sum = 0.0
    audio_sample_count = 0
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            samples = frame.to_ndarray().astype(np.float64, copy=False)
            audio_square_sum += float(np.square(samples).sum())
            audio_sample_count += samples.size

    metadata.update(
        decoded_frames=len(frame_means),
        frame_mean_variance=float(np.var(frame_means)),
        minimum_frame_pixel_variance=min(frame_variances),
        audio_rms=(audio_square_sum / audio_sample_count) ** 0.5,
    )
    expected = {
        "video_codec": "h264",
        "audio_codec": "aac",
        "decoded_frames": 243,
        "fps": 24.0,
        "audio_sample_rate": 32000,
        "audio_channels": 2,
    }
    for key, value in expected.items():
        if metadata[key] != value:
            raise RuntimeError(f"Unexpected {key}: {metadata[key]!r}, expected {value!r}")
    if metadata["frame_mean_variance"] <= 0 or metadata["minimum_frame_pixel_variance"] <= 0:
        raise RuntimeError("Decoded video has no measurable frame variance")
    if metadata["audio_rms"] <= 0:
        raise RuntimeError("Decoded audio has zero RMS")
    return metadata


def wait_for_start_temperature(
    maximum_c: int | None,
    timeout_s: float,
    gpu_indices: set[int] | None,
) -> tuple[float, dict[int, int]]:
    if maximum_c is None:
        return 0.0, {}
    started = time.perf_counter()
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        temperatures = {
            int(index): int(temperature)
            for line in result.stdout.splitlines()
            for index, temperature in [line.split(",", maxsplit=1)]
            if gpu_indices is None or int(index) in gpu_indices
        }
        waited = time.perf_counter() - started
        if temperatures and max(temperatures.values()) <= maximum_c:
            return waited, temperatures
        if waited >= timeout_s:
            raise TimeoutError(
                f"GPUs did not cool to {maximum_c} C within {timeout_s} seconds; "
                f"current temperatures: {temperatures}"
            )
        time.sleep(5.0)


def read_response_headers(path: Path) -> dict[str, object]:
    # curl may write multiple HTTP header blocks; the final block describes the
    # response body saved beside this file.
    blocks = path.read_bytes().replace(b"\r\n", b"\n").strip().split(b"\n\n")
    header_lines = blocks[-1].splitlines()[1:]
    message = BytesParser(policy=default).parsebytes(b"\n".join(header_lines) + b"\n\n")
    stage_durations = json.loads(message["X-Stage-Durations"])
    return {
        "request_id": message["X-Request-Id"],
        "server_inference_time_seconds": float(message["X-Inference-Time-S"]),
        "stage_durations": stage_durations,
        "peak_memory_mb": float(message["X-Peak-Memory-MB"]),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    if prompt_sha256 != EXPECTED_PROMPT_SHA256:
        raise RuntimeError(f"Unexpected prompt digest: {prompt_sha256}")
    gpu_indices = (
        {int(value) for value in args.gpu_indices.split(",")}
        if args.gpu_indices
        else None
    )

    results = []
    for index in range(1, args.warmup + args.measure + 1):
        fixed_wait = args.inter_request_delay_s if index > 1 else 0.0
        if fixed_wait:
            time.sleep(fixed_wait)
        thermal_wait, start_temperatures = wait_for_start_temperature(
            args.max_start_temperature_c,
            args.cooldown_timeout_s,
            gpu_indices,
        )
        cooldown_wait = fixed_wait + thermal_wait
        output_path = args.output_dir / f"response_{index}.mp4"
        headers_path = args.output_dir / f"response_{index}.headers"
        result = subprocess.run(
            [
                "curl",
                "--fail-with-body",
                "--silent",
                "--show-error",
                "--max-time",
                "1800",
                "--dump-header",
                str(headers_path),
                "--output",
                str(output_path),
                "--write-out",
                "%{time_total}",
                "--request",
                "POST",
                f"{args.server}/v1/videos/sync",
                "--form",
                f"prompt=<{args.prompt_file}",
                "--form",
                "width=1344",
                "--form",
                "height=768",
                "--form",
                "fps=24",
                "--form",
                "num_inference_steps=50",
                "--form",
                "seed=0",
                "--form",
                (
                    'extra_params={"task":"t2va","duration":10.0,'
                    '"aspect_ratio":"16:9","flow_shift":12.0,"audio_flow_shift":3.0}'
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        latency = float(result.stdout.strip())
        payload = output_path.read_bytes()
        record = {
            "request_index": index,
            "excluded_warmup": index <= args.warmup,
            "client_complete_mp4_e2e_seconds": latency,
            "video": str(output_path),
            "video_bytes": len(payload),
            "video_sha256": hashlib.sha256(payload).hexdigest(),
            "cooldown_wait_seconds": cooldown_wait,
            "start_temperatures_c": start_temperatures,
            **read_response_headers(headers_path),
        }
        results.append(record)
        (args.output_dir / "client_results.json").write_text(
            json.dumps({"runs": results}, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(record), flush=True)

    for record in results:
        record["media_validation"] = validate_media(Path(record["video"]))
    measured = [
        record["client_complete_mp4_e2e_seconds"]
        for record in results
        if not record["excluded_warmup"]
    ]
    summary = {
        "prompt_sha256_stripped": prompt_sha256,
        "warmup_requests": args.warmup,
        "measured_requests": args.measure,
        "max_start_temperature_c": args.max_start_temperature_c,
        "inter_request_delay_seconds": args.inter_request_delay_s,
        "gpu_indices": sorted(gpu_indices) if gpu_indices is not None else None,
        "runs": results,
        "measured_client_complete_mp4_e2e_seconds": measured,
        "median_client_complete_mp4_e2e_seconds": statistics.median(measured),
    }
    (args.output_dir / "client_results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from vllm_omni.entrypoints.omni import Omni


RUN_ROOT = Path(__file__).resolve().parent
DEPLOY_CONFIG = RUN_ROOT / "deploy.yaml"
INPUT_IMAGE = RUN_ROOT / "input.png"
OUTPUT_DIR = RUN_ROOT / "outputs"
METRICS_DIR = RUN_ROOT / "metrics"
H3_MODEL = Path(
    "/home/zjy/.cache/huggingface/hub/"
    "models--MiniMaxAI--MiniMax-H3/snapshots/"
    "42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA"
)
REPO = Path("/home/zjy/code/lsy/worktree/minimax-h3-super-acceleration")
PROMPT = (
    "A medium close-up shot features a Caucasian man with a beard, wearing a green and white "
    "baseball cap without any letters on the front, and a light blue shirt over a white t-shirt. "
    "He is positioned in the center of the frame, looking intently directly at the camera, his eyes "
    "focused on camera. His facial expression is one of deep concentration, with his brow slightly "
    "raised. As he looks straight at the camera, a quick sniff sound is heard, and then he speaks "
    "with a deep male voice and a satisfied tone, saying, 'I think it's so good.' The camera remains "
    "static throughout, maintaining a shallow depth of field, which keeps the man in sharp focus "
    "while the background is softly blurred, showing a beige wall behind him. After a brief pause, "
    "another short, audible sniff is heard. The man then continues to speak, his voice maintaining "
    "the same quality, as he states, 'So good. So good.' He elaborates further, emphasizing his point "
    "with a final statement, 'This got to be, it's got to be the best tool I've ever seen.'"
)
CASES = (
    (5, "warmup", 0),
    (5, "measured", 1),
    (5, "measured", 2),
    (5, "measured", 3),
    (10, "warmup", 0),
    (10, "measured", 1),
    (10, "measured", 2),
    (10, "measured", 3),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_text(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    payload = json.loads(
        run_text(
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-count_frames",
            "-of",
            "json",
            str(path),
        )
    )
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if decode.returncode != 0:
        raise RuntimeError(f"full ffmpeg decode failed for {path.name}: {decode.stderr}")
    video_streams = [stream for stream in payload["streams"] if stream["codec_type"] == "video"]
    audio_streams = [stream for stream in payload["streams"] if stream["codec_type"] == "audio"]
    if len(video_streams) != 1 or len(audio_streams) != 1:
        raise RuntimeError(
            f"expected one video and one audio stream in {path.name}, got "
            f"video={len(video_streams)} audio={len(audio_streams)}"
        )
    video = video_streams[0]
    audio = audio_streams[0]
    return {
        "container_duration_s": float(payload["format"]["duration"]),
        "video_codec": video["codec_name"],
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": video["avg_frame_rate"],
        "frames": int(video.get("nb_read_frames") or video.get("nb_frames")),
        "audio_codec": audio["codec_name"],
        "audio_sample_rate": int(audio["sample_rate"]),
        "audio_channels": int(audio["channels"]),
        "full_decode": "passed",
    }


def extract_video_bytes(output: Any) -> bytes:
    # The diffusion formatter stores the primary video payload in ``images``;
    # ``multimodal_output`` carries side-channel metadata and audio/actions.
    video = getattr(output, "images", None)
    if video is None:
        multimodal = getattr(output, "multimodal_output", None)
        if isinstance(multimodal, dict):
            video = multimodal.get("video")
    while isinstance(video, (list, tuple)) and len(video) == 1:
        video = video[0]
    if isinstance(video, (bytearray, memoryview)):
        video = bytes(video)
    if not isinstance(video, bytes):
        raise RuntimeError(f"final video payload is {type(video).__name__}, expected bytes")
    if not video.startswith((b"\x00\x00\x00", b"ftyp")):
        raise RuntimeError("final video payload does not look like MP4")
    return video


def build_metadata() -> dict[str, Any]:
    import diffusers
    import torch
    import transformers
    import vllm
    import vllm_omni

    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "pull_request": "https://github.com/vllm-project/vllm-omni/pull/6540",
        "commit": run_text("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "git_status": run_text("git", "-C", str(REPO), "status", "--short"),
        "hostname": platform.node(),
        "physical_gpus": [2, 3],
        "visible_logical_gpus": [0, 1],
        "gpu_inventory": run_text(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ).splitlines(),
        "input_image": INPUT_IMAGE.name,
        "input_sha256": sha256(INPUT_IMAGE),
        "prompt": PROMPT,
        "seed": 42,
        "protocol": {
            "durations_seconds": [5, 10],
            "warmups_per_duration": 1,
            "measured_repeats_per_duration": 3,
            "request_concurrency": 1,
            "latency_scope": "Omni.generate call through receipt of complete encoded MP4 bytes",
            "excluded": "engine startup/model load, output-file write, ffprobe, and full ffmpeg validation",
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "vllm_omni": getattr(vllm_omni, "__version__", "unknown"),
            "transformers": transformers.__version__,
            "diffusers": diffusers.__version__,
        },
        "deploy_config": DEPLOY_CONFIG.name,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "VLLM_OMNI_VIDEO_SYNC_TIMEOUT",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
            )
        },
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_duration: dict[str, Any] = {}
    for duration in (5, 10):
        samples = [row for row in records if row["duration_requested_s"] == duration and row["kind"] == "measured"]
        by_duration[str(duration)] = {
            "measured_repeats": len(samples),
            "stage_0_draft_mean_s": mean([row["stage_0_draft_s"] for row in samples]),
            "stage_1_refiner_mean_s": mean([row["stage_1_refiner_s"] for row in samples]),
            "wall_mean_s": mean([row["wall_s"] for row in samples]),
            "wall_min_s": min(row["wall_s"] for row in samples),
            "wall_max_s": max(row["wall_s"] for row in samples),
        }
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "all_outputs_valid": all(row["validation"]["full_decode"] == "passed" for row in records),
        "records": records,
        "measured_summary_by_duration": by_duration,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata()
    (METRICS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"event": "metadata", "data": metadata}, default=str), flush=True)

    image = Image.open(INPUT_IMAGE).convert("RGB")
    prompt = {"prompt": PROMPT, "multi_modal_data": {"image": image}}
    engine: Omni | None = None
    records: list[dict[str, Any]] = []
    try:
        init_started = time.perf_counter()
        engine = Omni(
            model=str(H3_MODEL),
            deploy_config=str(DEPLOY_CONFIG),
            trust_remote_code=True,
            init_timeout=3600,
            stage_init_timeout=3600,
        )
        init_s = time.perf_counter() - init_started
        metadata["engine_init_s"] = init_s
        (METRICS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"event": "engine_ready", "engine_init_s": init_s}), flush=True)

        for duration, kind, repeat in CASES:
            params = copy.deepcopy(engine.default_sampling_params_list)
            stage_zero = params[0]
            stage_zero.extra_args = dict(stage_zero.extra_args or {})
            stage_zero.extra_args["duration"] = float(duration)
            name = f"{duration:02d}s-{kind}-{repeat}"
            output_path = OUTPUT_DIR / f"{name}.mp4"
            print(json.dumps({"event": "request_start", "name": name, "at": utc_now()}), flush=True)
            started = time.perf_counter()
            outputs = engine.generate(prompt, params, use_tqdm=False)
            wall_s = time.perf_counter() - started
            if len(outputs) != 1:
                raise RuntimeError(f"{name} returned {len(outputs)} outputs, expected exactly one")
            output = outputs[0]
            video_bytes = extract_video_bytes(output)
            stage_durations = json_value(getattr(output, "stage_durations", {}) or {})
            stage_0_s = float(stage_durations["stage_0_gen_ms"]) / 1000.0
            stage_1_s = float(stage_durations["stage_1_gen_ms"]) / 1000.0
            output_path.write_bytes(video_bytes)
            validation = probe_video(output_path)
            expected_frames = 121 if duration == 5 else 241
            if (validation["width"], validation["height"], validation["frames"]) != (1344, 768, expected_frames):
                raise RuntimeError(f"{name} output contract mismatch: {validation}")
            if validation["fps"] != "24/1" or validation["audio_sample_rate"] != 32000:
                raise RuntimeError(f"{name} media-rate contract mismatch: {validation}")
            record = {
                "name": name,
                "kind": kind,
                "repeat": repeat,
                "duration_requested_s": duration,
                "stage_0_draft_s": stage_0_s,
                "stage_1_refiner_s": stage_1_s,
                "stage_sum_s": stage_0_s + stage_1_s,
                "wall_s": wall_s,
                "orchestration_residual_s": wall_s - stage_0_s - stage_1_s,
                "stage_durations_raw": stage_durations,
                "peak_memory_mb_reported": float(getattr(output, "peak_memory_mb", 0.0) or 0.0),
                "output": str(output_path.relative_to(RUN_ROOT)),
                "bytes": output_path.stat().st_size,
                "sha256": sha256(output_path),
                "validation": validation,
                "finished_at_utc": utc_now(),
            }
            records.append(record)
            (METRICS_DIR / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
            (METRICS_DIR / "partial-results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps({"event": "request_complete", "record": record}), flush=True)

        summary = summarize(records)
        (METRICS_DIR / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"event": "complete", "summary": summary}), flush=True)
    except BaseException as exc:
        failure = {
            "created_at_utc": utc_now(),
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "completed_records": records,
        }
        (METRICS_DIR / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps({"event": "failure", "data": failure}), flush=True)
        raise
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()

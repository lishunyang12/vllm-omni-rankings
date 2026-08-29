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


RUN_ROOT = Path(os.environ.get("H3_RUN_ROOT", Path(__file__).resolve().parent))
DEPLOY_CONFIG = RUN_ROOT / "deploy.yaml"
INPUT_IMAGE = Path(
    "/home/zjy/code/lsy/models/sana-h3-super-official/dataset/inputs/"
    "i2v_talking/bamboo-forest-wuxia-pair.jpg"
)
OUTPUT_DIR = RUN_ROOT / "outputs"
METRICS_DIR = RUN_ROOT / "metrics"
REPO = Path("/home/zjy/code/lsy/worktree/minimax-h3-super-acceleration")
H3_MODEL = Path(
    "/home/zjy/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/"
    "42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA"
)
PROMPT = (
    "Use the supplied image as the exact opening frame. Preserve every character's identity, facial "
    "features, clothing, hair, environment, camera composition, and lighting. Create one continuous "
    "5-second human speaking shot with natural movement and realistic timing. The two figures remain "
    "controlled and still as mist and fine snow drift through the bamboo. The person in white speaks "
    "first, and the person in black answers quietly with restrained tension. Dialogue: The person in "
    "white says, “You’re late.” The person in black replies, “But I came. Where is it?” The first "
    "speaker answers, “Where you would least dare to look.” All dialogue must be spoken in clear, "
    "natural English, with accurate phoneme-level lip synchronization, realistic expressions, blinking, "
    "breathing, and subtle head motion. Keep natural ambient sound. No subtitles, captions, watermarks, "
    "voice-over, extra people, duplicated characters, identity drift, or facial distortion."
)
CASES = (("warmup", 0), ("measured", 1), ("measured", 2), ("measured", 3))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_text(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if decoded.returncode:
        raise RuntimeError(f"full ffmpeg decode failed for {path.name}: {decoded.stderr}")
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in payload["streams"] if stream["codec_type"] == "audio")
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
    return video


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "pull_request": "https://github.com/vllm-project/vllm-omni/pull/6540",
        "commit": run_text("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "dirty_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(REPO), "diff", "--binary"])
        ).hexdigest(),
        "hostname": platform.node(),
        "physical_gpus": [4, 5],
        "visible_logical_gpus": [0, 1],
        "input_image": str(INPUT_IMAGE),
        "input_sha256": sha256(INPUT_IMAGE),
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "seed": 50803,
        "official_alignment": {
            "h3_lora": "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
            "h3_lora_sha256": "5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29",
            "h3_lora_scale": 0.0625,
            "h3_video_flow_shift": 12.0,
            "h3_audio_flow_shift": 3.0,
            "h3_sigma_points": 5,
            "h3_transformer_forwards": 4,
            "ltx_lora_scale": 0.8,
            "ltx_sigmas": [0.909375, 0.725, 0.421875, 0.0],
            "lossy_stage1_attention_override": None,
        },
        "protocol": {
            "duration_seconds": 5,
            "warmups": 1,
            "measured_repeats": 3,
            "request_concurrency": 1,
            "latency_scope": "Omni.generate through receipt of complete encoded MP4 bytes",
            "excluded": "engine/model startup, output write, ffprobe, and ffmpeg decode validation",
        },
        "software": {"python": sys.version.split()[0]},
        "deploy_config": str(DEPLOY_CONFIG),
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
    (METRICS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"event": "metadata", "data": metadata}), flush=True)

    prompt = {
        "prompt": PROMPT,
        "multi_modal_data": {"image": Image.open(INPUT_IMAGE).convert("RGB")},
    }
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
        metadata["engine_init_s"] = time.perf_counter() - init_started
        (METRICS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"event": "engine_ready", "engine_init_s": metadata["engine_init_s"]}), flush=True)

        for kind, repeat in CASES:
            params = copy.deepcopy(engine.default_sampling_params_list)
            name = f"05s-{kind}-{repeat}"
            output_path = OUTPUT_DIR / f"{name}.mp4"
            print(json.dumps({"event": "request_start", "name": name, "at": utc_now()}), flush=True)
            started = time.perf_counter()
            outputs = engine.generate(prompt, params, use_tqdm=False)
            wall_s = time.perf_counter() - started
            if len(outputs) != 1:
                raise RuntimeError(f"{name} returned {len(outputs)} outputs")
            output = outputs[0]
            output_path.write_bytes(extract_video_bytes(output))
            stage_durations = json_value(getattr(output, "stage_durations", {}) or {})
            stage_0_s = float(stage_durations["stage_0_gen_ms"]) / 1000.0
            stage_1_s = float(stage_durations["stage_1_gen_ms"]) / 1000.0
            validation = probe_video(output_path)
            if (validation["width"], validation["height"], validation["frames"]) != (1920, 1088, 121):
                raise RuntimeError(f"{name} output contract mismatch: {validation}")
            if validation["fps"] != "24/1" or validation["audio_sample_rate"] != 32000:
                raise RuntimeError(f"{name} media rate mismatch: {validation}")
            record = {
                "name": name,
                "kind": kind,
                "repeat": repeat,
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
            print(json.dumps({"event": "request_complete", "record": record}), flush=True)

        measured = [record for record in records if record["kind"] == "measured"]
        summary = {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "all_outputs_valid": True,
            "records": records,
            "measured_summary": {
                "repeats": len(measured),
                "stage_0_draft_mean_s": statistics.fmean(row["stage_0_draft_s"] for row in measured),
                "stage_1_refiner_mean_s": statistics.fmean(row["stage_1_refiner_s"] for row in measured),
                "wall_mean_s": statistics.fmean(row["wall_s"] for row in measured),
                "wall_min_s": min(row["wall_s"] for row in measured),
                "wall_max_s": max(row["wall_s"] for row in measured),
            },
        }
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

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import vllm_omni.diffusion.diffusion_engine as diffusion_engine
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


PROMPT = (
    "In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; "
    "footsteps crunch in the snow while the creature breathes and softly snorts."
)


MODE_CONFIGS: dict[str, dict[str, object]] = {
    # Explicit FA4 baseline using the recipe's four-GPU parallel profile.
    "recipe_flash": {"default": {"backend": "FLASH_ATTN"}},
    # Control for separating the TRTLLM dense-kernel effect from the two features.
    "trtllm_dense": {"default": {"backend": "TRTLLM_ATTN"}},
    # Calibration-free Skip-Softmax. The gate matches the PR composition probe.
    "skip_softmax": {
        "default": {
            "backend": "TRTLLM_ATTN",
            "skip_softmax": {
                "threshold": 0.5,
                "disabled_until_timestep": 0.94,
            },
        },
        "per_role": {"minimax_h3.token_refiner": {"backend": "TRTLLM_ATTN"}},
    },
    # B300 is SM103; use the FP8-QK SAGE kernel (the INT8 kernel is not a
    # valid B300 target). Block sizes match the #5509 example.
    "sage_fp8": {
        "default": {
            "backend": "TRTLLM_ATTN",
            "quant": {
                "dtype_qk": "fp8_e4m3",
                "q_block_size": 1,
                "k_block_size": 16,
            },
        },
        "per_role": {"minimax_h3.token_refiner": {"backend": "TRTLLM_ATTN"}},
    },
    "sage_fp8_skip": {
        "default": {
            "backend": "TRTLLM_ATTN",
            "skip_softmax": {
                "threshold": 0.3,
                "disabled_until_timestep": 0.9,
            },
            "quant": {
                "dtype_qk": "fp8_e4m3",
                "q_block_size": 1,
                "k_block_size": 16,
            },
        },
        "per_role": {"minimax_h3.token_refiner": {"backend": "TRTLLM_ATTN"}},
    },
    "sage_int8": {
        "default": {
            "backend": "TRTLLM_ATTN",
            "quant": {
                "dtype_qk": "int8",
                "q_block_size": 1,
                "k_block_size": 16,
            },
        },
        "per_role": {"minimax_h3.token_refiner": {"backend": "TRTLLM_ATTN"}},
    },
    "sage_int8_skip": {
        "default": {
            "backend": "TRTLLM_ATTN",
            "skip_softmax": {
                "threshold": 0.3,
                "disabled_until_timestep": 0.9,
            },
            "quant": {
                "dtype_qk": "int8",
                "q_block_size": 1,
                "k_block_size": 16,
            },
        },
        "per_role": {"minimax_h3.token_refiner": {"backend": "TRTLLM_ATTN"}},
    },
    "sage_int8_skip_03_gate099": {
        "default": {
            "backend": "TRTLLM_ATTN",
            "skip_softmax": {
                "threshold": 0.3,
                "disabled_until_timestep": 0.99,
            },
            "quant": {
                "dtype_qk": "int8",
                "q_block_size": 1,
                "k_block_size": 16,
            },
        },
        "per_role": {"minimax_h3.token_refiner": {"backend": "TRTLLM_ATTN"}},
    },
    "sage_int8_skip_05_gate099": {
        "default": {
            "backend": "TRTLLM_ATTN",
            "skip_softmax": {
                "threshold": 0.5,
                "disabled_until_timestep": 0.99,
            },
            "quant": {
                "dtype_qk": "int8",
                "q_block_size": 1,
                "k_block_size": 16,
            },
        },
        "per_role": {"minimax_h3.token_refiner": {"backend": "TRTLLM_ATTN"}},
    },
}


def _skip_softmax_mode_config(
    *,
    threshold: float,
    disabled_until_timestep: float,
    sage_dtype: str | None,
) -> dict[str, object]:
    default: dict[str, object] = {
        "backend": "TRTLLM_ATTN",
        "skip_softmax": {
            "threshold": threshold,
            "disabled_until_timestep": disabled_until_timestep,
        },
    }
    if sage_dtype is not None:
        default["quant"] = {
            "dtype_qk": sage_dtype,
            "q_block_size": 1,
            "k_block_size": 16,
        }
    return {
        "default": default,
        "per_role": {"minimax_h3.token_refiner": {"backend": "TRTLLM_ATTN"}},
    }


for threshold_name, threshold in {
    "005": 0.05,
    "010": 0.10,
    "03": 0.30,
    "05": 0.50,
}.items():
    for gate_name, gate in {"090": 0.90, "095": 0.95, "099": 0.99}.items():
        suffix = f"{threshold_name}_gate{gate_name}"
        MODE_CONFIGS[f"skip_softmax_{suffix}"] = _skip_softmax_mode_config(
            threshold=threshold,
            disabled_until_timestep=gate,
            sage_dtype=None,
        )
        MODE_CONFIGS[f"sage_fp8_skip_{suffix}"] = _skip_softmax_mode_config(
            threshold=threshold,
            disabled_until_timestep=gate,
            sage_dtype="fp8_e4m3",
        )


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _hardware() -> list[dict[str, object]]:
    result = []
    for device_index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_index)
        result.append(
            {
                "logical_index": device_index,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gib": props.total_memory / 2**30,
            }
        )
    return result


def main() -> None:
    mode = os.environ["MODE"]
    if mode not in MODE_CONFIGS:
        raise ValueError(f"MODE must be one of {sorted(MODE_CONFIGS)}, got {mode!r}")

    model_dir = Path(os.environ["MODEL_DIR"])
    output_root = Path(os.environ["OUTPUT_ROOT"])
    output_root.mkdir(parents=True, exist_ok=True)
    height = int(os.environ.get("HEIGHT", "768"))
    width = int(os.environ.get("WIDTH", "1248"))
    duration = float(os.environ.get("DURATION_SECONDS", "8.7"))
    steps = int(os.environ.get("NUM_INFERENCE_STEPS", "50"))
    num_runs = int(os.environ.get("NUM_RUNS", "3"))
    video_runs_value = os.environ.get("VIDEO_RUNS", "")
    if video_runs_value.lower() == "none":
        video_runs: set[int] | None = set()
    elif video_runs_value:
        video_runs = {int(item) for item in video_runs_value.split(",") if item}
    else:
        video_runs = None
    seed = int(os.environ.get("SEED", "1101"))
    text_encoder_tp_size = int(os.environ.get("TEXT_ENCODER_TP_SIZE", "1"))
    enforce_eager = os.environ.get("ENFORCE_EAGER", "0") == "1"
    async_output_timeout = float(os.environ.get("ASYNC_OUTPUT_TIMEOUT_SECONDS", "1800"))
    diffusion_engine._ASYNC_OUTPUT_TIMEOUT = async_output_timeout

    hardware = _hardware()
    if len(hardware) != 4:
        raise RuntimeError(f"Expected four visible GPUs, found {len(hardware)}")

    attention_config = MODE_CONFIGS[mode]
    print(
        "BENCH_CONFIG "
        + json.dumps(
            {
                "mode": mode,
                "model": str(model_dir),
                "attention_config": attention_config,
                "hardware": hardware,
                "height": height,
                "width": width,
                "duration_seconds": duration,
                "num_inference_steps": steps,
                "num_runs": num_runs,
                "video_runs": sorted(video_runs) if video_runs is not None else "all",
                "video_encoding": "deferred_until_after_all_timed_runs",
                "seed": seed,
                "text_encoder_tp_size": text_encoder_tp_size,
                "enforce_eager": enforce_eager,
                "async_output_timeout_seconds": async_output_timeout,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    engine = Omni(
        model=str(model_dir),
        parallel_config=DiffusionParallelConfig(
            # Exact recipe profile: Ulysses=4, Ring=1, DiT TP=1, VAE tile PP=4.
            tensor_parallel_size=1,
            ulysses_degree=4,
            ring_degree=1,
            text_encoder_tp_size=text_encoder_tp_size,
            vae_patch_parallel_size=4,
            vae_parallel_mode="tile",
        ),
        trust_remote_code=True,
        enable_cpu_offload=False,
        enforce_eager=enforce_eager,
        diffusion_attention_config=attention_config,
        enable_diffusion_pipeline_profiler=True,
    )

    records: list[dict[str, object]] = []
    pending_videos: list[tuple[Path, np.ndarray, int, np.ndarray, int]] = []
    try:
        for run_index in range(num_runs):
            started = time.perf_counter()
            outputs = engine.generate(
                PROMPT,
                OmniDiffusionSamplingParams(
                    height=height,
                    width=width,
                    fps=24,
                    num_inference_steps=steps,
                    seed=seed,
                    output_type="np",
                    extra_args={
                        "task": "t2va",
                        "aspect_ratio": "16:9",
                        "duration": duration,
                        "flow_shift": 12.0,
                        "audio_flow_shift": 3.0,
                    },
                ),
                use_tqdm=False,
            )
            wall_time = time.perf_counter() - started
            if len(outputs) != 1:
                raise RuntimeError(f"Expected one output, found {len(outputs)}")

            result = outputs[0]
            frames = np.asarray(result.images[0])
            multimodal = result.multimodal_output
            if multimodal is None:
                raise RuntimeError("MiniMax-H3 returned no audio metadata")
            audio = np.asarray(multimodal["audio"])
            fps = int(multimodal["fps"])
            sample_rate = int(multimodal["audio_sample_rate"])
            if frames.ndim != 4 or tuple(frames.shape[1:]) != (height, width, 3):
                raise RuntimeError(f"Unexpected video shape: {frames.shape}")
            if audio.ndim not in (2, 3) or 2 not in audio.shape:
                raise RuntimeError(f"Unexpected audio shape: {audio.shape}")
            if fps != 24 or sample_rate != 32000:
                raise RuntimeError(
                    f"Unexpected media rates: fps={fps}, audio={sample_rate}"
                )

            run_number = run_index + 1
            output_path = None
            if video_runs is None or run_number in video_runs:
                output_path = output_root / f"t2va_{mode}_run{run_number}.mp4"
                pending_videos.append(
                    (output_path, frames.copy(), fps, audio.copy(), sample_rate)
                )
            record = {
                "run": run_number,
                "warmup": run_index == 0,
                "wall_time_s": wall_time,
                "stage_durations": dict(getattr(result, "stage_durations", {}) or {}),
                "worker_peak_memory_mb": float(
                    getattr(result, "peak_memory_mb", 0.0) or 0.0
                ),
                "frames_shape": list(frames.shape),
                "audio_shape": list(audio.shape),
                "fps": fps,
                "audio_sample_rate": sample_rate,
                "frames_sha256": array_sha256(frames),
                "audio_sha256": array_sha256(audio),
                "mp4": str(output_path) if output_path is not None else None,
            }
            records.append(record)
            print("RUN_RESULT " + json.dumps(record, sort_keys=True), flush=True)
    finally:
        engine.close()

    for output_path, frames, fps, audio, sample_rate in pending_videos:
        output_path.write_bytes(
            _encode_video_bytes(
                frames,
                fps=fps,
                audio=audio,
                audio_sample_rate=sample_rate,
            )
        )

    steady_records = records[1:] if len(records) > 1 else records
    output_reference = (
        steady_records[0]["frames_sha256"],
        steady_records[0]["audio_sha256"],
    )
    summary = {
        "mode": mode,
        "model": str(model_dir),
        "hardware": hardware,
        "torch_version": torch.__version__,
        "flashinfer_version": metadata.version("flashinfer-python"),
        "parallel_config": (
            f"tp1_ulysses4_ring1_text_encoder_tp{text_encoder_tp_size}_vae_tile4"
        ),
        "attention_config": attention_config,
        "regional_compile": not enforce_eager,
        "prompt": PROMPT,
        "seed": seed,
        "height": height,
        "width": width,
        "duration_seconds": duration,
        "num_inference_steps": steps,
        "video_runs": sorted(video_runs) if video_runs is not None else "all",
        "video_encoding": "deferred_until_after_all_timed_runs",
        "runs": records,
        "steady_output_deterministic": all(
            (record["frames_sha256"], record["audio_sha256"]) == output_reference
            for record in steady_records[1:]
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

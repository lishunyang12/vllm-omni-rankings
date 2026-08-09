#!/usr/bin/env python3
"""Benchmark one fixed-width MiniMax-H3 multimodal DLO DP/SP point."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import multiprocessing
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.entrypoints.async_omni import AsyncOmni


PROMPTS = (
    "At night, three cats march into a bedroom playing tiny brass instruments, then file out.",
    "A paper boat crosses a rain-filled street while distant traffic and water sounds remain synchronized.",
    "A glassblower shapes a glowing vase while the furnace crackles softly in the workshop.",
    "A red tram climbs a snowy hill as its bell rings and wind moves through pine trees.",
    "A robot carefully waters sunflowers in a bright greenhouse with gentle mechanical sounds.",
    "Ocean waves roll around a lighthouse at sunset while seabirds circle overhead.",
    "A chef tosses vegetables in a wok as steam rises and the kitchen ambience stays synchronized.",
    "A toy train circles a miniature village while tiny station bells ring in the distance.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dp", type=int, required=True)
    parser.add_argument("--sp", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--batch-wait-ms", type=float, default=500.0)
    parser.add_argument("--init-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--task", choices=("t2va", "fl2va", "ref2va"), default="t2va"
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.task in {"fl2va", "ref2va"} and args.image is None:
        parser.error(f"--image is required for --task {args.task}")
    if args.task == "ref2va" and args.audio is None:
        parser.error("--audio is required for --task ref2va")
    for path in (args.image, args.audio):
        if path is not None and not path.is_file():
            parser.error(f"input asset does not exist: {path}")
    return args


def make_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.dp * args.sp != args.num_gpus:
        raise ValueError(
            f"dp * sp must equal num_gpus, got {args.dp} * {args.sp} != {args.num_gpus}"
        )
    return {
        "model": args.model,
        "trust_remote_code": True,
        "num_gpus": args.num_gpus,
        "tensor_parallel_size": 1,
        "data_parallel_size": args.dp,
        "ulysses_degree": args.sp,
        "ring_degree": 1,
        "text_encoder_tp_size": 1,
        "vae_patch_parallel_size": 1,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "diffusion_attention_backend": "CUDNN_ATTN",
        "request_batch_max_wait_ms": args.batch_wait_ms,
        "enforce_eager": True,
        "stage_init_timeout": args.init_timeout,
        "init_timeout": args.init_timeout,
        "enable_distributed_layerwise_offload": True,
        "dlo_use_allgather": os.environ.get("DLO_USE_ALLGATHER", "0") == "1",
        "dlo_resident_layers": 0,
    }


def make_sampling_params(
    engine: AsyncOmni,
    args: argparse.Namespace,
    *,
    seed: int,
    steps: int,
) -> list[Any]:
    params = copy.deepcopy(engine.default_sampling_params_list)
    diffusion = params[0]
    diffusion.width = args.width
    diffusion.height = args.height
    diffusion.fps = 24
    diffusion.num_inference_steps = steps
    diffusion.seed = seed
    diffusion.extra_args = {
        "task": args.task,
        "duration": args.duration,
        "aspect_ratio": "16:9",
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }
    return params


async def generate_one(
    engine: AsyncOmni,
    args: argparse.Namespace,
    *,
    request_id: str,
    prompt: Any,
    seed: int,
    steps: int,
) -> tuple[Any, float]:
    started = time.perf_counter()
    final_output = None
    async for output in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=make_sampling_params(
            engine, args, seed=seed, steps=steps
        ),
    ):
        if output.finished:
            final_output = output
    elapsed = time.perf_counter() - started
    if final_output is None:
        raise RuntimeError(f"{request_id} finished without an output")
    return final_output, elapsed


def make_prompt(args: argparse.Namespace, index: int) -> Any:
    prompt = PROMPTS[index]
    if args.task == "t2va":
        return prompt
    multi_modal_data = {"image": str(args.image)}
    if args.task == "ref2va":
        multi_modal_data["audio"] = str(args.audio)
    return {"prompt": prompt, "multi_modal_data": multi_modal_data}


def summarize_output(output: Any, args: argparse.Namespace, latency_s: float) -> dict[str, Any]:
    if not output.images:
        raise RuntimeError(f"{output.request_id} returned no video")
    frames = np.asarray(output.images[0])
    audio = np.asarray(output.multimodal_output.get("audio"))
    if tuple(frames.shape) != (124, args.height, args.width, 3):
        raise RuntimeError(
            f"{output.request_id} invalid video shape: {tuple(frames.shape)}"
        )
    if tuple(audio.shape) != (1, 2, 165600):
        raise RuntimeError(
            f"{output.request_id} invalid audio shape: {tuple(audio.shape)}"
        )
    return {
        "request_id": output.request_id,
        "latency_s": latency_s,
        "video_shape": list(frames.shape),
        "audio_shape": list(audio.shape),
        "worker_peak_memory_mb": output.peak_memory_mb,
        "stage_durations": output.stage_durations,
    }


async def run_wave(
    engine: AsyncOmni,
    args: argparse.Namespace,
    *,
    prefix: str,
    seed_base: int,
    steps: int,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    outputs = await asyncio.gather(
        *(
            generate_one(
                engine,
                args,
                request_id=f"{prefix}-{index}",
                prompt=make_prompt(args, index),
                seed=seed_base + index,
                steps=steps,
            )
            for index in range(args.dp)
        )
    )
    wave_s = time.perf_counter() - started
    summaries = [
        summarize_output(output, args, latency_s)
        for output, latency_s in outputs
    ]
    return summaries, wave_s


class GpuSampler:
    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.peak_memory_mib: dict[int, int] = {}
        self.peak_utilization_pct: dict[int, int] = {}
        self.errors: list[str] = []

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _run(self) -> None:
        query = (
            "index,memory.used,utilization.gpu"
        )
        while not self.stop_event.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for row in completed.stdout.splitlines():
                    index, memory, utilization = (
                        int(value.strip()) for value in row.split(",")
                    )
                    self.peak_memory_mib[index] = max(
                        memory, self.peak_memory_mib.get(index, 0)
                    )
                    self.peak_utilization_pct[index] = max(
                        utilization, self.peak_utilization_pct.get(index, 0)
                    )
            except Exception as exc:
                if len(self.errors) < 5:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
            self.stop_event.wait(self.interval_s)

    def summary(self) -> dict[str, Any]:
        return {
            "per_gpu_peak_memory_mib": self.peak_memory_mib,
            "per_gpu_peak_utilization_pct": self.peak_utilization_pct,
            "max_peak_memory_mib": max(self.peak_memory_mib.values(), default=0),
            "errors": self.errors,
        }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = make_engine_kwargs(args)
    summary: dict[str, Any] = {
        "status": "running",
        "topology": f"dp{args.dp}-sp{args.sp}",
        "engine_kwargs": kwargs,
        "workload": {
            "task": args.task,
            "image": str(args.image) if args.image is not None else None,
            "audio": str(args.audio) if args.audio is not None else None,
            "concurrency": args.dp,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "repeats": args.repeats,
            "height": args.height,
            "width": args.width,
            "duration_s": args.duration,
            "expected_frames": 124,
        },
    }
    sampler = GpuSampler()
    sampler.start()
    engine = None
    try:
        summary["engine_init_started_wall_s"] = time.time()
        started = time.perf_counter()
        engine = AsyncOmni(**kwargs)
        summary["engine_init_s"] = time.perf_counter() - started
        summary["engine_init_finished_wall_s"] = time.time()
        print(
            f"PROGRESS {summary['topology']} initialized in {summary['engine_init_s']:.3f}s",
            flush=True,
        )

        summary["warmup_started_wall_s"] = time.time()
        warmup_outputs, warmup_wave_s = await run_wave(
            engine,
            args,
            prefix="warmup",
            seed_base=1000,
            steps=args.warmup_steps,
        )
        summary["warmup_wave_s"] = warmup_wave_s
        summary["warmup_finished_wall_s"] = time.time()
        summary["warmup_outputs"] = warmup_outputs
        print(
            f"PROGRESS {summary['topology']} warmup completed in {warmup_wave_s:.3f}s",
            flush=True,
        )

        measured_waves = []
        for repeat in range(args.repeats):
            measured_started_wall_s = time.time()
            outputs, wave_s = await run_wave(
                engine,
                args,
                prefix=f"measured-{repeat}",
                seed_base=2000 + 100 * repeat,
                steps=args.steps,
            )
            measured_waves.append(
                {
                    "repeat": repeat,
                    "started_wall_s": measured_started_wall_s,
                    "finished_wall_s": time.time(),
                    "wave_s": wave_s,
                    "videos_per_second": args.dp / wave_s,
                    "videos_per_hour": 3600.0 * args.dp / wave_s,
                    "outputs": outputs,
                }
            )
            print(
                f"PROGRESS {summary['topology']} measured-{repeat} "
                f"wave={wave_s:.3f}s throughput={3600.0 * args.dp / wave_s:.3f} videos/hour",
                flush=True,
            )
        summary["measured_waves"] = measured_waves
        summary["mean_wave_s"] = sum(
            wave["wave_s"] for wave in measured_waves
        ) / len(measured_waves)
        summary["mean_videos_per_hour"] = sum(
            wave["videos_per_hour"] for wave in measured_waves
        ) / len(measured_waves)
        summary["status"] = "passed"
    finally:
        if engine is not None:
            started = time.perf_counter()
            engine.close()
            summary["shutdown_s"] = time.perf_counter() - started
        sampler.stop()
        summary["gpu_sampler"] = sampler.summary()

    children = multiprocessing.active_children()
    for child in children:
        child.join(timeout=30)
    children = multiprocessing.active_children()
    summary["active_children"] = [
        {"name": child.name, "pid": child.pid} for child in children
    ]
    summary["cleanup_status"] = "clean" if not children else "delayed-reap"
    return summary


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    try:
        summary = asyncio.run(run(args))
    except BaseException as exc:
        summary = {
            "status": "failed",
            "topology": f"dp{args.dp}-sp{args.sp}",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        write_summary(args.output, summary)
        print(f"E2E_FAILURE {json.dumps(summary, sort_keys=True)}", flush=True)
        raise
    write_summary(args.output, summary)
    print(f"E2E_RESULT {json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()

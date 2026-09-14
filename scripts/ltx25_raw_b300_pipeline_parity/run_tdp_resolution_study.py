#!/usr/bin/env python3
"""Run the resident-server LTX-2.5 Global-SP versus Stage-2 TDP study.

The controlled matrix uses the official long quickstart prompt, ten fixed
seeds, three delivery resolutions, 121 frames, 24 FPS, and the public sync
videos API.  One four-GPU server stays resident for the whole run.  Each
resolution/mode pair receives one excluded warm-up before measured requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

import pynvml
import requests

from run_pipeline_parity import PROMPT


MODEL_CLASS = "LTX2DistilledTwoStagePipeline"
NUM_FRAMES = 121
FPS = 24
NUM_INFERENCE_STEPS = 8
ALIGNMENT = 64
STAGE_2_SIGMAS = (0.625, 0.4, 0.0)
DEFAULT_SEEDS = tuple(range(42, 52))


@dataclass(frozen=True)
class Resolution:
    label: str
    width: int
    height: int

    @property
    def internal_width(self) -> int:
        return math.ceil(self.width / ALIGNMENT) * ALIGNMENT

    @property
    def internal_height(self) -> int:
        return math.ceil(self.height / ALIGNMENT) * ALIGNMENT


RESOLUTIONS = {
    item.label: item
    for item in (
        Resolution("1080p", 1920, 1080),
        Resolution("dci2k", 2048, 1080),
        Resolution("qhd2k", 2560, 1440),
    )
}
MODES = ("global_sp", "stage2_tdp")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("LTX25_OMNI_MODEL_ROOT", "Lightricks/LTX-2.5-Diffusers"),
    )
    parser.add_argument(
        "--vllm-omni-root",
        type=Path,
        default=Path(os.environ.get("VLLM_OMNI_ROOT", ".")),
    )
    parser.add_argument(
        "--vllm-bin",
        default=os.environ.get("VLLM_BIN", "vllm"),
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--attention-backend", default="CUDNN_ATTN")
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--warmup-seed", type=int, default=41)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--resolutions",
        nargs="+",
        choices=tuple(RESOLUTIONS),
        default=list(RESOLUTIONS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/dev/shm/ltx25_tdp_resolution_study"),
    )
    parser.add_argument("--server-ready-timeout", type=positive_float, default=1800.0)
    parser.add_argument("--request-timeout", type=positive_float, default=7200.0)
    parser.add_argument("--memory-poll-seconds", type=positive_float, default=0.1)
    parser.add_argument(
        "--force", action="store_true", help="Rerun and overwrite completed cases."
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(f"Executable not found on PATH: {value}")
    return resolved


def parse_gpu_indices(value: str) -> tuple[int, ...]:
    parts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(parts) != 4 or len(set(parts)) != 4 or any(item < 0 for item in parts):
        raise ValueError(
            "--gpus must contain exactly four distinct non-negative GPU indices"
        )
    return parts


def server_command(args: argparse.Namespace, vllm_bin: str) -> list[str]:
    return [
        vllm_bin,
        "serve",
        args.model,
        "--omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-class-name",
        MODEL_CLASS,
        "--diffusion-attention-backend",
        args.attention_backend,
        "--usp",
        "4",
        "--enforce-eager",
        "--enable-diffusion-pipeline-profiler",
    ]


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def wait_until_ready(
    session: requests.Session,
    process: subprocess.Popen,
    base_url: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"Resident server exited before readiness with code {returncode}"
            )
        try:
            response = session.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Resident server was not healthy within {timeout:.1f}s")


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10.0)


@contextmanager
def resident_server(
    args: argparse.Namespace,
    command: list[str],
    session: requests.Session,
    log_path: Path,
) -> Iterator[None]:
    if port_is_open(args.host, args.port):
        raise RuntimeError(f"Refusing to use occupied port {args.host}:{args.port}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["PYTHONUNBUFFERED"] = "1"
    root = str(args.vllm_omni_root.resolve())
    env["PYTHONPATH"] = root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=args.vllm_omni_root,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_until_ready(
                session,
                process,
                f"http://{args.host}:{args.port}",
                args.server_ready_timeout,
            )
            yield
        finally:
            stop_process_group(process)


class NVMLMemorySampler:
    def __init__(self, gpu_indices: tuple[int, ...], poll_seconds: float) -> None:
        self.handles = tuple(
            pynvml.nvmlDeviceGetHandleByIndex(index) for index in gpu_indices
        )
        self.poll_seconds = poll_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.baseline_mib: list[float] = []
        self.peak_mib: list[float] = []

    def _read(self) -> list[float]:
        return [
            pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)
            for handle in self.handles
        ]

    def _poll(self) -> None:
        while not self.stop_event.wait(self.poll_seconds):
            values = self._read()
            self.peak_mib = [
                max(old, new) for old, new in zip(self.peak_mib, values, strict=True)
            ]

    def __enter__(self) -> NVMLMemorySampler:
        self.baseline_mib = self._read()
        self.peak_mib = list(self.baseline_mib)
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        values = self._read()
        self.peak_mib = [
            max(old, new) for old, new in zip(self.peak_mib, values, strict=True)
        ]

    def result(self) -> dict[str, float | list[float]]:
        delta = [
            peak - base
            for peak, base in zip(self.peak_mib, self.baseline_mib, strict=True)
        ]
        return {
            "baseline_used_mib_per_gpu": self.baseline_mib,
            "peak_used_mib_per_gpu": self.peak_mib,
            "peak_delta_mib_per_gpu": delta,
            "max_peak_used_mib": max(self.peak_mib),
            "max_peak_delta_mib": max(delta),
        }


def request_form(
    resolution: Resolution,
    mode: str,
    seed: int,
    overlap: int,
) -> dict[str, str]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    tiled = mode == "stage2_tdp"
    extra_params: dict[str, object] = {"stage_2_sigmas": list(STAGE_2_SIGMAS)}
    if tiled:
        extra_params.update(
            {
                "ltx_tiled_data_parallel": True,
                "ltx_tiled_data_parallel_overlap": overlap,
            }
        )
    return {
        "prompt": PROMPT,
        "width": str(resolution.width if tiled else resolution.internal_width),
        "height": str(resolution.height if tiled else resolution.internal_height),
        "num_frames": str(NUM_FRAMES),
        "fps": str(FPS),
        "num_inference_steps": str(NUM_INFERENCE_STEPS),
        "seed": str(seed),
        "extra_params": json.dumps(extra_params, separators=(",", ":")),
    }


def send_request(
    session: requests.Session,
    *,
    base_url: str,
    form: dict[str, str],
    timeout: float,
    sampler: NVMLMemorySampler | None,
) -> tuple[float, bytes, dict[str, float | list[float]] | None]:
    context = sampler if sampler is not None else nullcontext()
    with context:
        started = time.perf_counter()
        response = session.post(
            f"{base_url}/v1/videos/sync",
            data=form,
            headers={"Accept": "video/mp4"},
            timeout=timeout,
        )
        body = response.content
        elapsed = time.perf_counter() - started
    if response.status_code != 200:
        detail = body.decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(
            f"Video request returned HTTP {response.status_code}: {detail}"
        )
    if len(body) < 12 or body[4:8] != b"ftyp":
        raise RuntimeError("Video request did not return a valid MP4 payload")
    return elapsed, body, None if sampler is None else sampler.result()


def probe_video(path: Path) -> dict[str, int | float]:
    completed = subprocess.run(
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
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "fps": int(numerator) / int(denominator),
    }


def write_bytes_atomic(path: Path, body: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(path)


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_revision(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout.strip() or None


def software_versions(python: str) -> dict[str, str]:
    code = (
        "import torch,vllm,vllm_omni;"
        "print(torch.__version__);print(torch.version.cuda);"
        "print(vllm.__version__);print(getattr(vllm_omni,'__version__','unknown'))"
    )
    completed = subprocess.run(
        [python, "-c", code], check=True, text=True, stdout=subprocess.PIPE
    )
    values = [
        line
        for line in completed.stdout.splitlines()
        if line and not line.startswith("INFO")
    ]
    return dict(zip(("torch", "cuda", "vllm", "vllm_omni"), values[-4:], strict=True))


def summarize_requests(requests_by_key: dict[str, dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for label in RESOLUTIONS:
        for mode in MODES:
            samples = [
                item["elapsed_seconds"]
                for item in requests_by_key.values()
                if item["resolution"] == label and item["mode"] == mode
            ]
            if not samples:
                continue
            summary[f"{label}/{mode}"] = {
                "samples": len(samples),
                "mean_seconds": statistics.fmean(samples),
                "min_seconds": min(samples),
                "max_seconds": max(samples),
                "stdev_seconds": statistics.pstdev(samples),
            }
    return summary


def new_result(
    args: argparse.Namespace, command: list[str], versions: dict[str, str]
) -> dict:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "model_class": MODEL_CLASS,
        "vllm_omni_revision": git_revision(args.vllm_omni_root),
        "software_versions": versions,
        "server_command": shlex.join(command),
        "gpus": args.gpus,
        "attention_backend": args.attention_backend,
        "contract": {
            "prompt": PROMPT,
            "seeds": args.seeds,
            "warmup_seed": args.warmup_seed,
            "resolutions": [asdict(RESOLUTIONS[label]) for label in args.resolutions],
            "internal_alignment": ALIGNMENT,
            "num_frames": NUM_FRAMES,
            "fps": FPS,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "stage_2_sigmas": STAGE_2_SIGMAS,
            "tdp_overlap_latent_cells": args.overlap,
            "warmup": "one excluded request per resolution and mode",
            "order": "modes alternate by seed after both shape-specific warm-ups",
            "latency_scope": "POST /v1/videos/sync through receipt of complete MP4; file write and probe excluded",
            "concurrency": 1,
        },
        "warmups": {},
        "requests": {},
        "summary": {},
    }


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if not args.vllm_omni_root.is_dir():
        raise FileNotFoundError(args.vllm_omni_root)
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.overlap < 0:
        raise ValueError("--overlap must be non-negative")
    if len(args.seeds) != 10 or len(set(args.seeds)) != 10:
        raise ValueError("--seeds must contain exactly ten distinct values")
    return parse_gpu_indices(args.gpus)


def main() -> None:
    args = parse_args()
    args.vllm_omni_root = args.vllm_omni_root.resolve()
    args.output_dir = args.output_dir.resolve()
    gpu_indices = validate_args(args)
    vllm_bin = resolve_executable(args.vllm_bin)
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe")
    command = server_command(args, vllm_bin)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "server_command": shlex.join(command),
                    "gpus": args.gpus,
                    "requests": {
                        f"{label}/{mode}": request_form(
                            RESOLUTIONS[label], mode, args.seeds[0], args.overlap
                        )
                        for label in args.resolutions
                        for mode in MODES
                    },
                },
                indent=2,
            )
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    manifest_path = args.output_dir / "results.json"
    server_log = args.output_dir / "server.log"
    versions = software_versions(str(Path(vllm_bin).with_name("python")))
    if manifest_path.is_file() and not args.force:
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        if result["vllm_omni_revision"] != git_revision(args.vllm_omni_root):
            raise ValueError(
                "Existing results use a different vLLM-Omni revision; pass --force or use a new directory"
            )
    else:
        result = new_result(args, command, versions)

    pynvml.nvmlInit()
    base_url = f"http://{args.host}:{args.port}"
    try:
        with requests.Session() as session:
            session.trust_env = False
            print(f"Launching resident server on GPUs {args.gpus}", flush=True)
            with resident_server(args, command, session, server_log):
                print("Resident server ready", flush=True)
                for label in args.resolutions:
                    resolution = RESOLUTIONS[label]
                    for mode in MODES:
                        warmup_key = f"{label}/{mode}"
                        print(
                            f"[{warmup_key}] warm-up seed {args.warmup_seed} (excluded)",
                            flush=True,
                        )
                        sampler = NVMLMemorySampler(
                            gpu_indices, args.memory_poll_seconds
                        )
                        elapsed, _body, memory = send_request(
                            session,
                            base_url=base_url,
                            form=request_form(
                                resolution, mode, args.warmup_seed, args.overlap
                            ),
                            timeout=args.request_timeout,
                            sampler=sampler,
                        )
                        result["warmups"][warmup_key] = {
                            "elapsed_seconds": elapsed,
                            "memory": memory,
                        }
                        result["updated_at_utc"] = datetime.now(UTC).isoformat()
                        write_json_atomic(manifest_path, result)

                    for seed_index, seed in enumerate(args.seeds):
                        mode_order = (
                            MODES if seed_index % 2 == 0 else tuple(reversed(MODES))
                        )
                        for mode in mode_order:
                            key = f"{label}/{mode}/seed-{seed}"
                            output_path = videos_dir / label / f"{mode}-seed-{seed}.mp4"
                            if (
                                key in result["requests"]
                                and output_path.is_file()
                                and not args.force
                            ):
                                print(f"[{key}] already complete; skipping", flush=True)
                                continue
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            print(f"[{key}] measured request", flush=True)
                            sampler = NVMLMemorySampler(
                                gpu_indices, args.memory_poll_seconds
                            )
                            elapsed, body, memory = send_request(
                                session,
                                base_url=base_url,
                                form=request_form(resolution, mode, seed, args.overlap),
                                timeout=args.request_timeout,
                                sampler=sampler,
                            )
                            write_bytes_atomic(output_path, body)
                            metadata = probe_video(output_path)
                            expected_width = (
                                resolution.width
                                if mode == "stage2_tdp"
                                else resolution.internal_width
                            )
                            expected_height = (
                                resolution.height
                                if mode == "stage2_tdp"
                                else resolution.internal_height
                            )
                            expected = {
                                "width": expected_width,
                                "height": expected_height,
                                "frames": NUM_FRAMES,
                                "fps": float(FPS),
                            }
                            if metadata != expected:
                                raise ValueError(
                                    f"Unexpected metadata for {key}: {metadata}; expected {expected}"
                                )
                            result["requests"][key] = {
                                "resolution": label,
                                "mode": mode,
                                "seed": seed,
                                "elapsed_seconds": elapsed,
                                "memory": memory,
                                "video": str(output_path.relative_to(args.output_dir)),
                                "bytes": len(body),
                                "sha256": hashlib.sha256(body).hexdigest(),
                                "metadata": metadata,
                                "request_form": request_form(
                                    resolution, mode, seed, args.overlap
                                ),
                            }
                            result["summary"] = summarize_requests(result["requests"])
                            result["updated_at_utc"] = datetime.now(UTC).isoformat()
                            write_json_atomic(manifest_path, result)
                            print(
                                f"[{key}] {elapsed:.3f}s, {len(body) / 1048576:.2f} MiB",
                                flush=True,
                            )
    finally:
        pynvml.nvmlShutdown()

    print(f"Study complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

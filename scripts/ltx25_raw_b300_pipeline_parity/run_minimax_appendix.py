#!/usr/bin/env python3
"""Benchmark MiniMax-H3 T2VA and first-frame FL2VA for the LTX gallery.

The harness starts one resident FL2VA-partition vLLM-Omni server, executes one
complete excluded warm-up per task, then times sequential requests through
receipt of the complete MP4 response.  Response-file writes and validation are
outside the reported latency.
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
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

import requests


WIDTH = 1344
HEIGHT = 768
FPS = 24
SEED = 42
DURATION_SECONDS = 5.0
NUM_INFERENCE_STEPS = 50
FLOW_SHIFT = 12.0
AUDIO_FLOW_SHIFT = 3.0
TASKS = ("t2va", "fl2va")


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
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="MiniMax-H3 FL2VA partition path or Hub ID.")
    parser.add_argument("--vllm-omni-root", type=Path, required=True)
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--gpu", default="0", help="Single physical GPU ID exposed to the server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18096)
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--contract",
        type=Path,
        default=here / "results-v2" / "contract.json",
        help="Gallery contract containing the exact difficult prompt.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=here / "minimax-appendix" / "inputs" / "quickstart-seed42-1344x768.png",
    )
    parser.add_argument("--attention-backend", default="TRTLLM_ATTN")
    parser.add_argument("--server-extra-args", default="")
    parser.add_argument("--server-ready-timeout", type=positive_float, default=1800.0)
    parser.add_argument("--request-timeout", type=positive_float, default=1800.0)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def server_command(args: argparse.Namespace, vllm_bin: str) -> list[str]:
    command = [
        vllm_bin,
        "serve",
        args.model,
        "--omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--task-type",
        "fl2va",
        "--num-gpus",
        "1",
        "--tensor-parallel-size",
        "1",
        "--usp",
        "1",
        "--ring",
        "1",
        "--text-encoder-tp-size",
        "1",
        "--vae-patch-parallel-size",
        "1",
        "--vae-parallel-mode",
        "tile",
        "--vae-use-tiling",
        "--diffusion-attention-backend",
        args.attention_backend,
    ]
    command.extend(shlex.split(args.server_extra_args))
    return command


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10.0)


def wait_until_ready(
    session: requests.Session,
    process: subprocess.Popen,
    base_url: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Resident server exited before readiness (code {process.returncode})")
        try:
            response = session.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Resident server was not healthy within {timeout:.1f}s")


@contextmanager
def resident_server(
    args: argparse.Namespace,
    command: list[str],
    session: requests.Session,
    log_path: Path,
) -> Iterator[None]:
    if port_is_open(args.host, args.port):
        raise RuntimeError(f"Refusing to use occupied address {args.host}:{args.port}")
    env = os.environ.copy()
    root = str(args.vllm_omni_root)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": str(int(args.request_timeout)),
            "PYTHONPATH": root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    with log_path.open("w") as log_stream:
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


def request_form(prompt: str, task: str) -> dict[str, str]:
    extra_params: dict[str, object] = {
        "task": task,
        "duration": DURATION_SECONDS,
        "audio_flow_shift": AUDIO_FLOW_SHIFT,
        "aspect_ratio": "16:9",
    }
    if task == "fl2va":
        extra_params["frame_indices"] = [0]
    return {
        "prompt": prompt,
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "fps": str(FPS),
        "num_inference_steps": str(NUM_INFERENCE_STEPS),
        "flow_shift": str(FLOW_SHIFT),
        "seed": str(SEED),
        "extra_params": json.dumps(extra_params, separators=(",", ":")),
    }


def send_request(
    session: requests.Session,
    *,
    base_url: str,
    prompt: str,
    task: str,
    image: Path,
    timeout: float,
) -> tuple[float, bytes]:
    image_stream = image.open("rb") if task == "fl2va" else None
    files = None
    if image_stream is not None:
        files = {"input_reference": (image.name, image_stream, "image/png")}
    try:
        started = time.perf_counter()
        response = session.post(
            f"{base_url}/v1/videos/sync",
            data=request_form(prompt, task),
            files=files,
            headers={"Accept": "video/mp4"},
            timeout=timeout,
        )
        body = response.content
        elapsed = time.perf_counter() - started
    finally:
        if image_stream is not None:
            image_stream.close()
    if response.status_code != 200:
        detail = body.decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"{task} returned HTTP {response.status_code}: {detail}")
    if len(body) < 12 or body[4:8] != b"ftyp":
        raise RuntimeError(f"{task} did not return an MP4 payload")
    return elapsed, body


def summarize(samples: list[float]) -> dict[str, object]:
    return {
        "timed_repeats": len(samples),
        "samples_seconds": samples,
        "mean_seconds": statistics.fmean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "stdev_seconds": statistics.pstdev(samples),
    }


def git_revision(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() or None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_description(gpu: str) -> str | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            gpu,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() or None


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.vllm_omni_root = args.vllm_omni_root.resolve()
    args.contract = args.contract.resolve()
    args.image = args.image.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.vllm_omni_root.is_dir():
        raise FileNotFoundError(args.vllm_omni_root)
    if not args.contract.is_file():
        raise FileNotFoundError(args.contract)
    if "fl2va" in args.tasks and not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    prompt = json.loads(args.contract.read_text())["prompt"]
    vllm_bin = resolve_executable(args.vllm_bin)
    command = server_command(args, vllm_bin)

    contract = {
        "name": "MiniMax-H3 appendix warm E2E",
        "prompt": prompt,
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "expected_frames": 124,
        "fps": FPS,
        "requested_duration_seconds": DURATION_SECONDS,
        "expected_mp4_duration_seconds": 5.175,
        "num_inference_steps_requested": NUM_INFERENCE_STEPS,
        "denoise_updates": 49,
        "flow_shift": FLOW_SHIFT,
        "audio_flow_shift": AUDIO_FLOW_SHIFT,
        "tasks": list(args.tasks),
        "task_partition": "FL2VA",
        "attention_backend": args.attention_backend,
        "precision": "BF16",
        "parallelism": "1 GPU, TP1, Ulysses1, Ring1, text-encoder TP1, VAE patch1",
        "offload": "disabled",
        "execution": "regional torch.compile; eager mode disabled",
        "latency_scope": "loopback POST /v1/videos/sync through complete MP4 receipt",
        "excluded": "server startup, model load, one full warm-up per task, response-file write, and validation",
        "warmup_requests_excluded_per_task": 1,
        "timed_repeats_per_task": args.repeats,
        "request_concurrency": 1,
        "i2v_image": str(args.image),
        "i2v_image_sha256": sha256(args.image),
        "comparison_caveat": (
            "Same hard prompt, seed, one-B300 hardware count, FPS, and "
            "approximately five-second playback as LTX; MiniMax uses "
            "1344x768/124 frames/TRTLLM_ATTN while LTX uses "
            "1920x1088/121 frames/cuDNN, so this is not equal-compute or "
            "same-backend."
        ),
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "server_command": shlex.join(command),
                    "server_environment": {"CUDA_VISIBLE_DEVICES": args.gpu},
                    "contract": contract,
                    "requests": {task: request_form(prompt, task) for task in args.tasks},
                },
                indent=2,
            )
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = args.output_dir / "videos"
    responses_dir.mkdir(exist_ok=True)
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "vllm_omni_revision": git_revision(args.vllm_omni_root),
        "physical_gpu": args.gpu,
        "gpu": gpu_description(args.gpu),
        "server_command": shlex.join(command),
        "benchmark_contract": contract,
        "tasks": {},
    }
    summary_path = args.output_dir / "warm-e2e.json"
    atomic_json(args.output_dir / "contract.json", contract)
    base_url = f"http://{args.host}:{args.port}"
    log_path = args.output_dir / "server.log"
    with requests.Session() as session:
        session.trust_env = False
        print("[server] launching resident MiniMax-H3 FL2VA partition", flush=True)
        with resident_server(args, command, session, log_path):
            print("[server] healthy", flush=True)
            for task in args.tasks:
                print(f"[{task}] full 50-step warm-up (excluded)", flush=True)
                send_request(
                    session,
                    base_url=base_url,
                    prompt=prompt,
                    task=task,
                    image=args.image,
                    timeout=args.request_timeout,
                )
                samples: list[float] = []
                last_body = b""
                for index in range(args.repeats):
                    elapsed, last_body = send_request(
                        session,
                        base_url=base_url,
                        prompt=prompt,
                        task=task,
                        image=args.image,
                        timeout=args.request_timeout,
                    )
                    samples.append(elapsed)
                    print(
                        f"[{task}] timed request {index + 1}/{args.repeats}: {elapsed:.3f}s",
                        flush=True,
                    )
                result["tasks"][task] = summarize(samples)
                response_path = responses_dir / f"minimax-h3-{task}-seed-{SEED}.mp4"
                response_path.write_bytes(last_body)
                atomic_json(summary_path, result)
        print("[server] stopped", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

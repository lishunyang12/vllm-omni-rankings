#!/usr/bin/env python3
"""Measure warm request E2E latency for the four public LTX-2.5 pipelines.

The harness starts one resident vLLM-Omni server per selected pipeline class.
For each resident server it sends one unmeasured warm-up request for T2V and
one for I2V, then measures a configurable number of sequential requests.  The
timer covers the loopback HTTP request through receipt of the complete MP4
response.  Server startup, model loading, warm-up, and optional response-file
writes are deliberately outside the reported latency.

This is a controlled serving workload, not the long-form quality showcase.
The prompt, seed, shape, frame count, FPS, I2V input, and CRF are fixed across
pipeline classes.  Full and distilled pipelines retain their public official
denoising schedules (30 and 8 Stage-1 steps, respectively).
"""

from __future__ import annotations

import argparse
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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

import requests

from run_pipeline_parity import NEGATIVE_PROMPT, PROMPT


CONTROLLED_WIDTH = 768
CONTROLLED_HEIGHT = 512
CONTROLLED_NUM_FRAMES = 121
CONTROLLED_FPS = 24
CONTROLLED_SEED = 42
CONTROLLED_IMAGE_CRF = 18


@dataclass(frozen=True)
class PipelineSpec:
    class_name: str
    label: str
    num_inference_steps: int
    guided: bool


PIPELINES = {
    "LTX2Pipeline": PipelineSpec(
        class_name="LTX2Pipeline",
        label="full_one_stage",
        num_inference_steps=30,
        guided=True,
    ),
    "LTX2TwoStagePipeline": PipelineSpec(
        class_name="LTX2TwoStagePipeline",
        label="full_two_stage",
        num_inference_steps=30,
        guided=True,
    ),
    "LTX2DistilledOneStagePipeline": PipelineSpec(
        class_name="LTX2DistilledOneStagePipeline",
        label="distilled_one_stage",
        num_inference_steps=8,
        guided=False,
    ),
    "LTX2DistilledTwoStagePipeline": PipelineSpec(
        class_name="LTX2DistilledTwoStagePipeline",
        label="distilled_two_stage",
        num_inference_steps=8,
        guided=False,
    ),
}


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
        help="Materialized LTX-2.5 Diffusers checkpoint path or Hub ID.",
    )
    parser.add_argument(
        "--vllm-omni-root",
        type=Path,
        default=Path(os.environ.get("VLLM_OMNI_ROOT", ".")),
        help="vLLM-Omni source root placed first on PYTHONPATH for the server.",
    )
    parser.add_argument(
        "--vllm-bin",
        default=os.environ.get("VLLM_BIN", "vllm"),
        help="vllm executable used to launch each resident server.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=tuple(PIPELINES),
        default=list(PIPELINES),
        help="Public pipeline classes to benchmark sequentially.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("t2v", "i2v"),
        default=["t2v", "i2v"],
        help="Tasks measured on each resident pipeline server.",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=3,
        help="Timed warm requests per pipeline/task after one excluded warm-up.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).parent / "inputs" / "quickstart-seed42-frame0.png",
        help="Common first-frame image for every I2V request.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument(
        "--attention-backend",
        default="CUDNN_ATTN",
        help="Server diffusion attention backend, or 'platform-default' to omit the flag.",
    )
    parser.add_argument(
        "--server-extra-args",
        default="",
        help="Additional vllm serve arguments parsed with shell quoting.",
    )
    parser.add_argument(
        "--server-ready-timeout",
        type=positive_float,
        default=1800.0,
        help="Maximum unmeasured server-start wait in seconds.",
    )
    parser.add_argument(
        "--request-timeout",
        type=positive_float,
        default=7200.0,
        help="Maximum time for each warm-up or timed sync request.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results-warm-e2e",
    )
    parser.add_argument(
        "--output-json",
        default="warm-e2e.json",
        help="Summary filename under --output-dir.",
    )
    parser.add_argument(
        "--save-responses",
        action="store_true",
        help="Save the final timed MP4 per pipeline/task after the timer stops.",
    )
    parser.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="Do not pass --enforce-eager to the resident server.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print server commands and the controlled request contract without launching GPUs.",
    )
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


def server_command(args: argparse.Namespace, spec: PipelineSpec, vllm_bin: str) -> list[str]:
    command = [
        vllm_bin,
        "serve",
        args.model,
        "--omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-class-name",
        spec.class_name,
    ]
    if args.attention_backend != "platform-default":
        command += ["--diffusion-attention-backend", args.attention_backend]
    if not args.no_enforce_eager:
        command.append("--enforce-eager")
    command.extend(shlex.split(args.server_extra_args))
    return command


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
            raise RuntimeError(f"Resident server exited before readiness (code {returncode})")
        try:
            response = session.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Resident server did not become healthy within {timeout:.1f}s")


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
    spec: PipelineSpec,
    command: list[str],
    session: requests.Session,
    log_path: Path,
) -> Iterator[None]:
    if port_is_open(args.host, args.port):
        raise RuntimeError(
            f"Refusing to launch {spec.class_name}: {args.host}:{args.port} is already in use"
        )

    env = os.environ.copy()
    root = str(args.vllm_omni_root.resolve())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
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


def request_form(spec: PipelineSpec) -> dict[str, str]:
    form = {
        "prompt": PROMPT,
        "width": str(CONTROLLED_WIDTH),
        "height": str(CONTROLLED_HEIGHT),
        "num_frames": str(CONTROLLED_NUM_FRAMES),
        "fps": str(CONTROLLED_FPS),
        "num_inference_steps": str(spec.num_inference_steps),
        "seed": str(CONTROLLED_SEED),
    }
    if spec.guided:
        form["negative_prompt"] = NEGATIVE_PROMPT
    return form


def send_request(
    session: requests.Session,
    *,
    base_url: str,
    spec: PipelineSpec,
    task: str,
    image: Path,
    timeout: float,
) -> tuple[float, bytes]:
    form = request_form(spec)
    files = None
    image_stream = None
    if task == "i2v":
        form["extra_params"] = json.dumps(
            {"image_crf": CONTROLLED_IMAGE_CRF}, separators=(",", ":")
        )
        image_stream = image.open("rb")
        files = {"input_reference": (image.name, image_stream, "image/png")}

    try:
        started = time.perf_counter()
        response = session.post(
            f"{base_url}/v1/videos/sync",
            data=form,
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
        detail = body.decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"{spec.class_name} {task} returned HTTP {response.status_code}: {detail}")
    if len(body) < 12 or body[4:8] != b"ftyp":
        raise RuntimeError(f"{spec.class_name} {task} did not return a valid MP4 payload")
    return elapsed, body


def summarize(samples: list[float]) -> dict[str, float | int | list[float]]:
    return {
        "timed_repeats": len(samples),
        "samples_seconds": samples,
        "mean_seconds": statistics.fmean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "stdev_seconds": statistics.pstdev(samples),
    }


def write_summary(path: Path, result: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


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


def benchmark_contract(args: argparse.Namespace) -> dict:
    return {
        "name": "LTX-2.5 controlled warm E2E",
        "latency_scope": (
            "client wall clock for POST /v1/videos/sync through complete MP4 receipt; "
            "server startup/model load, one warm-up per pipeline/task, validation, and optional file writes excluded"
        ),
        "resident_runtime": "one server per public pipeline class; T2V and I2V run sequentially on that server",
        "warmup_requests_excluded_per_pipeline_task": 1,
        "timed_repeats_per_pipeline_task": args.repeats,
        "prompt": PROMPT,
        "seed": CONTROLLED_SEED,
        "width": CONTROLLED_WIDTH,
        "height": CONTROLLED_HEIGHT,
        "num_frames": CONTROLLED_NUM_FRAMES,
        "fps": CONTROLLED_FPS,
        "i2v_image": str(args.image),
        "i2v_image_crf": CONTROLLED_IMAGE_CRF,
        "schedule_policy": "public official schedule: Full=30 Stage-1 steps, Distilled=8 Stage-1 steps; two-stage Stage 2 is pipeline-defined",
        "request_order": list(args.tasks),
        "request_concurrency": 1,
        "transport": "persistent loopback HTTP session",
        "showcase_separation": "This compact controlled workload is not the 1080p/481-frame quality showcase.",
    }


def dry_run_payload(args: argparse.Namespace, vllm_bin: str) -> dict:
    return {
        "benchmark_contract": benchmark_contract(args),
        "servers": {
            class_name: {
                "command": shlex.join(server_command(args, PIPELINES[class_name], vllm_bin)),
                "tasks": {
                    task: {
                        "form": {
                            **request_form(PIPELINES[class_name]),
                            **(
                                {"extra_params": json.dumps({"image_crf": CONTROLLED_IMAGE_CRF})}
                                if task == "i2v"
                                else {}
                            ),
                        },
                        "input_reference": str(args.image) if task == "i2v" else None,
                    }
                    for task in args.tasks
                },
            }
            for class_name in args.pipelines
        },
    }


def main() -> None:
    args = parse_args()
    args.vllm_omni_root = args.vllm_omni_root.resolve()
    args.image = args.image.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.vllm_omni_root.is_dir():
        raise FileNotFoundError(args.vllm_omni_root)
    if "i2v" in args.tasks and not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    vllm_bin = resolve_executable(args.vllm_bin)

    if args.dry_run:
        print(json.dumps(dry_run_payload(args, vllm_bin), indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / args.output_json
    responses_dir = args.output_dir / "responses"
    if args.save_responses:
        responses_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "vllm_omni_revision": git_revision(args.vllm_omni_root),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "attention_backend": args.attention_backend,
        "benchmark_contract": benchmark_contract(args),
        "pipelines": {},
    }
    base_url = f"http://{args.host}:{args.port}"
    with requests.Session() as session:
        session.trust_env = False
        for class_name in args.pipelines:
            spec = PIPELINES[class_name]
            command = server_command(args, spec, vllm_bin)
            log_path = args.output_dir / f"server-{spec.label}.log"
            print(f"[{class_name}] launching resident server", flush=True)
            pipeline_result = {
                "label": spec.label,
                "num_inference_steps": spec.num_inference_steps,
                "guided": spec.guided,
                "server_command": shlex.join(command),
                "warm_e2e_seconds": {},
            }
            result["pipelines"][class_name] = pipeline_result
            with resident_server(args, spec, command, session, log_path):
                print(f"[{class_name}] resident server ready", flush=True)
                for task in args.tasks:
                    print(f"[{class_name} {task}] warm-up (excluded)", flush=True)
                    send_request(
                        session,
                        base_url=base_url,
                        spec=spec,
                        task=task,
                        image=args.image,
                        timeout=args.request_timeout,
                    )
                    samples: list[float] = []
                    last_body = b""
                    for repeat_index in range(args.repeats):
                        elapsed, last_body = send_request(
                            session,
                            base_url=base_url,
                            spec=spec,
                            task=task,
                            image=args.image,
                            timeout=args.request_timeout,
                        )
                        samples.append(elapsed)
                        print(
                            f"[{class_name} {task}] timed warm request {repeat_index + 1}/{args.repeats} complete",
                            flush=True,
                        )
                    pipeline_result["warm_e2e_seconds"][task] = summarize(samples)
                    if args.save_responses:
                        response_path = responses_dir / f"{spec.label}-{task}-seed-{CONTROLLED_SEED}.mp4"
                        response_path.write_bytes(last_body)
                    write_summary(summary_path, result)
            print(f"[{class_name}] resident server stopped", flush=True)

    print(f"Warm E2E summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()

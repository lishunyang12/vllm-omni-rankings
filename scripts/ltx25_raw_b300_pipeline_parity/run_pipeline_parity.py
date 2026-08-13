#!/usr/bin/env python3
"""Run the official LTX-2.5 and vLLM-Omni pipeline parity matrix.

Each invocation owns one backend so the official and Omni sides can run in
parallel on separate GPUs.  Results are named deterministically and recorded
in a backend-specific manifest to avoid concurrent-writer races.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from collections.abc import Iterable
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROMPT = (
    "A medium close-up shot features a Caucasian man with a beard, wearing a green and white "
    "baseball cap without any letters on the front, and a light blue shirt over a white t-shirt. "
    "He is positioned in the center of the frame, looking intently directly at the camera, his "
    "eyes focused on camera. His facial expression is one of deep concentration, with his brow "
    "slightly raised. As he looks straight at the camera, a quick sniff sound is heard, and then "
    "he speaks with a deep male voice and a satisfied tone, saying, 'I think it's so good.' The "
    "camera remains static throughout, maintaining a shallow depth of field, which keeps the man "
    "in sharp focus while the background is softly blurred, showing a beige wall behind him. "
    "After a brief pause, another short, audible sniff is heard. The man then continues to speak, "
    "his voice maintaining the same quality, as he states, 'So good. So good.' He elaborates "
    "further, emphasizing his point with a final statement, 'This got to be, it's got to be the "
    "best tool I've ever seen.'"
)

NEGATIVE_PROMPT = (
    "has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, "
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive "
    "noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, "
    "unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, "
    "extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent "
    "perspective, camera shake, incorrect depth of field, background too sharp, background clutter, "
    "distracting reflections, harsh shadows, inconsistent lighting direction, color banding, "
    "cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect "
    "ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, "
    "silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, "
    "incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, "
    "incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

NUM_FRAMES = 481
FPS = 24
SEEDS = (42, 43, 44)
OFFICIAL_OPENIMAGEIO = "3.1.11.0"


@dataclass(frozen=True)
class PipelineCase:
    official_module: str
    model_class_name: str
    checkpoint_type: str
    width: int
    height: int
    num_inference_steps: int
    guided: bool
    spatial_upsampler: bool
    distilled_lora: bool = False


CASES = {
    "full_one_stage": PipelineCase(
        official_module="ltx_pipelines.ti2vid_one_stage",
        model_class_name="LTX2Pipeline",
        checkpoint_type="full",
        width=960,
        height=544,
        num_inference_steps=30,
        guided=True,
        spatial_upsampler=False,
    ),
    "distilled_one_stage": PipelineCase(
        official_module="ltx_pipelines.distilled",
        model_class_name="LTX2DistilledOneStagePipeline",
        checkpoint_type="distilled",
        width=960,
        height=544,
        num_inference_steps=8,
        guided=False,
        spatial_upsampler=False,
    ),
    "distilled_two_stage": PipelineCase(
        official_module="ltx_pipelines.distilled",
        model_class_name="LTX2DistilledTwoStagePipeline",
        checkpoint_type="distilled",
        width=1920,
        height=1088,
        num_inference_steps=8,
        guided=False,
        spatial_upsampler=True,
    ),
    "full_two_stage": PipelineCase(
        official_module="ltx_pipelines.ti2vid_two_stages",
        model_class_name="LTX2TwoStagePipeline",
        checkpoint_type="full",
        width=1920,
        height=1088,
        num_inference_steps=30,
        guided=True,
        spatial_upsampler=True,
        distilled_lora=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("official", "omni"), required=True)
    parser.add_argument("--mode", choices=(*CASES, "all"), default="all")
    parser.add_argument("--modality", choices=("t2v", "i2v", "all"), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("LTX25_MODEL_ROOT", "models/ltx-2.5-official")),
    )
    parser.add_argument(
        "--omni-model-root",
        type=Path,
        default=Path(os.environ.get("LTX25_OMNI_MODEL_ROOT", "models/ltx-2.5-diffusers")),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(os.environ.get("LTX25_OFFICIAL_ROOT", "/tmp/ltx2-official-pr272")),
    )
    parser.add_argument(
        "--vllm-omni-root",
        type=Path,
        default=Path(os.environ.get("VLLM_OMNI_ROOT", ".")),
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("VLLM_OMNI_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).parent / "inputs" / "quickstart-seed42-frame0.png",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "results-final"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def component_paths(root: Path, cases: Iterable[PipelineCase]) -> dict[str, Path]:
    paths = {
        "full_transformer": root
        / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        "distilled_transformer": root
        / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoder": root
        / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "video_vae": root / "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "audio_vae": root / "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "duration_head": root / "model_patches/ltx-2.5-duration-head-bf16.safetensors",
        "spatial_upsampler": root
        / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "distilled_lora": root
        / "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    }
    required = {"text_encoder", "video_vae", "audio_vae", "duration_head"}
    for case in cases:
        required.add(f"{case.checkpoint_type}_transformer")
        if case.spatial_upsampler:
            required.add("spatial_upsampler")
        if case.distilled_lora:
            required.add("distilled_lora")
    missing = [
        str(paths[name]) for name in sorted(required) if not paths[name].is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing official LTX-2.5 components:\n" + "\n".join(missing)
        )
    return paths


def official_command(
    args: argparse.Namespace,
    case: PipelineCase,
    modality: str,
    seed: int,
    output: Path,
    paths: dict[str, Path],
) -> list[str]:
    pythonpath = os.pathsep.join(
        str(args.official_root / relative)
        for relative in ("packages/ltx-core/src", "packages/ltx-pipelines/src")
    )
    module_args = ["-m", case.official_module]
    official_height = case.height
    official_width = case.width
    if case.model_class_name == "LTX2DistilledOneStagePipeline":
        official_height, official_width = case.height * 2, case.width * 2
    if case.model_class_name == "LTX2DistilledOneStagePipeline":
        module_args = [str(Path(__file__).with_name("official_distilled_stage1.py"))]

    command = [
        "uv",
        "run",
        "--no-project",
        "--with",
        f"openimageio=={OFFICIAL_OPENIMAGEIO}",
        "--python",
        args.python,
        "python",
        *module_args,
        "--transformer-path",
        str(paths[f"{case.checkpoint_type}_transformer"]),
        "--text-encoder-path",
        str(paths["text_encoder"]),
        "--video-vae-path",
        str(paths["video_vae"]),
        "--audio-vae-path",
        str(paths["audio_vae"]),
        "--duration-head-path",
        str(paths["duration_head"]),
        "--prompt",
        PROMPT,
        "--output-path",
        str(output),
        "--seed",
        str(seed),
        "--height",
        str(official_height),
        "--width",
        str(official_width),
        "--num-frames",
        str(NUM_FRAMES),
        "--frame-rate",
        str(FPS),
    ]
    if case.guided:
        command += [
            "--negative-prompt",
            NEGATIVE_PROMPT,
            "--num-inference-steps",
            str(case.num_inference_steps),
        ]
    if case.spatial_upsampler:
        command += ["--spatial-upsampler-path", str(paths["spatial_upsampler"])]
    if case.distilled_lora:
        command += ["--distilled-lora", str(paths["distilled_lora"]), "1.0"]
    if modality == "i2v":
        command += ["--image", str(args.image), "0", "1.0", "18"]
    os.environ["LTX25_OFFICIAL_PYTHONPATH"] = pythonpath
    return command


def omni_command(
    args: argparse.Namespace,
    case: PipelineCase,
    modality: str,
    seed: int,
    output: Path,
) -> list[str]:
    example = (
        args.vllm_omni_root
        / "examples/offline_inference"
        / (
            "image_to_video/image_to_video.py"
            if modality == "i2v"
            else "text_to_video/text_to_video.py"
        )
    )
    if not example.is_file():
        raise FileNotFoundError(example)
    command = [
        args.python,
        str(example),
        "--model",
        str(args.omni_model_root),
        "--model-class-name",
        case.model_class_name,
        "--prompt",
        PROMPT,
        "--height",
        str(case.height),
        "--width",
        str(case.width),
        "--num-frames",
        str(NUM_FRAMES),
        "--num-inference-steps",
        str(case.num_inference_steps),
        "--frame-rate",
        str(FPS),
        "--fps",
        str(FPS),
        "--seed",
        str(seed),
        "--audio-sample-rate",
        "48000",
        "--enforce-eager",
        "--output",
        str(output),
    ]
    if case.guided:
        command += ["--negative-prompt", NEGATIVE_PROMPT]
    if modality == "i2v":
        command += ["--image", str(args.image), "--extra-body", '{"image_crf":18}']
    return command


def ffprobe(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


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


def main() -> None:
    args = parse_args()
    args.omni_model_root = args.omni_model_root.resolve()
    args.model_root = args.model_root.resolve()
    args.official_root = args.official_root.resolve()
    args.vllm_omni_root = args.vllm_omni_root.resolve()
    args.image = args.image.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.modality in ("i2v", "all") and not args.image.is_file():
        raise FileNotFoundError(args.image)

    modes = CASES if args.mode == "all" else {args.mode: CASES[args.mode]}
    paths = (
        component_paths(args.model_root, modes.values()) if args.backend == "official" else {}
    )
    modalities = ("t2v", "i2v") if args.modality == "all" else (args.modality,)
    manifest_scope = args.backend if args.mode == "all" else f"{args.backend}-{args.mode}"
    manifest_path = args.output_dir / f"manifest-{manifest_scope}.json"
    manifest = {
        "backend": args.backend,
        "official_model_root": str(args.model_root),
        "omni_model_root": str(args.omni_model_root),
        "official_revision": git_revision(args.official_root),
        "omni_revision": git_revision(args.vllm_omni_root),
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "image": str(args.image),
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "diffusion_attention_backend": os.environ.get(
            "DIFFUSION_ATTENTION_BACKEND", "platform default"
        ),
        "runs": {},
    }
    if manifest_path.is_file():
        manifest["runs"].update(json.loads(manifest_path.read_text()).get("runs", {}))

    env = os.environ.copy()
    if args.backend == "official":
        env["PYTHONPATH"] = os.pathsep.join(
            str(args.official_root / relative)
            for relative in ("packages/ltx-core/src", "packages/ltx-pipelines/src")
        )
    else:
        env["PYTHONPATH"] = str(args.vllm_omni_root)

    for mode, case in modes.items():
        for modality in modalities:
            for seed in args.seeds:
                run_id = f"{mode}-{modality}-{args.backend}-seed-{seed}"
                output = args.output_dir / f"{run_id}.mp4"
                log = args.output_dir / f"{run_id}.log"
                command = (
                    official_command(args, case, modality, seed, output, paths)
                    if args.backend == "official"
                    else omni_command(args, case, modality, seed, output)
                )
                source_root = (
                    args.official_root
                    if args.backend == "official"
                    else args.vllm_omni_root
                )
                reference_kind = (
                    "official_distilled_stage_1_extraction"
                    if mode == "distilled_one_stage" and args.backend == "official"
                    else "official_public_pipeline"
                    if args.backend == "official"
                    else "vllm_omni_public_pipeline"
                )
                if output.is_file() and output.stat().st_size > 0 and not args.force:
                    prior = manifest["runs"].get(run_id, {})
                    manifest["runs"][run_id] = {
                        **prior,
                        "status": "passed",
                        "reused": True,
                        "output": output.name,
                        "probe": ffprobe(output),
                        "mode": mode,
                        "modality": modality,
                        "seed": seed,
                        "model_class_name": case.model_class_name,
                        "checkpoint_type": case.checkpoint_type,
                        "width": case.width,
                        "height": case.height,
                        "num_inference_steps": case.num_inference_steps,
                        "command": shlex.join(command),
                        "source_revision": git_revision(source_root),
                        "reference_kind": reference_kind,
                    }
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    continue
                started = time.perf_counter()
                print(f"[{run_id}] starting", flush=True)
                with log.open("w") as stream:
                    completed = subprocess.run(
                        command,
                        cwd=args.official_root
                        if args.backend == "official"
                        else args.vllm_omni_root,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                record = {
                    "status": "passed" if completed.returncode == 0 else "failed",
                    "returncode": completed.returncode,
                    "elapsed_seconds": time.perf_counter() - started,
                    "output": output.name,
                    "log": log.name,
                    "mode": mode,
                    "modality": modality,
                    "seed": seed,
                    "model_class_name": case.model_class_name,
                    "checkpoint_type": case.checkpoint_type,
                    "width": case.width,
                    "height": case.height,
                    "num_inference_steps": case.num_inference_steps,
                    "command": shlex.join(command),
                    "source_revision": git_revision(source_root),
                    "reference_kind": reference_kind,

                }
                if completed.returncode == 0:
                    record["probe"] = ffprobe(output)
                manifest["runs"][run_id] = record
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                if completed.returncode:
                    raise SystemExit(f"{run_id} failed; inspect {log}")
                print(
                    f"[{run_id}] complete in {record['elapsed_seconds']:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()

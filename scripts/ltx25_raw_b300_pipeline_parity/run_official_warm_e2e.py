#!/usr/bin/env python3
"""Benchmark reused Official LTX-2.5 pipeline objects with the gallery contract."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


PIPELINES = {
    "LTX2Pipeline": "full_one_stage",
    "LTX2TwoStagePipeline": "full_two_stage",
    "LTX2DistilledOneStagePipeline": "distilled_one_stage",
    "LTX2DistilledTwoStagePipeline": "distilled_two_stage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--connector-model", type=Path, required=True)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--pipeline", choices=PIPELINES, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def insert_official_paths(root: Path) -> None:
    for relative in ("packages/ltx-core/src", "packages/ltx-pipelines/src"):
        path = str((root / relative).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def configure_cudnn(pipeline: Any) -> None:
    from ltx_core.loader.attention_ops import set_attention_module_op
    from ltx_core.model.transformer.attention import PytorchAttention
    from torch.nn.attention import SDPBackend

    attention = PytorchAttention(priority=[SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH])
    module_op = set_attention_module_op(attention=attention, masked_attention=attention)
    transformer_owners = [
        owner
        for name in ("stage", "stage_1", "stage_2")
        if (owner := getattr(pipeline, name, None)) is not None
    ]
    owners = [
        *((owner, "_transformer_builder") for owner in transformer_owners),
        (pipeline.prompt_encoder, "_embeddings_processor_builder"),
    ]
    for owner, attribute in owners:
        builder = getattr(owner, attribute)
        setattr(
            owner,
            attribute,
            builder.with_module_ops((*builder.module_ops, module_op)),
        )


def use_connector_weights(prompt_encoder: Any, model: Path) -> None:
    from safetensors import safe_open

    shards = sorted((model / "connectors").glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No connector weights under {model / 'connectors'}")
    original_build = prompt_encoder._build_embeddings_processor

    def build_embeddings_processor():
        processor = original_build()
        parameters = dict(processor.named_parameters())
        expected = {
            name
            for name in parameters
            if name.startswith(("video_connector.", "audio_connector."))
        }
        loaded: set[str] = set()
        with torch.no_grad():
            for shard in shards:
                with safe_open(str(shard), framework="pt", device="cpu") as weights:
                    for source_name in weights.keys():
                        if not source_name.startswith(
                            ("video_connector.", "audio_connector.")
                        ):
                            continue
                        target_name = (
                            source_name.replace(
                                ".transformer_blocks.", ".transformer_1d_blocks."
                            )
                            .replace(".attn1.norm_q.", ".attn1.q_norm.")
                            .replace(".attn1.norm_k.", ".attn1.k_norm.")
                        )
                        parameter = parameters[target_name]
                        parameter.copy_(
                            weights.get_tensor(source_name).to(
                                parameter.device, parameter.dtype
                            )
                        )
                        loaded.add(target_name)
        if loaded != expected:
            raise ValueError(
                f"Connector mismatch: missing={sorted(expected - loaded)}, "
                f"unexpected={sorted(loaded - expected)}"
            )
        return processor

    prompt_encoder._build_embeddings_processor = build_embeddings_processor


def component_paths(root: Path) -> dict[str, Path]:
    return {
        "full": root / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        "distilled": root
        / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoder": root
        / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "video_vae": root / "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "audio_vae": root / "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "upsampler": root
        / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "lora": root / "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    }


def make_pipeline(class_name: str, paths: dict[str, Path]):
    from ltx_pipelines.utils.model_paths import ModelPaths

    variant = "distilled" if "Distilled" in class_name else "full"
    model_paths = ModelPaths.from_split(
        transformer_path=str(paths[variant]),
        text_encoder_path=str(paths["text_encoder"]),
        video_vae_path=str(paths["video_vae"]),
        audio_vae_path=str(paths["audio_vae"]),
    )
    if class_name == "LTX2Pipeline":
        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

        pipeline = TI2VidOneStagePipeline(model_paths=model_paths, loras=())
    elif class_name == "LTX2TwoStagePipeline":
        from ltx_core.loader import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
        )
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        pipeline = TI2VidTwoStagesPipeline(
            model_paths=model_paths,
            distilled_lora=[
                LoraPathStrengthAndSDOps(
                    path=str(paths["lora"]),
                    strength=1.0,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ],
            spatial_upsampler_path=str(paths["upsampler"]),
            loras=(),
        )
    elif class_name == "LTX2DistilledOneStagePipeline":
        from official_distilled_stage1 import OfficialDistilledStage1

        pipeline = OfficialDistilledStage1(model_paths=model_paths, loras=())
    else:
        from ltx_pipelines.distilled import DistilledPipeline

        pipeline = DistilledPipeline(
            model_paths=model_paths,
            spatial_upsampler_path=str(paths["upsampler"]),
            loras=(),
        )
    return pipeline


def load_request(path: Path, image: Path) -> dict[str, Any]:
    request = json.loads(path.read_text())
    request["image"] = str(image) if request.get("image") else None
    return request


def images_for(request: dict[str, Any]):
    from ltx_pipelines.utils.args import ImageConditioningInput

    if request["image"] is None:
        return []
    return [
        ImageConditioningInput(
            path=request["image"],
            frame_idx=0,
            strength=1.0,
            crf=request.get("image_crf", 18),
        )
    ]


def generate(pipeline: Any, class_name: str, request: dict[str, Any]):
    images = images_for(request)
    common = {
        "prompt": request["prompt"],
        "seed": request["seed"],
        "height": request["height"],
        "width": request["width"],
        "num_frames": request["num_frames"],
        "frame_rate": request["fps"],
        "images": images,
    }
    if class_name == "LTX2DistilledOneStagePipeline":
        common["height"] *= 2
        common["width"] *= 2
        return pipeline(**common)
    if class_name == "LTX2DistilledTwoStagePipeline":
        return pipeline(**common)

    from ltx_core.components.guiders import MultiModalGuiderParams

    common.update(
        {
            "negative_prompt": request["negative_prompt"],
            "num_inference_steps": request["num_inference_steps"],
            "video_guider_params": MultiModalGuiderParams(
                cfg_scale=request["video_cfg_scale"],
                stg_scale=request["video_stg_scale"],
                rescale_scale=request["video_rescale_scale"],
                modality_scale=request["video_modality_scale"],
                skip_step=0,
                stg_blocks=request["video_stg_blocks"],
            ),
            "audio_guider_params": MultiModalGuiderParams(
                cfg_scale=request["audio_cfg_scale"],
                stg_scale=request["audio_stg_scale"],
                rescale_scale=request["audio_rescale_scale"],
                modality_scale=request["audio_modality_scale"],
                skip_step=0,
                stg_blocks=request["audio_stg_blocks"],
            ),
            "max_batch_size": 4,
        }
    )
    if class_name == "LTX2Pipeline":
        common["sigmas"] = torch.tensor(request["sigmas"], dtype=torch.float32)
    else:
        common["stage_1_sigmas"] = torch.tensor(
            request["stage_1_sigmas"], dtype=torch.float32
        )
        common["stage_2_sigmas"] = torch.tensor(
            request["stage_2_sigmas"], dtype=torch.float32
        )
    return pipeline(**common)


def encode_result(
    result: tuple[Any, ...], request: dict[str, Any], output: Path
) -> None:
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    video, audio = result[:2]
    num_frames = result[2] if len(result) == 4 else request["num_frames"]
    tiling_config = result[-1]
    encode_video(
        video=video,
        fps=request["fps"],
        audio=audio,
        output_path=output,
        video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
        color_space=None,
    )


def timed_request(
    pipeline: Any,
    class_name: str,
    request: dict[str, Any],
    output: Path,
) -> float:
    torch.accelerator.synchronize()
    started = time.perf_counter()
    result = generate(pipeline, class_name, request)
    encode_result(result, request, output)
    torch.accelerator.synchronize()
    return time.perf_counter() - started


def summarize(samples: list[float]) -> dict[str, Any]:
    return {
        "timed_repeats": len(samples),
        "samples_seconds": samples,
        "mean_seconds": statistics.fmean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "stdev_seconds": statistics.pstdev(samples),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    for name in ("official_root", "model_root", "connector_model", "request_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.image = args.image.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    insert_official_paths(args.official_root)
    torch.backends.cuda.enable_cudnn_sdp(False)
    pipeline = make_pipeline(args.pipeline, component_paths(args.model_root))
    use_connector_weights(pipeline.prompt_encoder, args.connector_model)
    configure_cudnn(pipeline)

    label = PIPELINES[args.pipeline]
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pipeline": args.pipeline,
        "label": label,
        "attention_backend": "CUDNN_ATTENTION with MATH fallback",
        "timing_scope": "reused Official pipeline-object call through complete MP4 encoding",
        "warmup_requests_excluded_per_task": 1,
        "tasks": {},
    }
    for task in ("t2v", "i2v"):
        request = load_request(args.request_dir / f"{label}-{task}.json", args.image)
        warmup_path = args.output_dir / f"{label}-{task}-warmup.mp4"
        warmup_seconds = timed_request(pipeline, args.pipeline, request, warmup_path)
        samples: list[float] = []
        for repeat in range(args.repeats):
            output = args.output_dir / f"{label}-{task}-repeat-{repeat + 1}.mp4"
            samples.append(timed_request(pipeline, args.pipeline, request, output))
        result["tasks"][task] = {
            "warmup_seconds": warmup_seconds,
            **summarize(samples),
        }
        gc.collect()
    (args.output_dir / f"official-warm-{label}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

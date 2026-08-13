#!/usr/bin/env python3
"""Official LTX-2.5 DistilledPipeline Stage 1 extraction reference.

This intentionally mirrors the pinned official ``DistilledPipeline.__call__``
through the end of Stage 1, then decodes that low-resolution latent directly.
The CLI keeps the official two-stage height/width convention: a request for
1920x1088 emits the Stage 1 result at 960x544.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.distilled import DistilledPipeline, should_use_ancestral_sampler
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    add_generated_keyframes_arg,
    default_2_stage_distilled_arg_parser,
    resolve_cli_params,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    DurationPredictor,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    require_num_frames_source,
    resolve_num_frames,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    ensure_tiling_config,
    generated_keyframe_conditionings,
    get_device,
    has_generated_keyframes,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import (
    HDRColorSpace,
    encode_video,
    resolve_hdr_color_space,
    vae_dtype_for_hdr,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, AutoDuration, ModalitySpec, OffloadMode


class OfficialDistilledStage1(DistilledPipeline):
    """The official DistilledPipeline with only its Stage 1 components loaded."""

    def __init__(  # noqa: PLR0913
        self,
        model_paths: ModelPaths,
        loras: Sequence[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ) -> None:
        # This is DistilledPipeline.__init__ with only VideoUpsampler omitted. No
        # replacement model or scheduler is introduced by this reference runner.
        self.device = device or get_device()
        self.dtype = torch.bfloat16

        self.prompt_encoder = PromptEncoder(
            model_paths,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
            prompt_enhancer_gemma_root=prompt_enhancer_gemma_root,
        )
        self.image_conditioner = ImageConditioner(
            model_paths.video_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.video_decoder = VideoDecoder(
            model_paths.video_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
            diffvae_optimization=diffvae_optimization,
        )
        self.audio_decoder = AudioDecoder(
            model_paths.audio_vae(),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.duration_predictor = DurationPredictor.from_checkpoint(
            model_paths.duration_head_path,
            self.dtype,
            self.device,
        )
        self.use_ancestral_sampler = should_use_ancestral_sampler(model_paths.transformer())

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        vae_dtype: torch.dtype | None = None,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        color_space: HDRColorSpace | None = None,
        generated_keyframes: int | Sequence[int] = 0,
    ) -> tuple[Iterator[torch.Tensor], Audio, int, TilingConfig | None]:
        """Run the exact official DistilledPipeline Stage 1 and decode it."""
        require_num_frames_source(num_frames, self.duration_predictor)
        # For LTX-2.5, resolve_crf fills an omitted image CRF with the official 18.
        images = self.image_conditioner.resolve_crf(images)
        assert_resolution(height=height, width=width, is_two_stage=True)
        if has_generated_keyframes(generated_keyframes):
            self.stage.assert_generated_keyframes_supported()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        if vae_dtype is None:
            vae_dtype = dtype

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        num_frames = resolve_num_frames(
            num_frames,
            self.duration_predictor,
            video_encoding=video_context,
            audio_encoding=audio_context,
            frame_rate=frame_rate,
        )

        # Keep the same pre-Stage-1 tiling resolution as DistilledPipeline. The
        # ConvVAE decoder consumes the resulting chunk configuration below.
        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(
                batch=1,
                frames=num_frames,
                height=height,
                width=width,
                fps=frame_rate,
            ),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
        )

        # BEGIN exact extraction of official DistilledPipeline.__call__ Stage 1.
        # FP32 sigmas and the inherited _stage_1_sampler_kwargs are intentional:
        # LTX-2.5 selects eta=1/s_noise=1 ancestral Euler with seed + 10000 noise.
        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_1_w, stage_1_h = width // 2, height // 2
        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_h,
                width=stage_1_w,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
                color_space=color_space,
            )
        )
        stage_1_conditionings.extend(generated_keyframe_conditionings(generated_keyframes, num_frames))

        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_w,
            height=stage_1_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=audio_context),
            **self._stage_1_sampler_kwargs(seed),
        )
        # END exact Stage 1 extraction. Stage 2 upsampling/refinement is omitted.

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator, dtype=vae_dtype)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio, num_frames, tiling_config


def _stage_1_arg_parser():
    params = resolve_cli_params(distilled=True)
    parser = add_generated_keyframes_arg(
        default_2_stage_distilled_arg_parser(params=params, supports_auto_duration=True)
    )

    # Reuse the official parser, but do not require the sole component Stage 1
    # neither loads nor executes. The option remains accepted for CLI parity.
    spatial_upsampler_action = next(
        action for action in parser._actions if action.dest == "spatial_upsampler_path"
    )
    spatial_upsampler_action.required = False
    spatial_upsampler_action.help = (
        "Accepted for official two-stage CLI compatibility; unused by this Stage 1 extraction."
    )
    return parser


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _stage_1_arg_parser().parse_args()
    pipeline = OfficialDistilledStage1(
        model_paths=args.model_paths,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        diffvae_optimization=args.diffvae_optimization,
    )
    hdr = resolve_hdr_color_space(images=args.images, hdr=args.hdr)
    vae_dtype = vae_dtype_for_hdr(hdr, torch.bfloat16)
    video, audio, num_frames, tiling_config = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        vae_dtype=vae_dtype,
        color_space=hdr,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        tiling_config=AUTO_TILING,
        generated_keyframes=args.num_generated_keyframes,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
        color_space=hdr,
    )


if __name__ == "__main__":
    main()

# LTX-2.5 official prompts — B300 1080p/20s results

This directory contains vLLM-Omni outputs generated from three prompts copied
verbatim from the official `Lightricks/LTX-2` repository. Every prompt has
three deterministic seeds (`42`, `43`, and `44`).

## Generation profile

| Field | Value |
|---|---|
| Checkpoint | `Lightricks/LTX-2.5-Diffusers` |
| Pipeline | `LTX2DistilledPipeline` |
| GPU | NVIDIA B300 SXM6 |
| Attention | `CUDNN_ATTN` |
| Output | 1920×1088, 481 frames, 24 FPS |
| Duration | 20.0417 seconds |
| Denoising | official distilled 8-step stage + 3-step refinement tail |
| Audio | model-generated synchronized stereo audio |

The 1920×1088 result is produced by the official two-stage design: generation
at 960×544, 2× latent spatial upsampling, then the refinement tail.

## Prompt provenance

- `quickstart-dialogue`: [official LTX-2.5 quickstart][quickstart]
- `home-office` and `chef`: [official T2V validation prompts][t2v-prompts]
- Negative prompt: [official T2V validation negative prompt][t2v-prompts]

All source links are pinned to Lightricks commit
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`.

## Reproduce

Run one prompt on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/ltx25_official_b300_1080p20s/generate_official_results.py \
  --prompt-id home-office \
  --seeds 42 43 44 \
  --model /path/to/LTX-2.5-Diffusers \
  --vllm-omni-root /path/to/vllm-omni
```

Run the complete set by omitting `--prompt-id`. The script calls the public
`examples/offline_inference/text_to_video/text_to_video.py` entrypoint and
records output metadata in `results/manifest.json`.

[quickstart]: https://github.com/Lightricks/LTX-2/blob/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca/README.md#quick-start
[t2v-prompts]: https://github.com/Lightricks/LTX-2/blob/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca/packages/ltx-trainer/configs/t2v_lora.yaml#L205-L247

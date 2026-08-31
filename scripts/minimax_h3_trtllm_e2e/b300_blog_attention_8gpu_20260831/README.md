# MiniMax-H3 attention A/B for the production-serving blog

This bundle contains the final eight-B300 attention comparison used by the
MiniMax-H3 production-serving article. It retains the measured videos, raw
server timings, request records, thermal telemetry, attention configurations,
quality measurements, environment manifest, and exact source patch.

## Workload and placement

- MiniMax-H3 FL2VA, official [10-second starship prompt](../prompts/minimax_h3_official_starship.txt), seed 0
- 1344x768, 24 FPS, 243 decoded frames, 50 sigma points and 49 DiT forwards
- 8x NVIDIA B300 SXM6 AC
- DiT TP1, Ulysses8, Ring1, Fast Ulysses
- text-encoder TP8
- VAE patch parallel 8 with tiled decode
- BF16 model weights and `TRTLLM_ATTN`
- dense `minimax_h3.token_refiner` through an explicit `per_role` entry

Each server receives one excluded full-shape warmup followed by two measured
requests. Before every request, the client waits until all eight GPUs are at or
below the recorded temperature gate. The primary A/B timer is the server's
`diffusion_engine_exec_time_s`; client E2E includes output work and is retained
only as supplementary data.

The final Dense baseline was repeated with a 49 C start gate after a 50 C run
incremented GPU 3's software thermal-slowdown counter. The other modes passed
at 50 C. Every final row has zero active thermal-slowdown samples and zero
counter delta. The stricter Dense run produced the same deterministic MP4 hash
as the discarded run.

## Results

| Attention policy | Model execution runs | Mean | Speedup | Full-video LPIPS | RGB PSNR | Audio correlation | Video |
|---|---:|---:|---:|---:|---:|---:|---|
| Dense TRTLLM | 54.185 s, 54.306 s | 54.246 s | 1.000x | 0 | inf | 1.000 | [MP4](./media/trtllm_dense.mp4) |
| FP8 SAGE, Q block 1 / K block 4 | 46.647 s, 46.536 s | 46.592 s | 1.164x | 0.4093 | 13.86 dB | 0.956 | [MP4](./media/sage_fp8_k4.mp4) |
| Skip-Softmax, threshold 0.05 / gate 0.97 | 50.090 s, 49.968 s | 50.029 s | 1.084x | 0.0917 | 26.33 dB | 0.901 | [MP4](./media/skip_softmax_005_gate097.mp4) |
| FP8 SAGE + Skip-Softmax | 46.143 s, 46.003 s | 46.073 s | 1.177x | 0.4103 | 13.97 dB | 0.868 | [MP4](./media/sage_fp8_skip_005_gate097.mp4) |

SAGE quantizes Q and K as FP8 E4M3 with `q_block_size=1` and
`k_block_size=4`; V quantization follows the backend's automatic path. The
Skip-Softmax policy uses `threshold=0.05` and
`disabled_until_timestep=0.97`, which enables it for 35 of 49 DiT forwards.
The policies are lossy: the measurements quantify the sample-specific tradeoff
rather than establishing a general quality threshold.

LPIPS uses AlexNet over all 243 decoded RGB frames against Dense. PSNR is the
mean of the per-frame decoded-RGB PSNR values. Audio correlation and SNR compare
the complete decoded 32 kHz stereo AAC waveforms. See [results.json](./results.json)
and the individual files under [quality](./quality/).

## Reproduction

The exact runner is [run_b300_attention_serve_matrix_8gpu.sh](./reproduction/run_b300_attention_serve_matrix_8gpu.sh).
Its container name, model path, cache path, and artifact root are variables at
the top of the script. The final four policies can be launched serially with:

```bash
SESSION_NAME=b300_blog_attention_reproduction \
MODES='trtllm_dense sage_fp8_k4 skip_softmax_005_gate097 sage_fp8_skip_005_gate097' \
MEASURE=2 \
MAX_START_TEMPERATURE_C=49 \
INTER_REQUEST_DELAY_S=0 \
bash reproduction/run_b300_attention_serve_matrix_8gpu.sh
```

The run command starts a resident server per policy with these material flags:

```text
--num-gpus 8 --usp 8 --ring 1 --ulysses-a2a-permute
--text-encoder-tp-size 8
--vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling
--diffusion-attention-config <configs/POLICY.json>
```

The HTTP client submits the prompt with seed 0, 50 steps, 1344x768 output,
24 FPS, 10-second duration, video flow shift 12, and audio flow shift 3. The
exact client and quality scripts are retained under [reproduction](./reproduction/).

To recover the two model-execution samples from a raw server log:

```bash
gzip -cd raw/trtllm_dense_server.log.gz | grep diffusion_engine_exec_time_s
```

The first value is the excluded warmup; the following two are the measured
requests. Dense stores its warmup in a separate client record because the
temperature gate was tightened before the measured requests.

## Provenance

- Measured vLLM-Omni source: `99e94905b8e49236df42ae7f5e9fd5a2bb6e97aa`
- Upstream base: `b864374089749ad3b4e6d01cb5d4f627beeb6174`
- Fast-Ulysses row-staging change: upstream PR
  [#6814](https://github.com/vllm-project/vllm-omni/pull/6814)
- Exact source delta: [vllm_omni_source.patch](./provenance/vllm_omni_source.patch)
- Container image ID, Python packages, GPU inventory/topology, model index
  hashes, and media hashes: [provenance](./provenance/)

The second local patch listed in [source_manifest.json](./provenance/source_manifest.json)
only aligns CUDA headers for the build environment; it does not alter model
scheduling or the attention policies measured here.

## Raw records

- `raw/*_client_results.json`: request timing, temperatures, media validation,
  peak memory, and deterministic output hashes
- `raw/*_server.log.gz`: native server statistics, including the reported
  `diffusion_engine_exec_time_s` samples
- `raw/*thermal_audit.json` and `raw/*gpu_telemetry.csv.gz`: thermal acceptance
  decisions and source telemetry
- `configs/*.json`: exact Dense, SAGE, Skip-Softmax, and combined policies
- `media/*.mp4`: first measured deterministic output for each policy

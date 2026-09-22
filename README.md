# vLLM-Omni Model Rankings

A self-updating static site that ranks the models supported by
[vLLM-Omni](https://github.com/vllm-project/vllm-omni) by their HuggingFace
download counts.

**Live page:** `https://lishunyang12.github.io/vllm-omni-rankings/`

## Research reports

- **H3: from 15-second generation to a five-second interactive avatar (2026-09-22):**
  [Read the standalone English HTML report](https://lishunyang12.github.io/vllm-omni-rankings/videos/minimax-h3-avatar-optimization-20260922/).
  A systems paper with 15 embedded figures and 25 tables: FastH3-to-Ref2VA-Turbo migration,
  the complete live stack and interaction policy, reference/motion continuity, audio flow,
  latency breakdowns, eight historical Nsight reanalyses, and the retained optimization history.
  Includes Twitch and Grafana screenshots, offline evidence downloads, negative results,
  and explicit limits on long-run consistency and viewer-response latency.

- **H3 narrated cartoon: complete 15-second request Nsight trace (2026-09-17):**
  [native trace and capture notes](scripts/h3-cartoon-full15s-nsys-20260917/README.md).
  Eight ranks, all four DiT forwards, full video/audio VAE and MP4 completion; the profiled request took 28.399s.
- **MiniMax-H3 Ulysses communication overlap and blockwise FP8 on 8x SM120:**
  [four-way comparison](https://lishunyang12.github.io/vllm-omni-rankings/scripts/minimax_h3_ulysses_blockwise_ab/) ·
  [lossless BF16 gate overlap](https://lishunyang12.github.io/vllm-omni-rankings/scripts/minimax_h3_ulysses_blockwise_ab/lossless/) ·
  [measurements, videos, and native Nsight traces](scripts/minimax_h3_ulysses_blockwise_ab/README.md)
- **MiniMax-H3 moon teahouse, 15 seconds: BF16 VSA vs Sage PR4691 vs locally
  adapted Cake PR4951 on 8x SM120:**
  [three-video comparison](https://lishunyang12.github.io/vllm-omni-rankings/scripts/minimax_h3_sage_teahouse_three_way/) ·
  [measurements, videos, and native Nsight traces](scripts/minimax_h3_sage_teahouse_three_way/README.md)
- **MiniMax-H3 revised cumulative optimization: restored 27.4937s BF16 VSA baseline to a full-VAE 14.9692s profile:**
  [24-page revised PDF](scripts/minimax_h3_pro5000_cumulative_report/minimax_h3_653s_to_14s_cumulative_report.pdf) ·
  [web report](scripts/minimax_h3_pro5000_cumulative_report/index.html) ·
  [中文累计优化表](scripts/minimax_h3_pro5000_cumulative_report/update-20260914/adopted-cumulative.zh.md) ·
  [current measurements and quality](scripts/minimax_h3_pro5000_cumulative_report/update-20260914/data.json).
  The old E4M3-wire 14.561s recommendation is superseded; the original 653.838s report remains archived with its historical contract.
- **Native Fused SVDQuant for MiniMax-H3 on NVIDIA B300:**
  [HTML](https://lishunyang12.github.io/vllm-omni-rankings/scripts/minimax_h3_svdquant_b300_results/) ·
  [results](scripts/minimax_h3_svdquant_b300_results/results.json) ·
  [visual comparison](https://lishunyang12.github.io/vllm-omni-rankings/scripts/minimax_h3_svdquant_official_comparison.html)
- **Topology-Aware Distributed Layerwise Offloading for MiniMax-H3 on Eight
  NVIDIA B300 GPUs:** [HTML](https://lishunyang12.github.io/vllm-omni-rankings/scripts/minimax_h3_b300_dlo_industrial_report/) ·
  [PDF](scripts/minimax_h3_b300_dlo_industrial_report/minimax_h3_b300_dlo_study.pdf) ·
  [artifacts](scripts/minimax_h3_b300_dlo_industrial_report/)

## How it works

A GitHub Actions workflow runs every 6 hours (`.github/workflows/update-rankings.yml`):

1. Scrapes the authoritative model list from vLLM-Omni's
   [`docs/models/supported_models.md`](https://github.com/vllm-project/vllm-omni/blob/main/docs/models/supported_models.md).
2. Queries the public HuggingFace API for each repo's 30-day / all-time downloads and likes.
3. Regenerates `index.html` (sortable, searchable) and `data.json`.
4. Commits the result if anything changed.

New models added to vLLM-Omni appear automatically — the list is never edited by hand.

## Local run

```bash
python scripts/generate_rankings.py   # writes index.html + data.json to repo root
```

Stdlib only, no dependencies.

## Setup (one time)

1. **Settings → Actions → General → Workflow permissions → "Read and write"**
   (lets the Action commit the refreshed page).
2. **Settings → Pages → Source: "Deploy from a branch" → `main` / root**.
3. **Actions → "Update model rankings" → Run workflow** to populate immediately.

## Data notes

- `downloads` = last 30 days, `downloadsAllTime` = cumulative (HuggingFace definitions).
- Some entries are the upstream base weights an integration builds on, not omni-specific checkpoints.
- Gated repos (HTTP 401) appear with a `gated` badge and no counts; add an `HF_TOKEN`
  secret and authenticate the API call to fill them in.

## H3 full-request hardware metrics (2026-09-17)

[Eight-GPU hardware metrics, PCIe read/write, and full 15-second-request Nsight trace](scripts/h3-cartoon-full15s-gpumetrics-20260917/).

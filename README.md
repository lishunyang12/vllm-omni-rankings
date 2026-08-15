# vLLM-Omni Model Rankings

A self-updating static site that ranks the models supported by
[vLLM-Omni](https://github.com/vllm-project/vllm-omni) by their HuggingFace
download counts.

**Live page:** `https://lishunyang12.github.io/vllm-omni-rankings/`

## Research reports

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

# Topology-Aware MiniMax-H3 DLO on 8xB300

This directory contains the archival HTML and four-page academic research note
for the MiniMax-H3 distributed layerwise offload (DLO) study on one
8x NVIDIA B300 node.

## Report

- [`index.html`](index.html): accessible web edition
- [`minimax_h3_b300_dlo_study.pdf`](minimax_h3_b300_dlo_study.pdf): typeset paper
- [`report.tex`](report.tex): PDF source

## Reproducibility artifacts

- `industrial_results.csv`: all 15 formal cases
- `t2va_soak_n20.csv`: combined two-lifecycle T2VA results for the three
  recommended routes
- `wave_samples.csv`: 105 preserved wave-level measurements
- `environment.json.txt`: hardware, software, input hashes, and source-diff
  metadata
- `run_dp_sp_point.py`, `run_multimodal_dp_sp_point.py`: benchmark runners
- `generate_figures.py`: deterministic figure generator

Regenerate the figures and paper from this directory:

```bash
python generate_figures.py
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
mv report.pdf minimax_h3_b300_dlo_study.pdf
```

The paper reports requested 50-step generations. The MiniMax-H3 scheduler used
in this experiment executes 49 denoising updates for that request.

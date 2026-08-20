# MiniMax-H3 Ulysses SP=8 Profiling Slides

- `MiniMax-H3_Ulysses_U4_学术解析.pptx`: seven slides, 16:9, English only.
- `build_deck.py`: reproducible PowerPoint source.

The deck uses a plain white background and native PowerPoint elements only:
title placeholders, text boxes, tables, lines, rectangles, and arrows.

Slides:

1. Contents
2. Profile Baseline
3. Original Data Layout Flow
4. Bottleneck
5. Optimization 1 — Pack Q/K in RoPE
6. Optimization 2 — Pack V in QKV Projection
7. Optimization 3 — Remove Output Copies

Build:

```bash
python3 -m pip install -r requirements.txt
python3 build_deck.py
```

Validate:

```bash
python3 validate_deck.py MiniMax-H3_Ulysses_U4_学术解析.pptx
```

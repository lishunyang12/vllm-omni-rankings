"""DSpark audio head — real vLLM serving speedup from the config-loader fix.

Before/after the fix, measured end-to-end against a live vLLM server (Qwen3-Omni
Thinker + DSpark audio draft) on LibriSpeech ASR, greedy, identical output. Data
in dspark_serving_speedup_data.json. Emits PNG + PDF.

Root cause: vllm/transformers_utils/configs/speculators/algos.py hardcoded the
1+N fill-in block layout (dspark_bonus_anchor=True), ignoring the checkpoint's
sample_from_anchor. That shifts the draft block by one position, collapsing
acceptance. Fix: dspark_bonus_anchor = not sample_from_anchor.

Run: python scripts/plot_dspark_serving_speedup.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__) or "."
d = json.load(open(os.path.join(HERE, "dspark_serving_speedup_data.json")))
bk, fx = d["before"], d["after"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "axes.linewidth": 0.6, "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})
C_bad = "#c4c9d4"; C_good = "#3b6fb6"; INK = "#1a1a1a"; MUTED = "#555555"

metrics = [
    ("Accepted length\n$\\tau$ (/8)", bk["accept_len"], fx["accept_len"], "{:.2f}", False),
    ("Throughput\n(tok/s)", bk["out_tok_per_s"], fx["out_tok_per_s"], "{:.0f}", False),
    ("TPOT\n(ms/token)", bk["tpot_ms"], fx["tpot_ms"], "{:.1f}", True),
    ("End-to-end\nlatency (s, 12 utts)", bk["total_latency_s"], fx["total_latency_s"], "{:.1f}", True),
]

fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.9))
fig.suptitle("DSpark audio head — real vLLM serving speedup from the config-loader fix",
             fontsize=11.5, fontweight="bold", color=INK, y=1.06)
for ax, (name, vb, vf, fmt, lower_better) in zip(axes, metrics):
    bars = ax.bar([0, 1], [vb, vf], width=0.62, color=[C_bad, C_good],
                  edgecolor=INK, linewidth=0.5, zorder=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2.5, width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["before", "after"])
    ax.set_ylim(0, max(vb, vf) * 1.28)
    for b, v in zip(bars, [vb, vf]):
        ax.text(b.get_x() + b.get_width() / 2, v, " " + fmt.format(v), ha="center",
                va="bottom", fontsize=9, color=INK, fontweight="bold")
    ax.set_title(name, fontsize=9.5, color=INK, pad=4)
    ratio = (vb / vf) if lower_better else (vf / vb)
    ax.text(0.5, 0.93, f"{ratio:.1f}×", transform=ax.transAxes, ha="center",
            fontsize=13, color=C_good, fontweight="bold")

fig.text(0.5, -0.06, d["note"] + ".  " + d["fix"], ha="center", fontsize=7.6, color=MUTED)
fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(HERE, f"dspark_serving_speedup.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print("saved ->", p)

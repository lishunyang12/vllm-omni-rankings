"""Render the DSpark audio-head serving results as two booktabs-style tables (PNG+PDF).

Table 1: single-request performance (no draft vs DSpark).
Table 2: throughput/latency under concurrency.
Data measured end-to-end against a live vLLM server; see dspark_serving_speedup_data.json.

Run: python scripts/plot_dspark_serving_tables.py
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42,
})
INK = "#161514"; MUTED = "#5c5852"; ACCENT = "#3b6fb6"; BAND = "#eef3f9"

# ---- data (measured, greedy, 1xB300, LibriSpeech ASR, identical output) ----
T1_COLS = ["Method", "Accept. len\nτ (/8)", "Throughput\n(tok/s)", "TPOT\n(ms)", "Latency\n(s, 12 utt.)"]
T1_ROWS = [
    ["Autoregressive (no draft)", "1.00", "26.7", "37.5", "25.09"],
    ["+ DSpark draft (block 7)", "6.84", "136.1", "7.35", "4.92"],
    ["Speedup", "6.8×", "5.1×", "5.1×", "5.1×"],
]
T2_COLS = ["Concurrency", "Baseline\n(tok/s)", "DSpark\n(tok/s)", "Speedup", "Mean lat.\n(s)", "p99 lat.\n(s)", "τ (/8)"]
T2_ROWS = [
    ["1", "29.3", "145.4", "5.0×", "0.39", "0.58", "7.00"],
    ["4", "95.4", "494.6", "5.2×", "0.44", "0.68", "6.98"],
    ["8", "214.8", "1057.7", "4.9×", "0.39", "0.51", "6.98"],
    ["16", "388.7", "1203.0", "3.1×", "0.56", "0.92", "7.01"],
]


def draw_table(ax, cols, rows, title, sp_col, peak_row=None):
    ax.axis("off")
    nc = len(cols)
    xs = [0.0]
    # first column wider
    widths = [0.30] + [(0.70) / (nc - 1)] * (nc - 1)
    for w in widths:
        xs.append(xs[-1] + w)
    centers = [(xs[i] + xs[i + 1]) / 2 for i in range(nc)]
    left = [xs[i] + 0.012 for i in range(nc)]

    header_y = 0.72
    row_h = 0.125
    ax.text(0.0, 1.14, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=12.5, fontweight="bold", color=INK)

    # header text
    for j, c in enumerate(cols):
        x = left[0] if j == 0 else centers[j]
        ha = "left" if j == 0 else "center"
        ax.text(x, header_y, c, transform=ax.transAxes, ha=ha, va="bottom",
                fontsize=9.5, color=INK, linespacing=1.05)
    # top + mid rules
    ax.plot([0, 1], [header_y + 0.11, header_y + 0.11], color=INK, lw=1.6, transform=ax.transAxes)
    ax.plot([0, 1], [header_y - 0.045, header_y - 0.045], color=INK, lw=0.9, transform=ax.transAxes)

    y = header_y - 0.045
    for ri, r in enumerate(rows):
        y -= row_h
        yc = y + row_h / 2
        if peak_row is not None and ri == peak_row:
            ax.add_patch(plt.Rectangle((0, y), 1, row_h, transform=ax.transAxes,
                                       color=BAND, zorder=0))
        is_speed = (r[0].lower() == "speedup")
        for j, cell in enumerate(r):
            x = left[0] if j == 0 else centers[j]
            ha = "left" if j == 0 else "center"
            col = ACCENT if (j == sp_col or (is_speed and j > 0)) else INK
            fw = "bold" if (is_speed or j == sp_col) else "normal"
            ax.text(x, yc, cell, transform=ax.transAxes, ha=ha, va="center",
                    fontsize=10.5, color=col, fontweight=fw)
        if is_speed:
            ax.plot([0, 1], [y + row_h, y + row_h], color="#cfd6df", lw=0.6, transform=ax.transAxes)
    # bottom rule
    ax.plot([0, 1], [y, y], color=INK, lw=1.6, transform=ax.transAxes)


fig = plt.figure(figsize=(8.6, 7.0))
fig.suptitle("DSpark audio draft head — vLLM serving results (Qwen3-Omni ASR)",
             fontsize=13, fontweight="bold", color=INK, y=0.985)
ax1 = fig.add_axes([0.04, 0.62, 0.92, 0.24])
ax2 = fig.add_axes([0.04, 0.11, 0.92, 0.34])
draw_table(ax1, T1_COLS, T1_ROWS, "Table 1.  Single-request performance (concurrency 1, 12 utts)", sp_col=None)
draw_table(ax2, T2_COLS, T2_ROWS, "Table 2.  Throughput and latency under concurrency (32 reqs / level)",
           sp_col=3, peak_row=2)
fig.text(0.04, 0.045,
         "Qwen3-Omni-30B-A3B Thinker + DSpark audio draft (block 7) · LibriSpeech train-clean-100 · greedy · 1×B300 (TP=2) · vLLM 0.25.0.",
         ha="left", fontsize=8, color=MUTED)
fig.text(0.04, 0.022,
         "Output verified identical token-for-token between configs. Requires the speculators config-loader fix (algos.py: dspark_bonus_anchor = not sample_from_anchor).",
         ha="left", fontsize=8, color=MUTED)

HERE = os.path.dirname(__file__) or "."
for ext in ("png", "pdf"):
    p = os.path.join(HERE, f"dspark_serving_tables.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print("saved ->", p)

#!/usr/bin/env python3
"""Generate the archival figures for the MiniMax-H3 8xB300 DLO report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
ACCENT = "#76B900"
BLUE = "#1F5A94"
ORANGE = "#C66A1B"
GRAY = "#5E6875"


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.dpi": 140,
        }
    )


def pareto_figure(soak: list[dict[str, str]]) -> None:
    if len(soak) != 3:
        raise ValueError(f"expected 3 T2VA soak rows, found {len(soak)}")
    labels = ["DP1 x SP8\nAllGather", "DP4 x SP2\nAllGather", "DP8 x SP1\nRank-local"]
    x = [float(row["wave_p50_s"]) for row in soak]
    y = [float(row["sustained_videos_per_hour"]) for row in soak]
    energy = [float(row["energy_wh_per_video"]) for row in soak]

    fig, ax = plt.subplots(figsize=(6.7, 3.65))
    ax.plot(x, y, color=ACCENT, linewidth=2.2, zorder=1)
    points = ax.scatter(
        x,
        y,
        s=[72, 102, 72],
        c=[BLUE, ACCENT, BLUE],
        edgecolor="white",
        linewidth=0.9,
        zorder=2,
    )
    del points
    offsets = [(7, -27), (7, 8), (-88, 8)]
    for xi, yi, label, wh, offset in zip(x, y, labels, energy, offsets, strict=True):
        ax.annotate(
            f"{label}\n{wh:.1f} Wh/video",
            (xi, yi),
            xytext=offset,
            textcoords="offset points",
            color=GRAY,
            fontsize=8.4,
        )
    ax.set_xlabel("Wave latency P50 (s) -- lower is better")
    ax.set_ylabel("Sustained throughput (videos/hour) -- higher is better")
    ax.set_title("Latency--throughput Pareto frontier (T2VA, 50-step, n=20)")
    ax.set_xlim(20, 172)
    ax.set_ylim(90, 193)
    fig.tight_layout()
    save(fig, "pareto_frontier")


def collective_figure(formal: list[dict[str, str]]) -> None:
    t2va = {
        (row["topology"], row["mode"]): row
        for row in formal
        if row["task"] == "t2va"
    }
    topologies = ["dp1-sp8", "dp4-sp2", "dp8-sp1"]
    labels = ["DP1 x SP8", "DP4 x SP2", "DP8 x SP1"]
    modes = ["ranklocal", "allgather"]
    colors = [BLUE, ORANGE]
    metrics = [
        ("sustained_videos_per_hour", "Throughput (videos/hour)"),
        ("wave_p50_s", "Wave latency P50 (s)"),
        ("measured_peak_memory_gib", "Measured peak/GPU (GiB)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.25))
    width = 0.34
    positions = list(range(len(topologies)))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for offset_index, (mode, color) in enumerate(zip(modes, colors, strict=True)):
            values = [float(t2va[(topology, mode)][metric]) for topology in topologies]
            offset = (-0.5 if offset_index == 0 else 0.5) * width
            ax.bar(
                [position + offset for position in positions],
                values,
                width,
                label="Rank-local" if mode == "ranklocal" else "AllGather",
                color=color,
            )
        ax.set_title(title)
        ax.set_xticks(positions, labels, rotation=16, ha="right")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left")
    fig.suptitle("DLO execution mode is topology-dependent (paired n=5)", y=1.02)
    fig.tight_layout()
    save(fig, "allgather_ranklocal")


def multimodal_figure(formal: list[dict[str, str]]) -> None:
    task_names = [
        ("fl2va", "FL2VA first-frame I2VA"),
        ("ref2va", "Ref2VA image + audio"),
    ]
    order = {"dp1-sp8": 0, "dp4-sp2": 1, "dp8-sp1": 2}
    colors = [BLUE, ORANGE]
    markers = ["o", "s"]

    fig, ax = plt.subplots(figsize=(6.7, 3.65))
    for (task, label), color, marker in zip(task_names, colors, markers, strict=True):
        rows = sorted(
            (row for row in formal if row["task"] == task),
            key=lambda row: order[row["topology"]],
        )
        if len(rows) != 3:
            raise ValueError(f"expected 3 rows for {task}, found {len(rows)}")
        x = [float(row["wave_p50_s"]) for row in rows]
        y = [float(row["sustained_videos_per_hour"]) for row in rows]
        ax.plot(x, y, marker=marker, color=color, linewidth=2, label=label)
        for row, xi, yi in zip(rows, x, y, strict=True):
            ax.annotate(
                row["topology"].upper().replace("-", " x "),
                (xi, yi),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.8,
                color=GRAY,
            )
    ax.set_xlabel("Wave latency P50 (s)")
    ax.set_ylabel("Sustained throughput (videos/hour)")
    ax.set_title("Conditioning path changes scale, not the Pareto direction")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "multimodal_frontiers")


def main() -> None:
    configure()
    FIGURES.mkdir(exist_ok=True)
    formal = read_csv("industrial_results.csv")
    soak = read_csv("t2va_soak_n20.csv")
    pareto_figure(soak)
    collective_figure(formal)
    multimodal_figure(formal)
    print("generated 3 figures in PDF and PNG formats")


if __name__ == "__main__":
    main()

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


def architecture_figure() -> None:
    """Show the DP/SP process grid and the two DLO weight-flow modes."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, axes = plt.subplots(
        1, 3, figsize=(10.6, 3.9), gridspec_kw={"width_ratios": [1.25, 1, 1]}
    )
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    def box(ax, x, y, width, height, label, *, face="#FFFFFF", edge=GRAY, size=8.0):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.06,rounding_size=0.10",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=size,
        )

    def arrow(ax, start, end, *, color=GRAY, style="-|>", width=1.2):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle=style,
                mutation_scale=9,
                color=color,
                linewidth=width,
            )
        )

    topology = axes[0]
    topology.set_title("(a) DP4 x SP2 process grid", loc="left", fontweight="bold")
    topology.text(
        5.6, 9.05, "SP rank within replica", ha="center", fontsize=8.0, color=GRAY
    )
    topology.text(3.75, 8.55, "0", ha="center", fontsize=8.0, color=GRAY)
    topology.text(7.35, 8.55, "1", ha="center", fontsize=8.0, color=GRAY)
    for dp_rank, y in enumerate([6.9, 5.25, 3.6, 1.95]):
        topology.text(
            0.05,
            y + 0.55,
            f"Req. {chr(65 + dp_rank)}\nDP{dp_rank}",
            ha="left",
            va="center",
            fontsize=7.7,
        )
        for sp_rank, x in enumerate([2.7, 6.3]):
            box(
                topology,
                x,
                y,
                2.1,
                1.1,
                f"GPU {dp_rank * 2 + sp_rank}\nSP{sp_rank}",
                face=["#E8F0F8", "#FDF0E5"][sp_rank],
            )
        arrow(
            topology,
            (4.82, y + 0.55),
            (6.27, y + 0.55),
            color=BLUE,
            style="<->",
            width=1.35,
        )
    for x in [3.75, 7.35]:
        arrow(topology, (x, 7.0), (x, 2.0), color=ORANGE, style="<->", width=1.55)
    topology.text(
        5.55,
        0.92,
        "horizontal: SP activation exchange",
        ha="center",
        fontsize=7.5,
        color=BLUE,
    )
    topology.text(
        5.55,
        0.40,
        "vertical: DLO weight AllGather over DP",
        ha="center",
        fontsize=7.5,
        color=ORANGE,
    )

    allgather = axes[1]
    allgather.set_title("(b) DLO + AllGather", loc="left", fontweight="bold")
    allgather.text(
        5.0,
        9.18,
        "synchronized compatible request wave",
        ha="center",
        fontsize=7.8,
        color=ORANGE,
    )
    host_shards = [
        ("rank 0\nhost 1/G", 0.35),
        ("...", 2.70),
        ("rank G-1\nhost 1/G", 4.20),
    ]
    for label, x in host_shards:
        box(
            allgather, x, 7.18, 1.80, 1.02, label, face="#FDF0E5", edge=ORANGE, size=7.0
        )
    arrow(allgather, (6.10, 7.68), (7.55, 7.68), color=ORANGE)
    box(
        allgather,
        7.65,
        6.95,
        1.95,
        1.45,
        "full block\nN+1",
        face="#FDF0E5",
        edge=ORANGE,
    )
    allgather.text(7.0, 8.28, "H2D + AG", ha="center", fontsize=7.2, color=ORANGE)
    box(
        allgather,
        0.55,
        4.15,
        3.15,
        1.20,
        "slot A: block N\nCOMPUTE",
        face="#EAF4D7",
        edge=ACCENT,
    )
    box(
        allgather,
        5.15,
        4.15,
        3.35,
        1.20,
        "slot B: block N+1\nPREFETCH",
        face="#FDF0E5",
        edge=ORANGE,
    )
    arrow(allgather, (3.8, 4.75), (5.05, 4.75))
    arrow(allgather, (8.58, 6.88), (7.35, 5.42), color=ORANGE)
    allgather.text(
        4.95,
        3.47,
        "overlap on compute / copy / comm streams",
        ha="center",
        fontsize=7.5,
        color=GRAY,
    )
    arrow(allgather, (2.25, 2.65), (7.55, 2.65), color=ACCENT, width=2.0)
    allgather.text(2.25, 2.05, "compute N", ha="center", fontsize=7.2, color=ACCENT)
    arrow(allgather, (3.70, 1.45), (8.75, 1.45), color=ORANGE, width=2.0)
    allgather.text(6.15, 0.84, "H2D + AG N+1", ha="center", fontsize=7.2, color=ORANGE)
    allgather.text(
        5.0,
        0.18,
        "swap slots; repeat for every block and update",
        ha="center",
        fontsize=7.3,
    )

    local = axes[2]
    local.set_title("(c) Rank-local DLO", loc="left", fontweight="bold")
    local.text(
        5.0,
        9.18,
        "heterogeneous and partial waves allowed",
        ha="center",
        fontsize=7.8,
        color=BLUE,
    )
    box(
        local,
        0.55,
        7.08,
        3.20,
        1.35,
        "full rank-local\nhost block N+1",
        face="#E8F0F8",
        edge=BLUE,
    )
    arrow(local, (3.90, 7.75), (5.15, 7.75), color=BLUE)
    local.text(4.55, 8.22, "H2D", ha="center", fontsize=7.2, color=BLUE)
    box(
        local,
        5.30,
        7.08,
        3.65,
        1.35,
        "slot B: block N+1\n(no DLO collective)",
        face="#E8F0F8",
        edge=BLUE,
    )
    box(
        local,
        0.55,
        4.30,
        3.20,
        1.20,
        "slot A: block N\nCOMPUTE",
        face="#EAF4D7",
        edge=ACCENT,
    )
    arrow(local, (7.10, 7.0), (3.80, 5.55), color=BLUE)
    local.text(
        5.0,
        3.60,
        "each replica owns its transfer schedule",
        ha="center",
        fontsize=7.5,
        color=GRAY,
    )
    lanes = [
        (2.60, "Req. A:  step k / block N+1", ACCENT, 9.0),
        (1.55, "Req. B:  step j / block M", BLUE, 7.65),
        (0.50, "idle replica:  no duplicated work", GRAY, 5.8),
    ]
    for y, label, color, end in lanes:
        local.text(
            0.25, y + 0.22, label, ha="left", va="bottom", fontsize=7.1, color=color
        )
        arrow(local, (0.30, y), (end, y), color=color, width=1.7)

    fig.suptitle(
        "MiniMax-H3 distributed layerwise offload dynamics (resident_layers=0)",
        y=1.02,
        fontsize=11.2,
        fontweight="bold",
    )
    fig.tight_layout(w_pad=1.05)
    save(fig, "dlo_architecture")


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
    ax.set_title("Selected-route latency--throughput frontier (T2VA, 50-step, n=20)")
    ax.set_xlim(20, 172)
    ax.set_ylim(90, 193)
    fig.tight_layout()
    save(fig, "pareto_frontier")


def collective_figure(formal: list[dict[str, str]]) -> None:
    t2va = {
        (row["topology"], row["mode"]): row for row in formal if row["task"] == "t2va"
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
    architecture_figure()
    pareto_figure(soak)
    collective_figure(formal)
    multimodal_figure(formal)
    print("generated 4 figures in PDF and PNG formats")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the evidence-backed MiniMax-H3 653.838 s -> 14.561 s report.

The PDF is intentionally generated with ReportLab so the checked-in artifact can
be reproduced on a machine without a TeX installation. Video comparisons are
precomputed by compare_h3_decoder_mp4_quality.py; this program extracts only the
three representative frames used in the visual audit.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/minimax-h3-report-matplotlib")

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
WORKSPACE = Path("/lustre/raplab/client/sylarl/minimax-h3-native")
FIG = HERE / "figures"
SHOTS = HERE / "screenshots"
EVIDENCE = HERE / "evidence"
PDF = HERE / "minimax_h3_653s_to_14s_cumulative_report.pdf"
DATA_JSON = HERE / "report_data.json"
HTML = HERE / "index.html"

BLUE = "#2563eb"
NAVY = "#0f172a"
CYAN = "#0891b2"
GREEN = "#059669"
AMBER = "#d97706"
RED = "#dc2626"
SLATE = "#475569"
LIGHT = "#e2e8f0"


@dataclass(frozen=True)
class Stage:
    id: str
    name: str
    e2e: float
    phase: str
    status: str
    fidelity: str
    principle: str
    evidence: str
    note: str


STAGES = [
    Stage("S0", "Original H3 control", 653.838, "A", "baseline", "Original", "49 diffusion forwards with the original decoder and communication path.", "Formal E2E artifact; DiT 633.058 s.", "The denominator for every cumulative speedup claim."),
    Stage("S1", "AdaLN TP1 / SP8", 605.348, "A", "accepted", "Exact", "Replicate small AdaLN tensors and spend all eight ranks on sequence parallelism.", "DiT 584.707 s; -48.490 s E2E.", "Removes tensor-parallel collectives from a latency-dominated batch-one request."),
    Stage("S2", "Explicit RDMA Ulysses exchange", 592.079, "A", "accepted", "Exact", "Move the large sequence exchange through the topology-aware RDMA path.", "DiT 571.264 s; -13.269 s E2E.", "The gain is interconnect-path selection, not model approximation."),
    Stage("S3", "Direct Q/K layout", 590.562, "A", "accepted", "Exact", "Produce the attention layout directly and delete redundant transposes/copies.", "DiT 569.830 s; -1.517 s E2E.", "A small but repeatable layout win before distillation."),
    Stage("F0", "Dense FastH3, four forwards", 66.785, "B", "accepted", "Lossy", "Use the distilled four-step sigma schedule [0.999, 0.749, 0.5, 0.25, 0].", "DiT 46.988 s; -523.777 s E2E.", "This is the largest single change and the first explicit quality boundary."),
    Stage("P0", "FastH3 + VSA + FlashInfer", 43.420, "B", "accepted", "Lossy", "Apply visual sparse attention with exact FlashInfer sparse-kernel execution inside the distilled trajectory.", "DiT 22.704 s; -23.365 s E2E.", "VSA changes the attention graph; the kernel itself is exact for that sparse graph."),
    Stage("P1", "GPU FP32-to-uint8 video pack", 36.568, "C", "accepted", "Exact", "Quantize display pixels on GPU and copy uint8 instead of float frames.", "-6.852 s E2E.", "Semantics are unchanged at the MP4 contract boundary."),
    Stage("P2", "Chunked pinned D2H + background mux", 31.399, "C", "accepted", "Exact", "Overlap pinned host copies and CPU PyAV encode/mux with remaining device work.", "-5.169 s E2E.", "The POST endpoint still waits for a valid H.264/AAC MP4."),
    Stage("P3", "Persistent RDMA communication", 30.534, "C", "accepted", "Exact", "Reuse communication resources across layer iterations instead of reinitializing them.", "-0.865 s E2E.", "Reduces orchestration and allocator overhead."),
    Stage("P4a", "Fused VSA tile packing", 30.013, "C", "accepted", "Exact", "Fuse tile selection and packing while preserving selected values and order.", "Approximately -0.490 s in matched trials.", "A kernel-launch and memory-traffic optimization."),
    Stage("P4c", "Compact VSA untile", 29.880, "C", "accepted", "Exact", "Write only the compact valid output domain during sparse-attention reconstruction.", "-0.133 s E2E.", "Avoids materializing padding that no downstream operator consumes."),
    Stage("P4d", "Direct attention O path", 29.873, "C", "accepted", "Exact / noise-sized", "Route attention output directly into its consumer.", "-0.007 s E2E.", "Retained for structural simplicity; the measured delta is within noise."),
    Stage("P5", "Exact VAE/direct Q-to-K cleanup", 28.236, "C", "accepted", "Exact", "Consolidate exact decoder and Q-to-K dataflow cleanups.", "Q0 control: DiT 21.347 s.", "A mixed cleanup bundle; it is not used to assign per-kernel causality."),
    Stage("Q1", "Online FP8 MLP", 25.172, "D", "accepted", "Lossy", "Use online FP8 for MLP GEMMs while retaining wider precision where sensitivity is higher.", "DiT 18.522 s; -3.064 s E2E.", "First low-precision compute boundary."),
    Stage("Q2", "MLP + output FP8", 24.887, "D", "accepted", "Lossy", "Extend FP8 to output projections with calibrated scaling.", "DiT 18.124 s; -0.285 s E2E.", "A conservative expansion before QKV/wire bundling."),
    Stage("Q5", "QKV E4M3 wire + O bundle", 21.649, "D", "accepted", "Lossy", "Transmit QKV in E4M3 and bundle output communication to reduce bytes and launch count.", "DiT 14.994 s; -3.238 s E2E.", "The accepted precision/communication endpoint before decoder replacement."),
    Stage("D1", "TAEH3 decoder, FP32", 16.738, "E", "accepted", "Lossy", "Replace the full patch-parallel video VAE with the rank-local temporal autoencoder decoder.", "Decode 1.554 s; -4.911 s E2E.", "Removes decoder collectives and most video decode compute."),
    Stage("D2", "TAEH3 decoder, FP16", 16.289, "E", "accepted", "Lossy", "Run TAEH3 convolution weights and activations in FP16.", "Mean E2E 16.289 s; decode 1.050 s.", "Post-codec FP32-vs-FP16 quality is separately audited."),
    Stage("D3", "All-main FP8 coverage", 15.634, "E", "accepted", "Lossy", "Extend calibrated FP8 coverage across the remaining main DiT GEMM path.", "DiT 14.443 s; decode 1.003 s.", "Completes the selected low-precision compute policy."),
    Stage("D4", "Persistent RDMA steady state", 15.165, "E", "accepted", "Exact", "Keep the optimized communication path persistent through the formal repeated run.", "15.146/15.178/15.170 s; mean 15.165 s.", "First result close to the 15-second target."),
    Stage("D5", "OMP=28 CPU orchestration", 14.671, "E", "accepted", "Exact", "Right-size host parallelism for decode/encode orchestration on this node.", "14.680/14.672/14.660 s; mean 14.671 s.", "First formal three-run mean below 15 seconds."),
    Stage("D6", "Cross-rank Audio VAE", 14.561, "E", "accepted", "Exact", "Decode video with TAEH3 on rank 0 while rank 1 independently runs Audio VAE.", "14.576/14.555/14.551 s; mean 14.561 s.", "Final accepted result; profiler disabled, valid MP4 returned."),
]

MAINLINE_IDS = [
    "S0", "S1", "S2", "S3", "F0", "P0", "P1", "P2", "P3", "P4a", "P4c",
    "P4d", "P5", "Q1", "Q2", "Q5", "D1", "D2", "D3", "D4", "D5", "D6",
]

SIDE_RESULTS = [
    ("Exact-H3 side control", "TP2/SP4 communication control", 646.158, "Not on accepted path"),
    ("Exact-H3 side control", "Q chunk = 2", 587.398, "Faster single side result; not retained as cumulative parent"),
    ("FastH3 reference", "Reference VSA", 47.527, "Diagnostic reference"),
    ("FastH3 reference", "FlashInfer-only contaminated E2E", 52.965, "DiT 27.108 s remains useful; E2E rejected"),
    ("FP8 branch", "MLP + QKV E4M3", 23.930, "Branch, not additive with Q2"),
    ("FP8 branch", "MLP + output + wire", 23.679, "Three-run mean branch"),
    ("FP8 branch", "+ O bundle", 23.168, "Three-run mean branch"),
    ("FP8 branch", "MLP + output + QKV compute", 23.260, "Alternate branch"),
    ("FP8 branch", "+ BF16 wire", 22.333, "Alternate branch"),
]

REJECTED = [
    ("Layerwise offload", "~50 s per diffusion step", "PCIe transfers dominated; categorically incompatible with the latency target."),
    ("LSA / SymmetricMemory", "Unstable or slower", "The topology/runtime combination did not produce a reliable improvement."),
    ("Hybrid attention path", "30.625 vs 30.075 s", "Added dispatch and layout cost."),
    ("Q chunk variant", "32.240 vs 31.965 s", "More chunk overhead than locality benefit."),
    ("Single-output path", "43.575 vs 43.420 s", "No measurable win."),
    ("NVENC", "49.817 vs PyAV 49.494 s", "Slower and produced a different encoded artifact."),
    ("VAE temporal prune", "30.518 vs 30.503 s", "No meaningful speedup."),
    ("Alternative GEMM backends", "within +/-1.5%", "No robust end-to-end advantage."),
    ("FC1 + SwiGLU CuTeDSL", "4.72% slower", "Fusion did not compensate for the generated kernel's efficiency loss."),
    ("Gate overlap", "21.714 vs 21.676 s", "Noise-sized regression."),
    ("Head-major VSA", "15.785 s", "Layout change regressed the final stack."),
    ("KV64", "15.320 s", "Reduced work did not offset scheduling/conversion overhead."),
    ("KV64 without LSE", "15.217 s mean", "Still slower than the exact-general selected path."),
    ("Same-GPU audio overlap", "15.233 s", "Video and audio competed for one GPU."),
    ("Adaptive partial-N32", "+0.645%", "Extra branch/scheduling cost."),
    ("KV64 to KV32 / KV16", "+8.3% / +41.7%", "Occupancy and useful-work balance collapsed."),
    ("Skip softmax", "Approximate and slower", "A lossy change with no latency benefit."),
    ("VAE batch cap 2", "Numerics changed", "Rejected because it crossed an unplanned quality boundary."),
]

NSYS = {
    "Original H3": {
        "scope_ms": 11618.329051,
        "active_ms": 11597.612898,
        "skew_ms": 1985.388816,
        "skew_pct": 17.0884,
        "categories": {"cuDNN attention": 64.5337, "GEMM": 19.8223, "RDMA barrier": 14.3633, "Other": 1.2807},
        "source": "results/vllm-omni-h3-sm120-sp8-step03-nsys-20260905T025828Z/h3-analysis.json",
    },
    "FastH3 / VSA P4d": {
        "scope_ms": 5225.109011,
        "active_ms": 5197.833258,
        "skew_ms": 568.697322,
        "skew_pct": 10.8839,
        "categories": {"GEMM": 54.849, "VSA": 27.137, "RDMA barrier": 9.023, "Layout/index/copy": 3.485, "Other": 5.506},
        "source": "results/vllm-omni-fasth3-vsa-p4d-sm120-sp8-step03-nsys-20260906T042700Z/h3-analysis.json",
    },
    "Quantized Q4/Q5": {
        "scope_ms": 3503.223878,
        "active_ms": 3485.2345265,
        "skew_ms": 423.19658,
        "skew_pct": 12.0802,
        "categories": {"GEMM": 39.863, "VSA": 38.473, "RDMA barrier": 9.759, "Layout/reduction/copy": 5.150, "Other": 6.755},
        "source": "results/vllm-omni-fasth3-vsa-qkv-e4m3-mlp-out-qkv-fp8-o-bundle-sm120-sp8-step3-nsys-20260907T000000Z/h3-analysis.json",
    },
}

NCU = {
    "historical_native_vsa": {
        "duration_ms": 93.57,
        "memory_throughput_pct": 34.31,
        "dram_throughput_pct": 0.81,
        "l1_pct": 39.25,
        "l2_pct": 5.18,
        "compute_pct": 13.31,
        "issue_slots_busy_pct": 4.69,
        "sm_busy_pct": 8.01,
        "l2_hit_pct": 97.34,
        "registers_per_thread": 162,
        "dynamic_shared_kb": 49.15,
        "theoretical_occupancy_pct": 20.83,
        "achieved_occupancy_pct": 18.71,
        "branch_efficiency_pct": 97.51,
        "excess_shared_wavefronts_pct": 79.0,
        "warning": "Historical native VSA experimental kernel; not the final exact-general FlashInfer kernel.",
    },
    "exact_matrix": {
        "scope": "single-GPU synthetic-QKV exact FlashInfer SM120 VSA fine-kernel shape matrix",
        "status": "passed",
        "cases": 3,
        "geomean_p50_ms": 11.724132,
        "max_cv": 0.0171457,
        "hashes": "all reference hashes matched",
    },
    "scheduler": {
        "historical_topk64_ms": 14.381,
        "original_flashinfer_ms": 10.342,
        "pr_head_ms": 9.870,
        "production_pr_head_ms": 29.504,
        "restored_head_local_ms": 22.819,
        "exact_general_ms": 22.032896,
        "paired_parent_ms": 23.525841,
        "paired_candidate_ms": 23.316159,
        "paired_gain_pct": 0.891,
    },
}

QUALITY_PAIRS = [
    ("01_original_h3_vs_dense_fasth3.json", "Lossy boundary A: Original H3 -> four-forward FastH3", "Original H3", "Dense FastH3",
     WORKSPACE / "results/vllm-omni-h3-pro5000-comm-15s-50sigma-20260904T185346Z/tp2-sp4-regular/out_t2va1.mp4",
     WORKSPACE / "results/vllm-omni-fasth3-dense-sp8-formal-20260905T1420Z/tp1-sp8-fasth3-dense-adaln-rdma-qk-direct/out_t2va1.mp4"),
    ("02_bf16_vs_fp8_qkv.json", "Lossy boundary B: BF16 compute -> MLP FP8 + QKV E4M3", "BF16 control", "FP8 / E4M3",
     WORKSPACE / "results/bf16-exact-vae-control-20260907T1/vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct/out_t2va1.mp4",
     WORKSPACE / "results/qkv-e4m3-mlp-fp8-formal-20260907T1/vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-qkv-e4m3-mlp-fp8/out_t2va1.mp4"),
    ("03_full_vae_vs_taeh3_fp32.json", "Lossy boundary C: full video VAE -> TAEH3 FP32", "Full VAE", "TAEH3 FP32",
     WORKSPACE / "results/qkv-compute-wire-e4m3-o-bundle-formal-repeat3-20260907T1/vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-qkv-e4m3-mlp-out-qkv-fp8/out_t2va1-repeat-3.mp4",
     WORKSPACE / "results/vllm-omni-fasth3-cumulative-taeh3-fp32-node-default-retry3-20260907/vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-qkv-e4m3-mlp-out-qkv-fp8-o-bundle-taeh3-fp32/out_t2va1.mp4"),
    ("04_taeh3_fp32_vs_fp16.json", "Lossy boundary D: TAEH3 FP32 -> TAEH3 FP16", "TAEH3 FP32", "TAEH3 FP16",
     WORKSPACE / "results/vllm-omni-fasth3-cumulative-taeh3-fp32-node-default-retry3-20260907/vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-qkv-e4m3-mlp-out-qkv-fp8-o-bundle-taeh3-fp32/out_t2va1.mp4",
     WORKSPACE / "results/vllm-omni-fasth3-cumulative-taeh3-fp16-node-default-repeat3-20260907/vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-qkv-e4m3-mlp-out-qkv-fp8-o-bundle-taeh3-fp16/out_t2va1-repeat-3.mp4"),
    ("05_taeh3_fp16_vs_all_main_fp8.json", "Lossy boundary E: partial FP8 -> all-main FP8", "Partial FP8", "All-main FP8",
     WORKSPACE / "results/vllm-omni-fasth3-cumulative-taeh3-fp16-node-default-repeat3-20260907/vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-qkv-e4m3-mlp-out-qkv-fp8-o-bundle-taeh3-fp16/out_t2va1-repeat-3.mp4",
     WORKSPACE / "results/vllm-omni-fasth3-cumulative-all-main-fp8-e4m3-o-bundle-taeh3-fp16-warm-20260907T041600Z/vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-qkv-e4m3-all-main-fp8-o-bundle-taeh3-fp16/out_t2va1.mp4"),
]

EXECUTED_PROMPT_ZH = "傍晚小厨房的真人实拍的手 与手绘发光2d动画 融合在一起的影像。夕阳余晖残留在窗边，生活感十足的小厨房里有旧木桌、洗到一半的马克杯、起雾的玻璃瓶、悬挂的抹布。画面带有智能手机单手拍摄的手抖、近距离对焦的犹豫、逆光曝光波动。要像在家中慌忙拍下某个不可思议事件的自然质感，不要广告影像的精心整理。声音只用厨房环境声与手绘生物柔和的电子音、小小的叫声。"
PROMPT_SHA = "613e0a1beccf83566e3531852c52c2e11daffa3b255222cd67726cdf73c1a563"
GUIDE_COMMIT = "d21241f0a4b3acbb34c97dae47fa417b7065e438"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_quality() -> list[dict[str, Any]]:
    rows = []
    for filename, title, left, right, ref_video, cand_video in QUALITY_PAIRS:
        path = EVIDENCE / "quality" / filename
        payload = json.loads(path.read_text())
        rgb = payload["metrics"]["rgb"]["summary"]
        yuv = payload["metrics"]["yuv"]["summary"]
        audio = payload["metrics"]["audio"]
        rows.append({
            "filename": filename,
            "title": title,
            "left": left,
            "right": right,
            "reference_video": str(ref_video),
            "candidate_video": str(cand_video),
            "rgb_psnr": rgb["global_psnr_from_mean_mse_db"],
            "rgb_ssim": rgb["per_frame_ssim"]["mean"],
            "yuv_psnr": yuv["global_psnr_from_mean_mse_db"],
            "yuv_ssim": yuv["per_frame_ssim"]["mean"],
            "audio_identical": audio["pcm_bitwise_identical"],
            "audio_snr_db": audio["overlap_snr_db"],
            "evidence_sha256": sha256_file(path),
        })
    return rows


def extract_frames(video: Path, indices: tuple[int, ...] = (48, 180, 312)) -> list[Image.Image]:
    result: dict[int, Image.Image] = {}
    with av.open(str(video)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                result[i] = frame.to_image().convert("RGB")
            if i > max(indices):
                break
    if set(result) != set(indices):
        raise RuntimeError(f"Could not extract requested frames from {video}")
    return [result[i] for i in indices]


def make_contact_sheet(row: dict[str, Any]) -> Path:
    target = SHOTS / row["filename"].replace(".json", ".jpg")
    if target.exists():
        return target
    refs = extract_frames(Path(row["reference_video"]))
    cands = extract_frames(Path(row["candidate_video"]))
    thumb_w, thumb_h = 480, 264
    top = 112
    canvas = Image.new("RGB", (thumb_w * 3, top + thumb_h * 2 + 58), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((18, 14), row["title"], fill=(15, 23, 42), font=font)
    metric = (
        f"RGB PSNR {row['rgb_psnr']:.2f} dB | RGB SSIM {row['rgb_ssim']:.4f} | "
        f"YUV PSNR {row['yuv_psnr']:.2f} dB | YUV SSIM {row['yuv_ssim']:.4f}"
    )
    draw.text((18, 39), metric, fill=(37, 99, 235), font=font)
    draw.text((18, 66), f"Top: {row['left']}    Bottom: {row['right']}    Frames: 2.0 s, 7.5 s, 13.0 s", fill=(71, 85, 105), font=font)
    for col, (a, b) in enumerate(zip(refs, cands, strict=True)):
        canvas.paste(a.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (col * thumb_w, top))
        canvas.paste(b.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (col * thumb_w, top + thumb_h + 8))
    draw.text((18, top + thumb_h * 2 + 24), "Representative frames only; metrics above aggregate all 362 post-codec frames.", fill=(71, 85, 105), font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=88, optimize=True)
    return target


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def build_figures(quality: list[dict[str, Any]]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ids = [s.id for s in STAGES]
    vals = [s.e2e for s in STAGES]
    cols = [NAVY if s.id == "S0" else RED if s.fidelity == "Lossy" else BLUE for s in STAGES]

    plt.figure(figsize=(15, 7))
    bars = plt.bar(ids, vals, color=cols)
    plt.ylabel("E2E latency (seconds, POST through valid MP4)")
    plt.title("Cumulative E2E latency gradient: 653.838 s to 14.561 s")
    plt.grid(axis="y", alpha=.25)
    for b, v in zip(bars, vals, strict=True):
        plt.text(b.get_x() + b.get_width()/2, v + 8, f"{v:.3f}", ha="center", va="bottom", rotation=90, fontsize=7)
    plt.ylim(0, 720)
    savefig(FIG / "e2e_gradient_bar.png")

    plt.figure(figsize=(14, 6))
    late = [s for s in STAGES if s.e2e < 70]
    plt.plot([s.id for s in late], [s.e2e for s in late], marker="o", color=BLUE, lw=2)
    plt.axhline(15, color=RED, ls="--", label="15 s target")
    for s in late:
        plt.annotate(f"{s.e2e:.3f}", (s.id, s.e2e), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    plt.ylabel("E2E seconds")
    plt.title("FastH3-era convergence (linear zoom)")
    plt.legend()
    plt.grid(alpha=.25)
    savefig(FIG / "e2e_zoom.png")

    deltas = [STAGES[i-1].e2e - STAGES[i].e2e for i in range(1, len(STAGES))]
    plt.figure(figsize=(14, 7))
    order = np.argsort(deltas)[::-1]
    plt.barh([f"{STAGES[i+1].id} {STAGES[i+1].name}" for i in order][::-1], [deltas[i] for i in order][::-1], color=GREEN)
    plt.xlabel("Marginal E2E reduction vs accepted parent (seconds)")
    plt.title("Waterfall contributions; bundles are not decomposed beyond measured evidence")
    plt.grid(axis="x", alpha=.25)
    savefig(FIG / "waterfall.png")

    phase_end = [("Original", 653.838), ("Exact H3", 590.562), ("FastH3 + VSA", 43.420), ("Exact systems", 28.236), ("FP8", 21.649), ("Final decode", 14.561)]
    plt.figure(figsize=(10, 6))
    plt.semilogy([x[0] for x in phase_end], [x[1] for x in phase_end], marker="o", lw=3, color=CYAN)
    for name, value in phase_end:
        plt.annotate(f"{value:.3f}s", (name, value), xytext=(0, 8), textcoords="offset points", ha="center")
    plt.ylabel("E2E seconds (log scale)")
    plt.title("Phase endpoints: multiplicative progress")
    plt.grid(alpha=.25, which="both")
    savefig(FIG / "phase_endpoints.png")

    names = list(NSYS)
    categories = sorted({c for v in NSYS.values() for c in v["categories"]})
    bottoms = np.zeros(len(names))
    plt.figure(figsize=(11, 6))
    cmap = plt.get_cmap("tab20")
    for i, cat in enumerate(categories):
        v = np.array([NSYS[n]["categories"].get(cat, 0) for n in names])
        plt.bar(names, v, bottom=bottoms, label=cat, color=cmap(i))
        bottoms += v
    plt.ylabel("Summed GPU kernel time share (%)")
    plt.title("Nsight Systems: bottleneck migration across the optimization arc")
    plt.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -0.10))
    savefig(FIG / "nsys_categories.png")

    _, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    ax1.bar(x - .18, [NSYS[n]["scope_ms"] for n in names], .36, label="Step scope", color=BLUE)
    ax1.bar(x + .18, [NSYS[n]["skew_ms"] for n in names], .36, label="Barrier arrival skew sum", color=AMBER)
    ax1.set_xticks(x, names)
    ax1.set_ylabel("Milliseconds")
    ax1.set_title("Nsight Systems: step scope and synchronization skew")
    ax1.legend()
    ax1.grid(axis="y", alpha=.25)
    savefig(FIG / "nsys_scope_skew.png")

    hist = NCU["historical_native_vsa"]
    metrics = ["Memory throughput", "Compute throughput", "Issue slots busy", "SM busy", "Theoretical occupancy", "Achieved occupancy"]
    values = [hist["memory_throughput_pct"], hist["compute_pct"], hist["issue_slots_busy_pct"], hist["sm_busy_pct"], hist["theoretical_occupancy_pct"], hist["achieved_occupancy_pct"]]
    plt.figure(figsize=(11, 6))
    plt.barh(metrics[::-1], values[::-1], color=[CYAN, RED, AMBER, BLUE, SLATE, GREEN][::-1])
    plt.xlabel("Percent")
    plt.title("NCU: historical native-VSA resource and utilization signature")
    plt.grid(axis="x", alpha=.25)
    savefig(FIG / "ncu_resource.png")

    sched = NCU["scheduler"]
    labels = ["Historical top-k64", "Original FlashInfer", "PR head", "Production PR head", "Restored head-local", "Exact-general"]
    values = [sched["historical_topk64_ms"], sched["original_flashinfer_ms"], sched["pr_head_ms"], sched["production_pr_head_ms"], sched["restored_head_local_ms"], sched["exact_general_ms"]]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color=[SLATE, CYAN, BLUE, RED, AMBER, GREEN])
    plt.ylabel("Kernel / microbenchmark latency (ms)")
    plt.title("VSA scheduler evolution (different shapes are deliberately separated in the report)")
    plt.xticks(rotation=18, ha="right")
    for b, v in zip(bars, values, strict=True):
        plt.text(b.get_x()+b.get_width()/2, v+.4, f"{v:.3f}", ha="center", fontsize=8)
    savefig(FIG / "ncu_scheduler.png")

    qlabels = [r["title"].split(":", 1)[0].replace("Lossy boundary ", "") for r in quality]
    arr = np.array([[r["rgb_ssim"], r["yuv_ssim"]] for r in quality])
    plt.figure(figsize=(8, 6))
    im = plt.imshow(arr, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    plt.yticks(range(len(qlabels)), qlabels)
    plt.xticks([0, 1], ["RGB SSIM", "YUV SSIM"])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, f"{arr[i,j]:.4f}", ha="center", va="center", color="white" if arr[i,j] < .72 else "black")
    plt.colorbar(im, label="Post-codec full-video mean SSIM")
    plt.title("Lossy-boundary diagnostic matrix")
    savefig(FIG / "quality_matrix.png")

    # Rank-level decoder overlap diagram.
    _, ax = plt.subplots(figsize=(12, 4.8))
    ax.broken_barh([(0, 13.68), (13.68, .84), (14.52, .04)], (22, 8), facecolors=[BLUE, GREEN, SLATE])
    ax.broken_barh([(0, 13.68), (13.68, .75), (14.43, .13)], (10, 8), facecolors=[BLUE, AMBER, SLATE])
    ax.text(6.8, 26, "Distributed DiT", color="white", ha="center", va="center", weight="bold")
    ax.text(14.10, 26, "TAEH3", color="white", ha="center", va="center", fontsize=8)
    ax.text(14.05, 14, "Audio VAE", color="white", ha="center", va="center", fontsize=8)
    ax.set_yticks([26, 14], ["rank 0", "rank 1"])
    ax.set_xlim(0, 15.1)
    ax.set_xlabel("Illustrative E2E timeline (seconds; not an Nsight capture)")
    ax.set_title("Why one rank can decode TAEH3 while another rank decodes audio")
    ax.grid(axis="x", alpha=.2)
    savefig(FIG / "cross_rank_timeline.png")

    # Architectural block diagrams.
    for final in (False, True):
        _, ax = plt.subplots(figsize=(12, 5.5))
        ax.axis("off")
        title = "Final 14.561 s path" if final else "Original 653.838 s path"
        ax.set_title(title, fontsize=18, weight="bold")
        boxes = (["Request", "49-step DiT\nTP2/SP4", "Full video VAE\npatch/tile PP8", "Audio VAE", "CPU encode", "MP4"] if not final else
                 ["Request", "4-step DiT\nSP8 + VSA + FP8", "rank 0\nTAEH3 FP16", "rank 1\nAudio VAE", "overlapped encode", "MP4"])
        xs = np.linspace(.04, .84, len(boxes))
        for i, (x, label) in enumerate(zip(xs, boxes, strict=True)):
            c = RED if (not final and i in (1,2)) else GREEN if (final and i in (1,2,3,4)) else BLUE
            ax.add_patch(plt.Rectangle((x, .38), .13, .24, color=c, alpha=.92, transform=ax.transAxes))
            ax.text(x+.065, .50, label, color="white", ha="center", va="center", transform=ax.transAxes, fontsize=9, weight="bold")
            if i < len(boxes)-1:
                ax.annotate("", xy=(xs[i+1], .50), xytext=(x+.13, .50), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 2, "color": NAVY})
        if final:
            ax.text(.52, .22, "rank-local decoders overlap after the distributed DiT barrier", ha="center", transform=ax.transAxes, color=SLATE)
        savefig(FIG / ("architecture_final.png" if final else "architecture_original.png"))


def paragraph_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleX", parent=base["Title"], fontName="Helvetica-Bold", fontSize=28, leading=33, textColor=colors.HexColor(NAVY), alignment=TA_LEFT, spaceAfter=12),
        "h1": ParagraphStyle("H1X", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=colors.HexColor(NAVY), spaceAfter=10),
        "h2": ParagraphStyle("H2X", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor(BLUE), spaceBefore=7, spaceAfter=5),
        "body": ParagraphStyle("BodyX", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=colors.HexColor(NAVY), spaceAfter=6),
        "small": ParagraphStyle("SmallX", parent=base["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9.5, textColor=colors.HexColor(SLATE), spaceAfter=3),
        "callout": ParagraphStyle("CalloutX", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=colors.HexColor(NAVY), backColor=colors.HexColor("#dbeafe"), borderColor=colors.HexColor(BLUE), borderWidth=.7, borderPadding=10, spaceAfter=10),
        "mono": ParagraphStyle("MonoX", parent=base["Code"], fontName="Courier", fontSize=7.2, leading=9.2, textColor=colors.HexColor(NAVY), backColor=colors.HexColor("#f8fafc"), borderPadding=6, spaceAfter=5),
        "caption": ParagraphStyle("CaptionX", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=7.2, leading=9, textColor=colors.HexColor(SLATE), alignment=TA_CENTER, spaceAfter=6),
    }


def table(data: list[list[Any]], widths: list[float] | None = None, header: bool = True, font_size: float = 7.4) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), font_size),
        ("LEADING", (0,0), (-1,-1), font_size + 2),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0,1 if header else 0), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0,0), (-1,0), colors.HexColor(NAVY)), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def img(path: Path, width: float = 7.1*inch) -> RLImage:
    im = Image.open(path)
    ratio = im.height / im.width
    return RLImage(str(path), width=width, height=width*ratio)


def make_doc(quality: list[dict[str, Any]], shots: list[Path]) -> int:
    styles = paragraph_styles()
    story: list[Any] = []
    page_title = {"value": ""}

    def p(txt: str, style: str = "body") -> None:
        story.append(Paragraph(txt, styles[style]))

    def page(title: str, subtitle: str | None = None) -> None:
        if story:
            story.append(PageBreak())
        page_title["value"] = title
        p(title, "h1")
        if subtitle:
            p(subtitle, "callout")

    def figure(path: Path, caption: str, width: float = 7.1*inch) -> None:
        story.append(img(path, width))
        p(caption, "caption")

    # Cover.
    story.append(Spacer(1, 34*mm))
    p("MiniMax-H3: From 653.838 Seconds to 14.561 Seconds", "title")
    p("A cumulative, evidence-backed latency optimization report for 15.083-second text-to-video-with-audio generation", "callout")
    p("NVIDIA RTX PRO 5000-class SM120 x8 | vLLM-Omni | FastH3 distillation | VSA | FP8 | TAEH3 | valid H.264/AAC MP4", "h2")
    story.append(Spacer(1, 12*mm))
    p("FINAL FORMAL RESULT", "h2")
    story.append(table([["Mean E2E", "Samples", "Speedup", "Reduction", "Target margin", "RTF"], ["14.561 s", "14.576 / 14.555 / 14.551 s", "44.903x", "97.773%", "0.439 s", "0.965"]], [24*mm, 48*mm, 22*mm, 24*mm, 25*mm, 18*mm]))
    story.append(Spacer(1, 12*mm))
    p(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Benchmark optimization stopped after D6; D7 remains unmeasured.", "small")
    p("Claim policy: measured, inferred, implemented-only, and rejected results are visually distinguished. No missing profile is reconstructed or simulated.", "small")

    page("Executive summary", "The final three-run mean is 14.561 s for a 15.083333 s deliverable: faster than real time and 44.903x faster than the 653.838 s original-H3 control.")
    p("Most of the speedup came from two deliberate approximation boundaries: four-forward FastH3 distillation and visual sparse attention. Exact systems work then removed layout copies, communication setup, device-to-host volume, and serialization. Calibrated FP8/E4M3 reduced DiT cost. TAEH3 removed the full patch-parallel VAE from the critical path, and cross-rank audio overlap supplied the final margin.")
    story.append(table([["Question", "Answer"], ["Did the benchmark satisfy the exact endpoint contract?", "Yes: POST through a validated 1280x704, 24 fps, 362-frame H.264 video with stereo 32 kHz AAC."], ["Was the final result a single best sample?", "No: 14.561 s is the arithmetic mean of 14.576, 14.555, and 14.551 s."], ["Are all changes lossless?", "No. Distillation, VSA, FP8/E4M3, and TAEH3 are explicitly marked as lossy/approximate."], ["Was a final decoder Nsight trace captured?", "No. NVTX instrumentation exists, but the D7 profiling attempt was stopped before model load after GPUs were occupied externally."], ["Can exact optimizations be added again?", "No. The final D6 already contains the retained exact-general VSA and systems changes."]], [48*mm, 125*mm]))

    page("Claims, labels, and evidence discipline")
    p("This report uses four evidence labels so that cumulative numbers cannot silently become stronger claims than the artifacts support.")
    story.append(table([["Label", "Meaning", "Allowed conclusion"], ["Measured", "A persisted benchmark, profiler, media, or metric artifact exists.", "Quote the value with its stated scope."], ["Exact", "Operation is intended to preserve the selected pipeline's numerical/media boundary.", "Attribute latency, not perceptual improvement."], ["Lossy", "Sampling graph, numeric precision, or decoder changed.", "Require screenshots and quantitative diagnostics."], ["Implemented-only", "Code/instrumentation exists without a valid formal timing.", "No latency claim."], ["Rejected", "Matched result was slower, unstable, contaminated, or crossed an unplanned quality boundary.", "Use as negative evidence, never add to the cumulative path."]], [28*mm, 66*mm, 79*mm]))
    p("The Nsight category tables sum GPU kernel durations across ranks and streams. They are composition views, not wall-clock decompositions. Barrier residency is synchronization/spin residency, not a direct measurement of bytes-on-wire.", "callout")

    page("Fixed benchmark contract")
    story.append(table([["Field", "Locked value"], ["Hardware", "8x NVIDIA RTX PRO 5000-class SM120 GPU; node-default/unlocked clocks"], ["Distributed shape", "DiT TP1/SP8; text encoder Qwen TP8; original baseline retained its documented control topology"], ["Request", "Text-to-video-with-audio, seed 1101, 1280x720 request"], ["Decoded output", "1280x704, 24 fps, 362 frames, 15.083333 s"], ["Audio", "Stereo, 32 kHz AAC"], ["FastH3 schedule", "4 forwards, sigma [0.999, 0.749, 0.5, 0.25, 0]"], ["E2E boundary", "HTTP POST start through a probe-valid H.264/AAC MP4"], ["Acceptance statistic", "Three repeated formal samples where available; arithmetic mean"], ["Target", "Strictly less than 15.000 seconds"]], [43*mm, 130*mm]))
    p("Output height 704 is the decoder-valid realization of the 720-pixel request. The media contract, not the request field alone, is what was verified.")

    page("Timing semantics: what 14.561 seconds includes")
    p("The timer includes request handling, distributed DiT work, video decoding, audio decoding, device-to-host transfers, H.264/AAC encoding and MP4 muxing. It stops only after the returned file passes the fixed media probe.")
    p("It does not include model cold start, weights download, report generation, or offline quality analysis. Profiler overhead is excluded from D6; profiling runs are used only for bottleneck localization.")
    figure(FIG / "architecture_final.png", "The final online request path and endpoint boundary.")

    page("Official MiniMax-H3 prompt-writing guide")
    p(f"The guide was read from MiniMax-AI/MiniMax-H3 at commit <font name='Courier'>{GUIDE_COMMIT}</font>. Its T2VA structure is: integrated multimodal description, overall soundscape, then non-diegetic music. It asks for concrete shot composition, subject, action, environment, camera motion type/amplitude/speed, timed later shots, and a total duration of 4-15 seconds.")
    p(f"Source: https://github.com/MiniMax-AI/MiniMax-H3/tree/{GUIDE_COMMIT}/skills/h3-prompt-writing", "mono")
    p("The calibration prompt predates that guide and was hash-locked. It was not rewritten for benchmark execution. The guide is therefore used as a documentation lens, not as evidence that a different prompt produced the reported timing.", "callout")

    page("Executed prompt and guide-conformant semantic rendering")
    p(f"Executed UTF-8 prompt SHA-256: <font name='Courier'>{PROMPT_SHA}</font>")
    p("Faithful English rendering of the locked prompt: A live-action handheld smartphone shot in a small kitchen at dusk blends with luminous hand-drawn 2D animation. Sunset light remains at the window. An old wooden table, a half-washed mug, a fogged glass bottle, and a hanging rag create an ordinary lived-in space. One-handed shake, hesitant close focus, and backlight exposure pumping make the event feel caught in a rush rather than polished advertising. Audio contains only kitchen ambience, soft electronic sounds from the illustrated creature, and its tiny calls.")
    p("Guide-field documentation only (not executed input):", "h2")
    story.append(table([["Guide field", "Documentation rendering"], ["integrated_multimodal_description", "Single continuous 15 s handheld close shot; live-action kitchen, dusk backlight, glowing 2D creature, hesitant rack focus and mild exposure breathing."], ["overall_soundscape", "Diegetic kitchen room tone, dish and cloth detail, soft electronic creature tones and tiny calls."], ["non_diegetic_music", "None."], ["camera", "Smartphone, one-handed micro-shake, close-focus hesitation, natural exposure fluctuation; no commercial polish."]], [48*mm, 125*mm]))

    page("Original architecture: where 653.838 seconds went")
    figure(FIG / "architecture_original.png", "The original-H3 control used 49 diffusion forwards and a collective-heavy full video VAE.")
    p("DiT alone consumed 633.058 s. The first accepted optimizations attacked distributed overhead without changing the original model. Nsight later showed cuDNN attention as 64.53% of summed GPU kernel time in the profiled step, followed by GEMM at 19.82% and RDMA-barrier residency at 14.36%.")

    page("Final architecture: why 14.561 seconds is feasible")
    figure(FIG / "architecture_final.png", "Four DiT forwards, sparse attention, low-precision GEMMs and rank-local parallel decoders form the final path.")
    p("The final design is not one magic kernel. It is a critical-path rewrite: remove 45 forwards, execute a smaller attention graph, reduce bytes and layouts, replace the full video VAE, and overlap independent terminal work on separate ranks.")

    page("Standalone E2E gradient bar chart", "Requested standalone visualization: every accepted cumulative stage is a bar, descending from 653.838 s to 14.561 s.")
    figure(FIG / "e2e_gradient_bar.png", "Red bars mark lossy boundary entries; blue bars are exact/system changes inside the selected pipeline.", 7.2*inch)

    page("FastH3-era convergence")
    figure(FIG / "e2e_zoom.png", "Linear zoom after distillation; the 15-second target is shown independently of the 653-second baseline.")

    page("Marginal contribution waterfall")
    figure(FIG / "waterfall.png", "Each bar is the observed difference from the accepted parent. Mixed bundles are not falsely decomposed.")

    page("Optimization phases on a logarithmic scale")
    figure(FIG / "phase_endpoints.png", "Phase endpoints expose the multiplicative nature of the speedup.")
    p("FastH3 distillation dominates absolute seconds. Once below 70 seconds, decoder and systems work become first-order. Near 15 seconds, even a 0.1-second overlap matters, so repeated samples and clean profiler-off measurements become essential.")

    page("Accepted cumulative path")
    rows = [["ID", "Optimization", "E2E s", "Delta s", "Speedup vs S0", "Fidelity"]]
    for i, s in enumerate(STAGES):
        d = "-" if i == 0 else f"{STAGES[i-1].e2e-s.e2e:.3f}"
        rows.append([s.id, s.name, f"{s.e2e:.3f}", d, f"{STAGES[0].e2e/s.e2e:.2f}x", s.fidelity])
    story.append(table(rows, [12*mm, 65*mm, 20*mm, 20*mm, 25*mm, 30*mm], font_size=6.3))
    p("Only this parent chain appears in the standalone gradient chart. Side branches are preserved separately and never stacked.", "callout")

    # One page per accepted transition.
    for i, s in enumerate(STAGES):
        if i == 0:
            continue
        parent = STAGES[i-1]
        delta = parent.e2e - s.e2e
        page(f"Optimization {i:02d} — {s.id}: {s.name}", f"{parent.e2e:.3f} s -> {s.e2e:.3f} s | measured reduction {delta:.3f} s | cumulative speedup {STAGES[0].e2e/s.e2e:.3f}x")
        story.append(table([["Dimension", "Evidence-backed statement"], ["Phase", s.phase], ["Status", s.status], ["Fidelity", s.fidelity], ["Principle", s.principle], ["Evidence", s.evidence], ["Interpretation", s.note]], [34*mm, 139*mm]))
        p("Causality boundary", "h2")
        if "bundle" in s.name.lower() or s.id in {"P5", "P0", "D3"}:
            p("This rung contains a measured bundle. The E2E delta is attributed to the bundle only; component-level claims come from separate microbenchmarks or profiler evidence and are not summed into a fictional decomposition.")
        elif abs(delta) < .05:
            p("The measured delta is noise-sized. The change is retained for simpler dataflow or because later stages depend on it, not because this single E2E pair proves a stable speedup.")
        else:
            p("The delta is the direct accepted-parent comparison. It is not combined with side-branch deltas.")
        p("Quality handling", "h2")
        if s.fidelity == "Lossy":
            p("This optimization crosses a declared approximation boundary. The dedicated quality section contains screenshots and full-video post-codec metrics for the nearest available matched artifact pair.")
        else:
            p("This is classified as an exact systems/dataflow optimization inside the already-selected model and precision policy. Exactness is checked with tensor/media hashes where the corresponding artifact exists.")

    page("Side branches: original-H3 and early FastH3")
    story.append(table([["Family", "Experiment", "E2E s", "Disposition"]] + [[a,b,f"{c:.3f}",d] for a,b,c,d in SIDE_RESULTS[:4]], [34*mm, 55*mm, 22*mm, 62*mm]))
    p("The 52.965-second FlashInfer-only E2E run was contaminated, so it is not an accepted endpoint. Its DiT measurement (27.108 s) remains a useful localization datum. The qchunk=2 exact-H3 side result was not inherited as the parent of later cumulative stages.")

    page("Side branches: FP8 and communication combinations")
    story.append(table([["Family", "Experiment", "E2E s", "Disposition"]] + [[a,b,f"{c:.3f}",d] for a,b,c,d in SIDE_RESULTS[4:]], [31*mm, 64*mm, 22*mm, 56*mm]))
    p("Q3a/Q4a and Q3b/Q4b/Q5 are alternative precision/communication branches. Treating their deltas as additive would overstate the speedup. Q5 is the only branch endpoint inherited by D1.", "callout")

    for page_index in range(0, len(REJECTED), 6):
        chunk = REJECTED[page_index:page_index+6]
        page(f"Rejected or non-retained experiments — set {page_index//6 + 1}")
        story.append(table([["Experiment", "Measured result", "Why it was not retained"]] + [list(x) for x in chunk], [48*mm, 37*mm, 88*mm], font_size=7.0))
        p("Negative results are part of the optimization map: they prevent rediscovery and show that smaller tensors, more overlap, more fusion, or a different encoder are not automatically faster at this shape.")

    page("D7: implemented, attempted, but deliberately unclaimed")
    p("AAC pre-encoding and decode-focused NVTX ranges were implemented and tested. A formal A/B and final decoder-focused Nsight capture were attempted. All GPUs were briefly idle, then an external SGLang workload occupied GPUs 4-7. The runner was stopped before model load, so there is no D7 E2E number and no final decode Nsight trace.", "callout")
    p("The final accepted endpoint therefore remains D6 at 14.561 s. This is conservative: implementation status is not converted into benchmark evidence.")

    page("Nsight Systems methodology")
    p("Nsight Systems answers timeline questions: which subsystem owns the step, whether kernels and transfers overlap, which rank reaches a barrier last, and whether decode/audio/encode are serialized. Three valid DiT-era traces are available: original H3, FastH3/VSA P4d, and the quantized Q4/Q5-era stack.")
    p("Category GPU time is summed over ranks and streams and may exceed wall time. Barrier arrival-skew sum is already contained inside the step wall and must not be subtracted again.", "callout")
    figure(FIG / "nsys_categories.png", "Composition changes from cuDNN-attention-dominated to a near-even GEMM/VSA split.")

    for name, trace in NSYS.items():
        page(f"Nsight Systems trace — {name}")
        story.append(table([["Metric", "Value"], ["Profiled step scope", f"{trace['scope_ms']:.3f} ms"], ["Median active rank span", f"{trace['active_ms']:.3f} ms"], ["Barrier arrival-skew sum", f"{trace['skew_ms']:.3f} ms ({trace['skew_pct']:.2f}% of scope)"], ["Source", trace["source"]]], [51*mm, 122*mm]))
        story.append(Spacer(1, 5*mm))
        story.append(table([["Kernel category", "Summed GPU-time share"]] + [[k, f"{v:.3f}%"] for k,v in trace["categories"].items()], [95*mm, 78*mm]))
        if name == "Original H3":
            p("Interpretation: dense attention dominates; rank arrival imbalance is material. This trace justifies prioritizing the attention graph and communication path.")
        elif name == "FastH3 / VSA P4d":
            p("Interpretation: four-step distillation and VSA move the bottleneck toward GEMM. Sparse attention remains substantial, while layout/index/copy costs become visible enough to optimize.")
        else:
            p("Interpretation: FP8 lowers GEMM share until VSA is nearly co-dominant. Further progress requires shape-specific sparse scheduling, not a generic GEMM-only campaign.")

    page("Nsight Systems: scope and synchronization skew")
    figure(FIG / "nsys_scope_skew.png", "The step shrinks sharply; the synchronization fraction remains material even as its absolute duration falls.")
    p("The original trace's dominant late rank was GPU 7, accounting for 38.35% of the recorded barrier-arrival skew. That diagnostic is only visible in a multi-rank timeline; per-kernel timing cannot identify it.")

    page("What genuinely requires Nsight Systems")
    story.append(table([["Question", "Why ordinary timers are insufficient"], ["Are video and audio decode concurrent?", "Only a cross-rank timeline proves overlap and exposes a hidden serialization gap."], ["Which rank gates a collective?", "A mean kernel time hides last-arriving-rank skew."], ["Does D2H overlap encode?", "Separate timers can double-count overlap; the timeline shows causality."], ["Is barrier residency network transfer?", "Timeline context separates waiting/spinning from data movement."], ["Does profiler overhead explain a regression?", "Compare profiler-on/off scopes and launch patterns."], ["Where is the remaining decoder critical path?", "Nested NVTX ranges around TAEH3, Audio VAE, copies, encode and mux are required."]], [58*mm, 115*mm]))
    p("The last item remains instrumented but unmeasured for D6. The report does not substitute the illustrative rank diagram for a trace.", "callout")

    page("Decoder NVTX plan for the next clean profiling window")
    story.append(table([["Range", "Question answered"], ["decode.video.taeh3", "Exact TAEH3 GPU interval on rank 0"], ["decode.audio.vae", "Audio VAE interval on rank 1"], ["decode.video.d2h", "Pinned-copy duration and overlap"], ["encode.video.pyav", "CPU video encode interval"], ["encode.audio.aac", "AAC encoding placement"], ["mux.mp4", "Final serialization and filesystem completion"], ["request.post", "True endpoint wall boundary"]], [58*mm, 115*mm]))
    p("Capture conditions: reserve all eight GPUs, warm once, record one formal request with CUDA/NVTX/OS runtime, export .sqlite, and run the nested-range analyzer. Do not accept a trace if another process enters the device set.")

    page("NCU methodology")
    p("Nsight Compute answers kernel-mechanism questions: occupancy limits, issue efficiency, cache behavior, shared-memory pressure, wavefronts, and instruction-level stalls. It does not establish request E2E or cross-rank overlap.")
    p("The historical native-VSA report and the final exact-general FlashInfer matrix have different scopes. They are reported separately and are never merged into one artificial speedup claim.", "callout")
    figure(FIG / "ncu_resource.png", "Historical native-VSA resource signature; useful for mechanism, not the final-kernel latency.")

    page("NCU: historical native-VSA bottleneck")
    h = NCU["historical_native_vsa"]
    story.append(table([["Metric", "Value"]] + [[k.replace("_", " "), f"{v}" if not isinstance(v,float) else f"{v:.3f}"] for k,v in h.items()], [80*mm, 93*mm], font_size=7.0))
    p("Registers (162/thread) and dynamic shared memory (49.15 KiB) each cap blocks per SM at two, limiting theoretical occupancy to 20.83%. Achieved occupancy is 18.71%. Low issue-slot and SM busy values, combined with 79% excessive shared wavefronts, point to scheduling/data-movement inefficiency rather than raw DRAM bandwidth.")

    page("NCU: exact FlashInfer shape matrix")
    m = NCU["exact_matrix"]
    story.append(table([["Field", "Value"]] + [[k.replace("_", " "), str(v)] for k,v in m.items()], [65*mm, 108*mm]))
    p("The three cases cover square 10-second geometry, the production-like 720p 15-second landscape geometry, and an SP4 head geometry. All reference hashes matched and the maximum coefficient of variation was 0.01715.")
    p("This matrix supports exactness and stability of the candidate fine-kernel path; it does not replace an E2E benchmark.", "callout")

    page("VSA scheduler evolution")
    figure(FIG / "ncu_scheduler.png", "Shape changes are shown, but only matched-shape pairs are used for direct percent claims.")
    p("For the production shape, restoring head-local scheduling improved 29.504 ms to 22.819 ms (~22.7%). Exact-general measured 22.032896 ms. In the final same-GPU paired microbenchmark, parent was 23.525841 ms and candidate 23.316159 ms (~0.89%), with identical output hashes.")

    page("What genuinely requires Nsight Compute")
    story.append(table([["Question", "Required NCU evidence"], ["Why does KV32 regress?", "Warp occupancy, branch divergence, sectors, waves and scheduler stalls."], ["Is a fusion register-bound?", "Registers/thread, occupancy and spill metrics."], ["Is shared-memory tiling pathological?", "Shared wavefronts, bank conflicts, dynamic shared bytes and block limits."], ["Does a new top-k shape underfill SMs?", "Grid size, active warps, eligible warps and achieved occupancy."], ["Is L2 reuse hiding DRAM cost?", "L2 hit rate, sectors and DRAM throughput together."], ["Did FP8 move the bottleneck?", "Tensor-pipe utilization, instruction mix and memory traffic on matched shapes."]], [58*mm, 115*mm]))
    p("NCU should be reserved for a small, shape-locked kernel matrix because replay overhead makes it unsuitable for full E2E timing.")

    page("Quality audit method")
    p("Each declared lossy boundary is represented by a real pair of persisted MP4s. Both files are validated as 362-frame, 1280x704, 24 fps H.264 with stereo 32 kHz AAC. FFmpeg deterministically converts both decoded streams to planar RGB and YUV444 before per-frame PSNR and SSIM. Audio is decoded to stereo interleaved float32 PCM.")
    p("PSNR/SSIM compare two stochastic-generation trajectories under one seed; they are diagnostic, not a human preference score and not an automated admission threshold. Low values after distillation or broad FP8 changes mean the trajectory diverged, not necessarily that the candidate is unusable.", "callout")
    figure(FIG / "quality_matrix.png", "All values are full-video post-codec means across 362 frames.")

    for row, shot in zip(quality, shots, strict=True):
        page(row["title"])
        figure(shot, "Top row is the reference; bottom row is the candidate. Columns are 2.0 s, 7.5 s and 13.0 s.", 7.25*inch)
        audio_text = "bitwise identical" if row["audio_identical"] else (f"not identical; overlap SNR {row['audio_snr_db']:.2f} dB" if row["audio_snr_db"] is not None else "not identical")
        story.append(table([["Metric", "Full-video result"], ["RGB global PSNR", f"{row['rgb_psnr']:.3f} dB"], ["RGB mean SSIM", f"{row['rgb_ssim']:.6f}"], ["YUV global PSNR", f"{row['yuv_psnr']:.3f} dB"], ["YUV mean SSIM", f"{row['yuv_ssim']:.6f}"], ["Decoded PCM", audio_text], ["Evidence SHA-256", row["evidence_sha256"]]], [50*mm, 123*mm], font_size=6.8))
        if row["filename"].startswith("01_"):
            p("Distillation changes the denoising trajectory from 49 forwards to four; large pixel differences are expected. The contact sheet is the primary human-auditable record for scene retention.")
        elif row["filename"].startswith("02_") or row["filename"].startswith("05_"):
            p("Online low precision perturbs the generated trajectory. This pair is not evidence of pixel identity; it quantifies divergence and exposes frames for inspection.")
        elif row["filename"].startswith("03_"):
            p("The video decoder alone changes; decoded audio PCM is bitwise identical, isolating the audiovisual boundary.")
        else:
            p("This matched decoder-precision pair is the closest quality comparison: YUV mean SSIM 0.983943 and identical decoded audio PCM.")

    page("Quality metric ledger")
    rows = [["Boundary", "RGB PSNR", "RGB SSIM", "YUV PSNR", "YUV SSIM", "Audio"]]
    for r in quality:
        audio = "identical" if r["audio_identical"] else f"SNR {r['audio_snr_db']:.2f} dB"
        rows.append([r["title"].split(":",1)[0].replace("Lossy boundary ", ""), f"{r['rgb_psnr']:.2f}", f"{r['rgb_ssim']:.4f}", f"{r['yuv_psnr']:.2f}", f"{r['yuv_ssim']:.4f}", audio])
    story.append(table(rows, [22*mm, 27*mm, 25*mm, 27*mm, 25*mm, 47*mm], font_size=6.8))
    p("Additional historical comparisons: matched QKV-E4M3 output measured approximately SSIM 0.5658 / PSNR 16.41 dB; an older MLP-FP8 vs BF16 comparison measured video SSIM ~0.5808 / PSNR ~17.53 dB and corrected full-scale audio PSNR ~31.62-32.02 dB. They are retained as historical diagnostics, not thresholds.")

    page("Exactness and artifact identity ledger")
    story.append(table([["Boundary", "Identity statement"], ["Exact-general VSA fine kernel", "All exact-matrix reference hashes matched; padding poison unchanged; return_lse=false buffer unchanged."], ["TAEH3 FP32 -> FP16", "Not identical; dedicated 362-frame quality audit supplied."], ["D3 -> D4 -> D5 -> D6 systems changes", "Selected pipeline semantics unchanged; final D6 media SHA-256 dbdbc19811ac656678648cc49555563f3fc31c2a572956dd24de48aa9f8e624f."], ["Audio for full VAE -> TAEH3 FP32", "Decoded stereo-f32 PCM bitwise identical."], ["Audio for TAEH3 FP32 -> FP16", "Decoded stereo-f32 PCM bitwise identical."], ["Prompt", f"Locked UTF-8 SHA-256 {PROMPT_SHA}."]], [57*mm, 116*mm]))
    p("Exactness is scoped to the chosen parent pipeline. It does not imply equality to original 49-forward H3 after earlier lossy boundaries.", "callout")

    page("Why TAEH3 needs only one rank")
    p("The original full video VAE is patch/tile parallel across eight ranks and contains collective communication. It needs every participating rank because each rank owns a spatial/temporal patch and collective exchanges reconstruct the full result.")
    p("TAEH3 is a different decoder topology. After the distributed DiT finishes, each rank has access to the required latent for rank-local temporal decoding. TAEH3 contains no patch-parallel collectives in this path, so rank 0 alone can decode the video. Rank 1 can simultaneously run Audio VAE on a different GPU. Remaining ranks do not need to join either decoder.", "callout")
    figure(FIG / "cross_rank_timeline.png", "Conceptual overlap, derived from stage measurements; explicitly not a substitute for the pending final Nsight capture.")

    page("Cross-rank decoder critical path")
    p("D1 reduced full-VAE decode from about 6.484 s to 1.554 s. D2 reduced TAEH3 decode to about 1.050 s. D4 measured about 0.847 s and D5 about 0.835 s after steady-state/system tuning. D6 then moved Audio VAE to rank 1 so its independent terminal work overlapped rank-0 TAEH3.")
    story.append(table([["Stage", "E2E", "DiT", "Video decode", "Critical-path change"], ["Q5 full VAE", "21.649", "14.994", "6.484", "Collective full VAE"], ["D1 TAEH3 FP32", "16.738", "15.010", "1.554", "Rank-local video decoder"], ["D2 TAEH3 FP16", "16.289", "15.068", "1.050", "Lower decoder precision"], ["D4 persistent", "15.165", "14.149", "0.847", "Steady-state resources"], ["D5 OMP=28", "14.671", "13.677", "0.835", "Host orchestration"], ["D6 cross-rank audio", "14.561", "not separately profiled", "not separately profiled", "Video/audio overlap"]], [37*mm, 24*mm, 32*mm, 33*mm, 47*mm], font_size=6.8))

    page("Remaining lossless optimization space")
    p("The remaining honest lossless headroom is probably measured in tens to low hundreds of milliseconds, not another order of magnitude. The final result is already below real time, and exact-general VSA plus the retained dataflow changes are already present.")
    story.append(table([["Candidate", "Expected scale", "Evidence needed", "Risk"], ["Confirm/optimize terminal mux tail", "10-100 ms", "Final decode Nsight NVTX capture", "Low"], ["AAC pre-encode overlap (D7)", "Unknown; likely small", "Clean three-run A/B + timeline", "Low semantics; measurement pending"], ["Reduce rank barrier skew", "Potentially 10-100+ ms", "Per-rank Nsight arrival analysis", "Medium topology sensitivity"], ["Shape-specific exact VSA scheduling", "Sub-percent to few percent kernel", "Matched NCU + paired microbench + E2E", "Medium maintenance"], ["Pinned-buffer lifetime/allocator cleanup", "Low tens of ms", "Allocator trace and repeated E2E", "Low"], ["CPU affinity/NUMA placement", "Low tens of ms", "OS runtime trace + repeated E2E", "Node-specific"], ["CUDA graph expansion", "Unknown", "Launch-gap timeline + exactness tests", "Medium memory/shape rigidity"]], [50*mm, 35*mm, 55*mm, 33*mm], font_size=6.7))
    p("Do not count current exact-general VSA, persistent RDMA, OMP=28, or cross-rank audio a second time: all are already in D6.", "callout")

    page("Optimization decision tree")
    story.append(table([["Observation", "Tool", "Action rule"], ["A subsystem consumes seconds of wall time", "Nsight Systems", "Change architecture/overlap before micro-tuning."], ["One kernel dominates and occupancy is low", "NCU", "Tune resource use and scheduling on the exact production shape."], ["A proposed approximation changes the trajectory", "Media audit", "Persist matched MP4s, screenshots and full-video metrics."], ["A delta is below run variance", "Repeated E2E", "Treat as noise or structural cleanup, not a speedup claim."], ["A run is contaminated", "Process/GPU provenance", "Reject E2E; salvage only independently valid scoped metrics."], ["An implementation lacks a run", "None", "Label implemented-only."]], [52*mm, 38*mm, 83*mm]))

    page("Validation and tests")
    story.append(table([["Suite", "Result"], ["Combined decoder/NVTX/analyzer tests", "126 passed"], ["FastVideo contract tests", "100 passed"], ["Formatting/static checks", "ruff, mypy and pre-commit passed in the recorded environment"], ["shellcheck", "Unavailable; no pass claim"], ["Media validation", "362 frames, 1280x704, 24 fps, H.264 + stereo 32 kHz AAC"], ["Final repetitions", "14.576 / 14.555 / 14.551 s"], ["Final media SHA-256", "dbdbc19811ac656678648cc49555563f3fc31c2a572956dd24de48aa9f8e624f"]], [59*mm, 114*mm]))
    p("Tests establish software and artifact contracts; they do not turn a lossy model change into a lossless one.")

    page("Reproduction outline")
    p("The full model workspace is intentionally not copied into this report repository. The report preserves command scope, artifact paths, hashes and compact profiler summaries. Reproduction requires the corresponding vLLM-Omni/FastVideo/FlashInfer worktrees and model weights.")
    p("1. Reserve all eight SM120 GPUs and verify no foreign compute process.\n2. Use node-default/unlocked clocks and the locked request/seed/prompt hash.\n3. Warm once; run three profiler-off POST requests.\n4. Probe every returned MP4.\n5. Capture one separate Nsight Systems request with NVTX ranges.\n6. Run NCU only on the production-shape sparse kernel matrix.\n7. Re-run post-codec quality comparison for every lossy boundary.", "mono")
    p("Report generator: python scripts/minimax_h3_pro5000_cumulative_report/generate_report.py", "mono")

    page("Evidence inventory")
    story.append(table([["Artifact class", "Checked-in evidence"], ["Cumulative measurements", "report_data.json and this PDF"], ["Lossy video comparisons", "evidence/quality/*.json (full 362-frame arrays and summaries)"], ["Screenshots", "screenshots/*.jpg at 2.0, 7.5 and 13.0 s"], ["Nsight Systems", "evidence/nsys/summary.json plus source artifact paths"], ["Nsight Compute", "evidence/ncu/summary.json plus exact-matrix scope"], ["Figures", "figures/*.png"], ["Landing page", "index.html"], ["Integrity", "SHA256SUMS.txt"]], [55*mm, 118*mm]))
    p("Raw .nsys-rep, .sqlite, .ncu-rep and source MP4s are not duplicated because they are large. Their derived summaries and quality comparison artifacts are sufficient to audit the claims made here, while source paths identify the originals.")

    page("Profiler source ledger")
    for name, trace in NSYS.items():
        p(name, "h2")
        p(trace["source"], "mono")
    p("NCU historical import: results/vsa-native-v1-real.ncu-rep", "mono")
    p("NCU exact matrix: results/profile/h3-vsa-exact-ncu-cold-candidate-forward-20260907/results.json", "mono")
    p("Analyzer: vllm-omni/tools/minimax_h3/analyze_h3_nsys_step.py", "mono")
    p("Decode profile wrapper: tools/minimax_h3/nsys_profile_h3_taeh3_cross_rank_decode.sh", "mono")

    page("Limitations")
    story.append(table([["Limitation", "Consequence"], ["Single node/GPU class", "Results do not automatically transfer to B300, H100, or another topology."], ["One locked prompt and seed", "Latency is tightly controlled; quality/generalization requires a prompt suite."], ["Profiler traces stop before final decoder stack", "D6 overlap is supported by E2E behavior and architecture, but not yet by a final Nsight timeline."], ["PSNR/SSIM only", "They expose pixel trajectory differences but do not replace human preference or modern perceptual metrics."], ["Mixed bundles exist", "Some E2E gains cannot be assigned to individual kernel edits."], ["Dynamic clocks", "Node-default policy reflects deployment behavior but increases variance relative to locked clocks."]], [61*mm, 112*mm]))

    page("Final conclusion", "The strict target is met: a 15.083-second audiovisual artifact is generated in a 14.561-second formal mean, with a 0.439-second margin.")
    p("The cumulative speedup is 44.903x and the latency reduction is 97.773%. The result is credible because the report keeps three categories separate: architectural approximations that require quality evidence, exact systems optimizations that preserve the selected pipeline, and unmeasured/rejected work that contributes no claimed seconds.")
    p("The next responsible action is not to stack speculative optimizations. It is to reserve a clean GPU window and capture the final decode-focused Nsight timeline, then test only the small lossless tails that the trace reveals. Until that evidence exists, D6 remains the final result.")
    p("END OF REPORT", "h2")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.line(18*mm, 13*mm, 192*mm, 13*mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor(SLATE))
        canvas.drawString(18*mm, 8*mm, "MiniMax-H3 cumulative optimization | evidence-backed | D6 final")
        canvas.drawRightString(192*mm, 8*mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=17*mm, bottomMargin=17*mm, title="MiniMax-H3: From 653.838 Seconds to 14.561 Seconds", author="Cumulative optimization study")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return len(PdfReader(str(PDF)).pages)


def write_data(quality: list[dict[str, Any]], pages: int) -> None:
    payload = {
        "schema": "minimax_h3_cumulative_report_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "headline": {"baseline_s": 653.838, "final_mean_s": 14.561, "samples_s": [14.576,14.555,14.551], "speedup": 653.838/14.561, "reduction_pct": (1-14.561/653.838)*100, "target_margin_s": 15-14.561, "media_duration_s": 15.083333, "rtf": 14.561/15.083333},
        "benchmark": {"gpus": 8, "gpu_class": "NVIDIA RTX PRO 5000-class SM120", "clocks": "node-default/unlocked", "dit": "TP1/SP8", "qwen": "TP8", "frames": 362, "width": 1280, "height": 704, "fps": 24, "audio_hz": 32000, "seed": 1101, "e2e_boundary": "POST through valid H.264/AAC MP4"},
        "prompt": {"executed_utf8": EXECUTED_PROMPT_ZH, "sha256": PROMPT_SHA, "official_guide_commit": GUIDE_COMMIT, "guide_url": f"https://github.com/MiniMax-AI/MiniMax-H3/tree/{GUIDE_COMMIT}/skills/h3-prompt-writing"},
        "stages": [asdict(x) for x in STAGES],
        "accepted_mainline_ids": MAINLINE_IDS,
        "side_results": SIDE_RESULTS,
        "rejected": REJECTED,
        "nsys": NSYS,
        "ncu": NCU,
        "quality": [{k:v for k,v in r.items() if k not in {"reference_video","candidate_video"}} for r in quality],
        "report_pages": pages,
        "disclosures": ["D7 AAC pre-encode was implemented but not formally measured.", "No final D6 decoder-focused Nsight capture exists.", "Nsight category GPU times are summed across ranks/streams and are not wall time.", "Quality metrics are diagnostics, not admission thresholds."],
    }
    DATA_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (EVIDENCE / "nsys" / "summary.json").write_text(json.dumps(NSYS, indent=2) + "\n")
    (EVIDENCE / "ncu" / "summary.json").write_text(json.dumps(NCU, indent=2) + "\n")


def write_html(pages: int, quality: list[dict[str, Any]]) -> None:
    rows = "\n".join(f"<tr><td>{r['title'].split(':',1)[0]}</td><td>{r['rgb_psnr']:.2f}</td><td>{r['rgb_ssim']:.4f}</td><td>{r['yuv_psnr']:.2f}</td><td>{r['yuv_ssim']:.4f}</td></tr>" for r in quality)
    HTML.write_text(f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>MiniMax-H3 653.838s to 14.561s</title>
<style>body{{font:16px/1.55 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#0f172a}}h1{{font-size:42px;line-height:1.05}}.hero{{background:#dbeafe;border-left:6px solid #2563eb;padding:22px}}.k{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.k div{{background:#f1f5f9;padding:15px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #cbd5e1;padding:8px;text-align:left}}a.button{{display:inline-block;background:#2563eb;color:white;padding:12px 18px;text-decoration:none;border-radius:6px}}code{{word-break:break-all}}@media(max-width:700px){{.k{{grid-template-columns:1fr 1fr}}}}</style></head>
<body><h1>MiniMax-H3: 653.838 s -> 14.561 s</h1><div class=\"hero\"><b>44.903x speedup, 97.773% latency reduction.</b> A 15.083-second H.264/AAC artifact completes in a 14.561-second three-run mean.</div>
<p><a class=\"button\" href=\"minimax_h3_653s_to_14s_cumulative_report.pdf\">Open the {pages}-page English PDF</a> &nbsp; <a href=\"report_data.json\">Machine-readable data</a> &nbsp; <a href=\"SHA256SUMS.txt\">Checksums</a></p>
<div class=\"k\"><div><b>Baseline</b><br>653.838 s</div><div><b>Final samples</b><br>14.576 / 14.555 / 14.551 s</div><div><b>Target margin</b><br>0.439 s</div><div><b>Media</b><br>362 frames, 1280x704, 24 fps</div></div>
<h2>Cumulative gradient</h2><img src=\"figures/e2e_gradient_bar.png\" alt=\"E2E latency gradient\">
<h2>Lossy-boundary audit</h2><p>Every declared lossy boundary has real video screenshots and full 362-frame post-codec metrics. These diagnostics measure trajectory divergence; they are not perceptual admission thresholds.</p>
<table><tr><th>Boundary</th><th>RGB PSNR</th><th>RGB SSIM</th><th>YUV PSNR</th><th>YUV SSIM</th></tr>{rows}</table>
<h2>Evidence disclosure</h2><p>D7 AAC pre-encoding was implemented but not formally measured. The final decoder-focused Nsight Systems capture was blocked by an external workload before model load; no synthetic timing is substituted. The report explains which conclusions require Nsight Systems versus Nsight Compute.</p>
<p>Official prompt-writing guide source: <a href=\"https://github.com/MiniMax-AI/MiniMax-H3/tree/{GUIDE_COMMIT}/skills/h3-prompt-writing\"><code>{GUIDE_COMMIT}</code></a>. The benchmark used a pre-existing hash-locked Chinese prompt; the guide-conformant English rendering is documentation only.</p>
</body></html>\n""")


def write_checksums() -> None:
    paths = [PDF, DATA_JSON, HTML] + sorted(FIG.glob("*.png")) + sorted(SHOTS.glob("*.jpg")) + sorted(EVIDENCE.rglob("*.json"))
    lines = [f"{sha256_file(p)}  {p.relative_to(HERE)}" for p in paths]
    (HERE / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    for d in (FIG, SHOTS, EVIDENCE / "nsys", EVIDENCE / "ncu"):
        d.mkdir(parents=True, exist_ok=True)
    quality = load_quality()
    shots = [make_contact_sheet(r) for r in quality]
    build_figures(quality)
    pages = make_doc(quality, shots)
    if pages < 50:
        raise RuntimeError(f"Report has only {pages} pages; expected at least 50")
    write_data(quality, pages)
    write_html(pages, quality)
    write_checksums()
    print(f"Built {PDF} ({pages} pages, {PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

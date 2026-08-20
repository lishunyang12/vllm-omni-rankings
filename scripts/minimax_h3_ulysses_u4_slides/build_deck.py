#!/usr/bin/env python3
"""Build a seven-slide MiniMax-H3 Ulysses SP=8 profiling deck."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "MiniMax-H3_Ulysses_U4_学术解析.pptx"
SW, SH = 13.333, 7.5

WHITE = "FFFFFF"
INK = "111827"
MUTED = "5F6B76"
GREEN = "047857"
GREEN_DARK = "064E3B"
GREEN_LIGHT = "ECFDF5"
GREEN_MID = "A7F3D0"
PANEL = "F8FAFC"
BORDER = "D1D5DB"
AMBER = "B45309"
AMBER_LIGHT = "FFFBEB"
RED = "B91C1C"
RED_LIGHT = "FEF2F2"
FONT = "Microsoft YaHei"
MONO = "Consolas"


def rgb(value):
    return RGBColor.from_string(value)


def blank(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(WHITE)
    return slide


def rect(slide, x, y, w, h, fill=WHITE, line=BORDER, width=0.8):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line)
    shape.line.width = Pt(width)
    return shape


def text(
    slide,
    x,
    y,
    w,
    h,
    value,
    size=16,
    color=INK,
    bold=False,
    font=FONT,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0.03,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    for idx, line in enumerate(str(value).split("\n")):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        p.line_spacing = 1.0
        for run in p.runs:
            run.font.name = font
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = rgb(color)
    return box


def rule(slide, x, y, w, color=BORDER, h=0.015):
    rect(slide, x, y, w, h, color, color, 0)


def title(slide, number, heading, subtitle=""):
    text(slide, 0.58, 0.29, 0.48, 0.30, f"{number:02d}", 11, GREEN, True, MONO)
    text(slide, 1.12, 0.22, 11.5, 0.45, heading, 25, INK, True)
    rule(slide, 0.58, 0.82, 12.12, GREEN, 0.025)
    if subtitle:
        text(slide, 0.60, 0.94, 12.0, 0.32, subtitle, 11, MUTED)


def footer(slide, number, source="原始 profile：agent/minimax-h3-pillar1@50cf90da7 · 2026-08-20"):
    rule(slide, 0.58, 7.04, 12.12, BORDER, 0.01)
    text(slide, 0.60, 7.10, 11.2, 0.18, source, 8, MUTED, font=MONO)
    text(slide, 12.05, 7.08, 0.62, 0.18, str(number), 9, GREEN, True, MONO, PP_ALIGN.RIGHT)


def pill(slide, x, y, w, value):
    rect(slide, x, y, w, 0.34, GREEN_LIGHT, GREEN_MID)
    text(slide, x + 0.03, y + 0.01, w - 0.06, 0.28, value, 10, GREEN_DARK, True, MONO, PP_ALIGN.CENTER)


def card(slide, x, y, w, h, heading, body, accent=GREEN, fill=WHITE, hs=14, bs=12):
    rect(slide, x, y, w, h, fill, BORDER)
    rect(slide, x, y, 0.055, h, accent, accent, 0)
    if h <= 0.90:
        text(slide, x + 0.20, y + 0.08, w - 0.34, 0.24, heading, hs, accent, True)
        text(slide, x + 0.20, y + 0.37, w - 0.34, h - 0.42, body, bs, INK)
    else:
        text(slide, x + 0.20, y + 0.15, w - 0.34, 0.33, heading, hs, accent, True)
        text(slide, x + 0.20, y + 0.57, w - 0.34, h - 0.69, body, bs, INK)


def metric(slide, x, y, w, value, label, detail=""):
    rect(slide, x, y, w, 1.08, PANEL, BORDER)
    text(slide, x + 0.16, y + 0.11, w - 0.32, 0.42, value, 22, GREEN_DARK, True, MONO)
    text(slide, x + 0.16, y + 0.56, w - 0.32, 0.23, label, 10.5, INK, True)
    if detail:
        text(slide, x + 0.16, y + 0.82, w - 0.32, 0.17, detail, 8, MUTED, font=MONO)


def code(slide, x, y, w, h, value, size=11):
    rect(slide, x, y, w, h, PANEL, BORDER)
    rect(slide, x, y, 0.055, h, GREEN, GREEN, 0)
    text(slide, x + 0.18, y + 0.14, w - 0.34, h - 0.26, value, size, INK, font=MONO)


def arrow(slide, x, y, w=0.36, h=0.42):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(GREEN)
    shape.line.color.rgb = rgb(GREEN)


def flow(slide, x, y, w, heading, shape_text, highlight=False):
    fill = GREEN_LIGHT if highlight else WHITE
    line = GREEN if highlight else BORDER
    rect(slide, x, y, w, 1.05, fill, line, 1.0)
    text(slide, x + 0.07, y + 0.13, w - 0.14, 0.28, heading, 11.5, GREEN_DARK if highlight else INK, True, align=PP_ALIGN.CENTER)
    text(slide, x + 0.07, y + 0.55, w - 0.14, 0.24, shape_text, 8.5, MUTED, font=MONO, align=PP_ALIGN.CENTER)


def table(slide, x, y, widths, rows, row_h=0.48):
    for ri, row in enumerate(rows):
        xx = x
        for ci, (value, width) in enumerate(zip(row, widths)):
            if ri == 0:
                fill, line, color, bold = GREEN_DARK, GREEN_DARK, WHITE, True
            else:
                fill, line, color, bold = (PANEL if ri % 2 else WHITE), BORDER, INK, False
            rect(slide, xx, y + ri * row_h, width, row_h, fill, line, 0.6)
            text(
                slide,
                xx + 0.06,
                y + ri * row_h + 0.08,
                width - 0.12,
                row_h - 0.12,
                value,
                9.2,
                color,
                bold,
                MONO if ci in (0, 1) else FONT,
            )
            xx += width


def bullets(slide, x, y, w, items, size=11, gap=0.36):
    for idx, item in enumerate(items):
        yy = y + idx * gap
        rect(slide, x, yy + 0.10, 0.07, 0.07, GREEN, GREEN, 0)
        text(slide, x + 0.18, yy, w - 0.18, gap, item, size, INK)


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    prs.core_properties.title = "MiniMax-H3 Ulysses SP=8：Nsight Systems 与 direct-copy 优化"
    prs.core_properties.subject = "SP=8 profile, layout copies, and packed Q/K optimization"
    prs.core_properties.author = "Codex · local source and Nsight Systems analysis"

    # 01 — title
    slide = blank(prs)
    pill(slide, 0.72, 0.62, 2.15, "MINIMAX-H3 · SP=8")
    text(slide, 0.72, 1.25, 11.8, 0.72, "Ulysses 热路径与 direct-copy 优化", 32, INK, True)
    text(slide, 0.74, 2.16, 11.4, 0.50, "从原始 Nsight Systems trace 到 RoPE 直写 NCCL packed Q/K", 18, GREEN_DARK, True)
    rule(slide, 0.74, 2.88, 11.84, GREEN, 0.03)
    metric(slide, 0.74, 3.28, 2.78, "8× B300", "硬件", "NV18 full fabric")
    metric(slide, 3.68, 3.28, 2.78, "TP1 · U8", "DiT 并行", "Ring1 · TRTLLM_ATTN")
    metric(slide, 6.62, 3.28, 2.78, "50 blocks", "重复热路径", "4 A2A / block")
    metric(slide, 9.56, 3.28, 3.02, "419.38 ms", "diffuse", "2-step profile")
    rect(slide, 0.74, 4.78, 11.84, 1.24, GREEN_LIGHT, GREEN_MID)
    text(slide, 1.04, 5.04, 2.18, 0.30, "核心结论", 15, GREEN_DARK, True)
    text(
        slide, 3.02, 4.95, 9.12, 0.70,
        "NCCL 前的 elementwise_kernel 是 transpose().contiguous() 产生的 64.53 MiB direct-copy。\n"
        "让 _rope_combine_kernel 直接写 Ulysses send layout，可消掉 Q/K 两次 pack。",
        15, INK, True,
    )
    text(slide, 0.76, 6.48, 11.7, 0.28, "分析版本：agent/minimax-h3-pillar1 @ 50cf90da7（不是 PR #5990 的单-pass kernel）", 10, MUTED, font=MONO)
    footer(slide, 1)

    # 02 — execution model
    slide = blank(prs)
    title(slide, 2, "SP=8 的数据所有权转换", "Attention 内部从 sequence-shard 变成 head-shard；Attention 外恢复 sequence-shard")
    nodes = [
        ("本地 Q/K/V", "[1, 4,720, 56, 128]", False, 1.90),
        ("pack send buffer", "[8, 4,720, 1, 7, 128]", True, 2.04),
        ("Q/K/V All-to-All", "3 × all_to_all_single", True, 1.94),
        ("TRT-LLM FMHA", "[1, 37,760, 7, 128]", False, 1.94),
        ("Output All-to-All", "恢复 [1, 4,720, 56, 128]", True, 2.34),
    ]
    x = 0.58
    for idx, (heading, shape_text, highlight, width) in enumerate(nodes):
        flow(slide, x, 1.70, width, heading, shape_text, highlight)
        x += width
        if idx < len(nodes) - 1:
            arrow(slide, x + 0.06, 2.00, 0.34, 0.42)
            x += 0.46
    rect(slide, 0.60, 3.20, 12.08, 1.18, PANEL, BORDER)
    text(slide, 0.88, 3.46, 1.35, 0.32, "元素守恒", 14, GREEN_DARK, True)
    text(slide, 2.35, 3.39, 4.44, 0.46, "(S/P)·H·D = S·(H/P)·D", 18, INK, True, MONO)
    text(slide, 7.12, 3.43, 5.10, 0.42, "Ulysses 改变 ownership，不减少 attention 数学量。", 13, MUTED)
    card(slide, 0.60, 4.72, 3.70, 1.62, "每层通信", "Q + K + V + Output\n= 4 次 All-to-All")
    card(slide, 4.48, 4.72, 3.70, 1.62, "50 层", "4 × 50 = 200\nNCCL SendRecv / GPU")
    card(slide, 8.36, 4.72, 4.32, 1.62, "关键执行事实", "1331 个 Ulysses 区间 kernel 全在 CUDA stream 7；本次没有计算—通信 overlap。", AMBER, AMBER_LIGHT)
    footer(slide, 2, "源码：minimax_h3_transformer.py → attention/layer.py → attention/parallel/ulysses.py")

    # 03 — trace alignment
    slide = blank(prs)
    title(slide, 3, "单个 DiT block：trace 与源码逐项对齐", "代表性 GPU0 中间层；时间用于说明关键路径结构")
    rows = [
        ["阶段", "代表耗时", "trace kernel", "源码归因"],
        ["RoPE products", "51.7 µs", "_rope_products_kernel", "fused_qk_norm_rope.py:55"],
        ["RoPE combine", "67.1 µs", "_rope_combine_kernel", "fused_qk_norm_rope.py:147"],
        ["Q pack", "91.3 µs", "elementwise/direct_copy", "comm.py:44"],
        ["Q A2A", "160.6 µs", "ncclDevKernel_SendRecv", "comm.py:51"],
        ["K pack + A2A", "90.2 + 152.9 µs", "direct_copy → SendRecv", "ulysses.py:333"],
        ["V pack + A2A", "90.6 + 150.7 µs", "direct_copy → SendRecv", "ulysses.py:334"],
        ["Attention", "3,441.3 µs", "fmha", "trtllm_attn.py"],
        ["O pack + A2A", "97.7 + 156.3 µs", "direct_copy → SendRecv", "comm.py:74–86"],
        ["O unpack", "86.5 µs", "elementwise/direct_copy", "comm.py:96"],
    ]
    table(slide, 0.60, 1.52, [2.05, 1.72, 3.34, 4.42], rows, 0.48)
    rect(slide, 0.60, 6.50, 12.12, 0.34, GREEN_LIGHT, GREEN_MID)
    text(slide, 0.82, 6.56, 11.72, 0.20, "ncclDevKernel_SendRecv 是 dist.all_to_all_single 的 NCCL 实现，不是模型显式调用的独立 Send/Recv 算子。", 10, GREEN_DARK, True)
    footer(slide, 3, "Trace：minimax_h3_sp8_2step_b300_20260820.nsys-rep · representative GPU0 block")

    # 04 — root cause
    slide = blank(prs)
    title(slide, 4, "direct-copy 的根因：NCCL send layout 需要物化", "transpose 只是 view；真正产生 elementwise_kernel 的是 contiguous()")
    code(
        slide, 0.60, 1.52, 5.90, 2.02,
        "# distributed/comm.py:44\n"
        "input_t = input.reshape(\n"
        "    B, S_local, P, H_local, D\n"
        ").transpose(0, 2).contiguous()\n\n"
        "# distributed/comm.py:51\n"
        "dist.all_to_all_single(output, input_t, group=group)",
        11.5,
    )
    card(slide, 6.76, 1.52, 5.92, 2.02, "物理布局变化", "[1, 4,720, 56, 128]\n→ view [8, 4,720, 1, 7, 128]\n→ contiguous：读写整个 64.53 MiB tensor", GREEN, GREEN_LIGHT, 15, 14)
    metric(slide, 0.60, 3.94, 2.86, "64.53 MiB", "每次 pack tensor", "BF16")
    metric(slide, 3.66, 3.94, 2.86, "5 / block", "Q、K、V、O pack、O unpack")
    metric(slide, 6.72, 3.94, 2.86, "250 / GPU", "50 层 direct-copy", "5 × 50")
    metric(slide, 9.78, 3.94, 2.90, "22.37 ms", "累计 critical copy", "约 diffuse 5.33%")
    rect(slide, 0.60, 5.38, 12.08, 1.14, AMBER_LIGHT, "FDE68A")
    text(slide, 0.88, 5.61, 2.30, 0.44, "为什么不能删 contiguous？", 14, AMBER, True)
    text(slide, 3.20, 5.55, 9.00, 0.58, "NCCL 按连续 send chunk 解释 buffer。直接传 transpose view 会按错误物理顺序通信；正确做法是让上游 producer 直接按 send layout 写。", 13, INK, True)
    footer(slide, 4, "Nsight demangled：at::native::direct_copy_kernel_cuda · gridX=66080, blockX=128")

    # 05 — proposed optimization
    slide = blank(prs)
    title(slide, 5, "第一阶段优化：RoPE 直接生成 packed Q/K", "只改变最终 store 地址；RoPE 数学顺序与 bitwise-exact 中间边界保持不变")
    card(slide, 0.60, 1.50, 5.66, 2.16, "当前路径", "_rope_combine_kernel\n  ↓ 写 [T_local, H, D]\nq_out / k_out\n  ↓ transpose().contiguous()\nNCCL send buffer", RED, RED_LIGHT, 15, 13)
    arrow(slide, 6.47, 2.32, 0.50, 0.54)
    card(slide, 7.18, 1.50, 5.50, 2.16, "优化路径", "_rope_combine_kernel_packed\n  ↓ 写 [P, T_local, B, H_local, D]\nprepacked Q / K\n  ↓ all_to_all_single\n不再执行 Q/K contiguous", GREEN, GREEN_LIGHT, 15, 13)
    code(
        slide, 0.60, 4.02, 6.20, 1.68,
        "heads_per_rank = H // P       # 7\n"
        "dst_rank = head // heads_per_rank\n"
        "rank_head = head % heads_per_rank\n"
        "packed[dst_rank, token, 0, rank_head, dim] = rope_result",
        11.5,
    )
    card(slide, 7.06, 4.02, 5.62, 1.68, "Ulysses 接口必须同步修改", "新增 prepacked helper：直接分配 output 并调用 all_to_all_single。不能继续走 SeqAllToAll4D，否则 comm.py:44 会再次 pack。", AMBER, AMBER_LIGHT, 14, 12)
    rect(slide, 0.60, 5.98, 12.08, 0.56, PANEL, BORDER)
    text(slide, 0.82, 6.10, 11.65, 0.26, "与 PR #5990 的关系：#5990 优化 RMSNorm+RoPE 计算；本方案优化 RoPE 输出→Ulysses 通信布局。两者互补。", 11, GREEN_DARK, True)
    footer(slide, 5, "修改点：fused_qk_norm_rope.py:147/292 · ulysses.py:332–334 · distributed/comm.py")

    # 06 — expected impact
    slide = blank(prs)
    title(slide, 6, "预期收益与边界", "以下是原始 trace 推导的理论上限；实现后必须重新 benchmark")
    metric(slide, 0.60, 1.52, 2.86, "250 → 150", "direct-copy / GPU", "每层消掉 Q、K")
    metric(slide, 3.66, 1.52, 2.86, "−100", "kernel launches / GPU", "2 × 50 blocks")
    metric(slide, 6.72, 1.52, 2.86, "≈ 8.90 ms", "Q+K pack critical time", "4.47 + 4.43 ms")
    metric(slide, 9.78, 1.52, 2.90, "≈ 2.1%", "diffuse 理论上限", "未验证")
    rows = [
        ["项目", "优化前", "优化后", "是否变化"],
        ["Q/K direct-copy", "2 / block", "0 / block", "减少"],
        ["V direct-copy", "1 / block", "1 / block", "不变"],
        ["Output pack/unpack", "2 / block", "2 / block", "不变"],
        ["NCCL SendRecv", "4 / block", "4 / block", "不变"],
        ["通信 payload", "原始字节数", "原始字节数", "不变"],
    ]
    table(slide, 0.60, 3.02, [3.00, 3.03, 3.03, 3.03], rows, 0.47)
    card(slide, 0.60, 6.02, 3.78, 0.72, "下一步：V", "融合到 QKV projection epilogue", GREEN, WHITE, 12, 10)
    card(slide, 4.57, 6.02, 3.78, 0.72, "下一步：Output", "FMHA 输出或 OutProj 消费特殊布局", GREEN, WHITE, 12, 10)
    card(slide, 8.54, 6.02, 4.14, 0.72, "不优先：只合并 NCCL launch", "不能消除 HBM pack 流量", AMBER, AMBER_LIGHT, 12, 10)
    footer(slide, 6, "原始 trace：Q pack 4.468 ms · K pack 4.429 ms · total direct-copy 22.371 ms")

    # 07 — implementation and validation
    slide = blank(prs)
    title(slide, 7, "实现与验证清单", "性能改动必须同时满足数值、分布式语义和 profiler 证据")
    rows = [
        ["代码位置", "需要修改", "验收点"],
        ["fused_qk_norm_rope.py:147", "packed store 地址与输出 shape", "unpack 后 torch.equal(reference)"],
        ["fused_qk_norm_rope.py:292", "分配 q_send / k_send；fake impl", "SP4、SP8、空序列与 fallback"],
        ["ulysses.py:332–334", "Q/K 用 prepacked helper；V 走旧路径", "输出等于 SeqAllToAll4D"],
        ["distributed/comm.py", "新增 all_to_all_4D_prepacked", "不执行 transpose().contiguous()"],
        ["tests/...fused_kernel.py", "H=56、D=128、P=4/8", "bitwise exact + e2e 固定 seed"],
    ]
    table(slide, 0.60, 1.48, [3.34, 4.12, 4.62], rows, 0.52)
    rect(slide, 0.60, 4.76, 12.08, 1.18, GREEN_LIGHT, GREEN_MID)
    text(slide, 0.86, 4.98, 2.54, 0.28, "优化后期望 trace", 14, GREEN_DARK, True)
    text(
        slide, 3.12, 4.91, 9.18, 0.56,
        "rope_combine_packed → NCCL(Q) → NCCL(K) → direct_copy(V) → NCCL(V)\n"
        "NCCL 数量保持 200；direct_copy 从 250 降到 150。",
        13, INK, True, MONO,
    )
    bullets(
        slide, 0.78, 6.10, 11.80,
        [
            "同一 model / prompt / seed / 分辨率 / frames / steps / backend / SP 配置做 before-after。",
            "先验证输出正确，再报告 diffuse、E2E、copy 数量和 exposed NCCL time。",
        ],
    )
    footer(slide, 7, "验收原则：same workload · no correctness regression · repeated measurement · trace attribution")

    prs.save(OUT_FILE)
    return OUT_FILE, len(prs.slides)


if __name__ == "__main__":
    output, count = build_deck()
    print(f"wrote {output}")
    print(f"slides {count}")

#!/usr/bin/env python3
"""Build the six-slide MiniMax-H3 SP=8 optimization deck."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "MiniMax-H3_Ulysses_U4_学术解析.pptx"

FONT = "Aptos"
MONO = "Consolas"
BLACK = "1F1F1F"
GRAY = "666666"
LIGHT_GRAY = "D9D9D9"
GREEN = "548235"
WHITE = "FFFFFF"


def rgb(value):
    return RGBColor.from_string(value)


def format_run(run, size, color=BLACK, bold=False, font=FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.color.rgb = rgb(color)
    run.font.bold = bold


def add_text(
    slide,
    x,
    y,
    w,
    h,
    value,
    size=17,
    color=BLACK,
    bold=False,
    font=FONT,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.04)
    frame.margin_right = Inches(0.04)
    frame.margin_top = Inches(0.02)
    frame.margin_bottom = Inches(0.02)
    frame.vertical_anchor = valign
    for index, line in enumerate(str(value).split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.alignment = align
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(0)
        paragraph.line_spacing = 1.0
        for run in paragraph.runs:
            format_run(run, size, color, bold, font)
    return shape


def add_line(slide, x1, y1, x2, y2, color=LIGHT_GRAY, width=1.0):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1),
        Inches(y1),
        Inches(x2),
        Inches(y2),
    )
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)
    return line


def add_title(slide, value):
    shape = slide.shapes.title
    shape.left = Inches(0.65)
    shape.top = Inches(0.28)
    shape.width = Inches(12.0)
    shape.height = Inches(0.55)
    shape.text = value
    for paragraph in shape.text_frame.paragraphs:
        for run in paragraph.runs:
            format_run(run, 26, BLACK, True)
    add_line(slide, 0.67, 0.95, 12.65, 0.95, GREEN, 1.4)


def add_footer(slide, page, source="Original profile: agent/minimax-h3-pillar1@50cf90da7"):
    add_line(slide, 0.67, 7.05, 12.65, 7.05, LIGHT_GRAY, 0.7)
    add_text(slide, 0.70, 7.10, 10.8, 0.18, source, 8, GRAY, font=MONO)
    add_text(slide, 12.0, 7.09, 0.60, 0.18, str(page), 8, GRAY, align=PP_ALIGN.RIGHT)


def add_bullets(slide, x, y, w, items, size=16, gap=0.62):
    for index, item in enumerate(items):
        yy = y + index * gap
        add_text(slide, x, yy, 0.25, gap, "•", size, GREEN, True)
        add_text(slide, x + 0.28, yy, w - 0.28, gap, item, size, BLACK)


def add_box(slide, x, y, w, h, value, size=15, font=FONT, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(w),
        Inches(h),
    )
    box.fill.solid()
    box.fill.fore_color.rgb = rgb(WHITE)
    box.line.color.rgb = rgb(LIGHT_GRAY)
    box.line.width = Pt(1)
    box.text_frame.clear()
    box.text_frame.word_wrap = True
    box.text_frame.margin_left = Inches(0.14)
    box.text_frame.margin_right = Inches(0.14)
    box.text_frame.margin_top = Inches(0.10)
    box.text_frame.margin_bottom = Inches(0.08)
    for index, line in enumerate(str(value).split("\n")):
        paragraph = box.text_frame.paragraphs[0] if index == 0 else box.text_frame.add_paragraph()
        paragraph.text = line
        paragraph.alignment = align
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(0)
        for run in paragraph.runs:
            format_run(run, size, BLACK, False, font)
    return box


def set_cell(cell, value, size=13, bold=False, color=BLACK, font=FONT):
    cell.text = str(value)
    cell.fill.solid()
    cell.fill.fore_color.rgb = rgb(WHITE)
    cell.margin_left = Inches(0.08)
    cell.margin_right = Inches(0.08)
    cell.margin_top = Inches(0.04)
    cell.margin_bottom = Inches(0.03)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for paragraph in cell.text_frame.paragraphs:
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(0)
        for run in paragraph.runs:
            format_run(run, size, color, bold, font)


def add_table(slide, x, y, w, h, rows, widths, size=13):
    shape = slide.shapes.add_table(
        len(rows),
        len(rows[0]),
        Inches(x),
        Inches(y),
        Inches(w),
        Inches(h),
    )
    table = shape.table
    for index, width in enumerate(widths):
        table.columns[index].width = Inches(width)
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            set_cell(
                table.cell(row_index, col_index),
                value,
                size,
                row_index == 0,
                GREEN if row_index == 0 else BLACK,
                MONO if col_index == 0 and row_index > 0 else FONT,
            )
    return shape


def new_slide(prs, title):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    add_title(slide, title)
    return slide


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "MiniMax-H3 Ulysses SP=8 Optimizations"
    prs.core_properties.subject = "Nsight Systems analysis"
    prs.core_properties.author = "Codex"

    # 1. Contents
    slide = new_slide(prs, "Contents")
    entries = [
        ("2", "Profile Baseline"),
        ("3", "Bottleneck"),
        ("4", "Optimization 1 — Pack Q/K in RoPE"),
        ("5", "Optimization 2 — Pack V in QKV Projection"),
        ("6", "Optimization 3 — Remove Output Copies"),
    ]
    for index, (page, label) in enumerate(entries):
        y = 1.48 + index * 0.95
        add_text(slide, 1.20, y, 0.55, 0.45, page, 21, GREEN, True, MONO)
        add_text(slide, 1.95, y, 9.90, 0.45, label, 21, BLACK)
        add_line(slide, 1.20, y + 0.58, 12.0, y + 0.58, LIGHT_GRAY, 0.6)
    add_footer(slide, 1)

    # 2. Profile Baseline
    slide = new_slide(prs, "Profile Baseline")
    rows = [
        ["Item", "Value"],
        ["Hardware", "8× NVIDIA B300, full NV18 fabric"],
        ["Parallelism", "TP1 · Ulysses8 · Ring1"],
        ["Attention", "TRTLLM_ATTN"],
        ["Model", "50 DiT blocks · 56 heads · head_dim 128"],
        ["Workload", "124 frames · 768×1344 · 2 profiler steps"],
        ["Diffuse time", "419.38 ms"],
        ["Trace", "minimax_h3_sp8_2step_b300_20260820.nsys-rep"],
    ]
    add_table(slide, 0.85, 1.35, 11.65, 4.70, rows, [2.45, 9.20], 14)
    add_text(
        slide,
        0.88,
        6.32,
        11.5,
        0.40,
        "Local Q/K/V: [1, 4,720, 56, 128]  →  Attention: [1, 37,760, 7, 128]",
        15,
        GREEN,
        True,
        MONO,
        PP_ALIGN.CENTER,
    )
    add_footer(slide, 2)

    # 3. Bottleneck
    slide = new_slide(prs, "Bottleneck")
    rows = [
        ["Stage", "Representative time", "Kernel"],
        ["RoPE products", "51.7 µs", "_rope_products_kernel"],
        ["RoPE combine", "67.1 µs", "_rope_combine_kernel"],
        ["Q pack + A2A", "91.3 + 160.6 µs", "direct_copy → SendRecv"],
        ["K pack + A2A", "90.2 + 152.9 µs", "direct_copy → SendRecv"],
        ["V pack + A2A", "90.6 + 150.7 µs", "direct_copy → SendRecv"],
        ["Attention", "3,441.3 µs", "fmha"],
        ["Output pack + A2A", "97.7 + 156.3 µs", "direct_copy → SendRecv"],
        ["Output unpack", "86.5 µs", "direct_copy"],
    ]
    add_table(slide, 0.75, 1.25, 11.85, 4.75, rows, [3.15, 2.75, 5.95], 12.5)
    add_bullets(
        slide,
        0.92,
        6.18,
        11.6,
        [
            "Each direct-copy moves 64.53 MiB.",
            "5 copies per block · 250 per GPU · 22.37 ms critical copy time.",
        ],
        13,
        0.34,
    )
    add_footer(slide, 3)

    # 4. Optimization 1
    slide = new_slide(prs, "Optimization 1 — Pack Q/K in RoPE")
    rows = [
        ["Current", "Proposed"],
        [
            "_rope_combine_kernel\n→ [T_local, H, D]\n→ transpose().contiguous()\n→ NCCL",
            "_rope_combine_kernel_packed\n→ [P, T_local, B, H/P, D]\n→ NCCL",
        ],
    ]
    add_table(slide, 0.80, 1.38, 11.75, 2.55, rows, [5.875, 5.875], 15)
    add_box(
        slide,
        1.00,
        4.25,
        11.30,
        1.20,
        "dst_rank = head // heads_per_rank\n"
        "rank_head = head % heads_per_rank\n"
        "packed[dst_rank, token, 0, rank_head, dim] = rope_result",
        15,
        MONO,
    )
    add_bullets(
        slide,
        0.95,
        5.78,
        11.5,
        [
            "Add a prepacked All-to-All helper.",
            "Target: remove 100 Q/K copies per GPU; 8.90 ms trace-derived upper bound.",
        ],
        14,
        0.42,
    )
    add_footer(slide, 4)

    # 5. Optimization 2
    slide = new_slide(prs, "Optimization 2 — Pack V in QKV Projection")
    rows = [
        ["Current", "Proposed"],
        [
            "QKV projection\n→ V [T_local, H, D]\n→ transpose().contiguous()\n→ NCCL",
            "QKV projection epilogue\n→ V [P, T_local, B, H/P, D]\n→ NCCL",
        ],
    ]
    add_table(slide, 0.80, 1.38, 11.75, 2.55, rows, [5.875, 5.875], 15)
    add_bullets(
        slide,
        1.00,
        4.38,
        11.3,
        [
            "Write V directly in Ulysses send order.",
            "Keep Q/K on the RMSNorm + RoPE path.",
            "Target: remove 50 V copies per GPU; 4.45 ms trace-derived upper bound.",
            "Requires a custom QKV projection epilogue.",
        ],
        16,
        0.55,
    )
    add_footer(slide, 5)

    # 6. Optimization 3
    slide = new_slide(prs, "Optimization 3 — Remove Output Copies")
    add_box(
        slide,
        0.85,
        1.38,
        11.65,
        1.15,
        "FMHA [B, S, H/P, D]  →  output pack  →  NCCL  →  output unpack  →  [B, S/P, H, D]",
        16,
        MONO,
        PP_ALIGN.CENTER,
    )
    add_bullets(
        slide,
        1.00,
        2.95,
        11.3,
        [
            "Option A: make FMHA write the reverse All-to-All send layout.",
            "Option B: let the output projection consume the NCCL receive layout.",
            "Target: remove 100 output copies per GPU.",
            "Trace-derived upper bound: 9.02 ms.",
            "Highest implementation risk.",
        ],
        17,
        0.62,
    )
    add_footer(slide, 6)

    prs.save(OUT_FILE)
    return OUT_FILE, len(prs.slides)


if __name__ == "__main__":
    output, count = build_deck()
    print(f"wrote {output}")
    print(f"slides {count}")

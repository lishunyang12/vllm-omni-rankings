#!/usr/bin/env python3
"""Regenerate the revised report from checked-in evidence; no GPU or model load."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", "/tmp/h3-report-revision-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
ARCHIVE = REPORT / "archive-20260907"
FIG = HERE / "figures"
DATA = json.loads((HERE / "data.json").read_text())
PDF_NAME = "minimax_h3_653s_to_14s_cumulative_report.pdf"
PUBLIC = "https://lishunyang12.github.io/vllm-omni-rankings/"
BASE = PUBLIC + "scripts/minimax_h3_pro5000_cumulative_report/"
NAVY, BLUE, GREEN, AMBER, SLATE = "#10233f", "#2563eb", "#07856d", "#b86c0a", "#52637b"
PALE, LINE = "#f1f5fa", "#d8e0ec"
W, H = A4
M = 43
CW = W - 2*M
LEGACY_PAGES = [8, *range(15, 25), 26]
NEW_PAGES = 12
TOTAL = NEW_PAGES + len(LEGACY_PAGES)
ORIGINAL_PDF_SHA256 = "a7d1fe30a1fad2688d2f54272250099950944112f4715abe84922e800293fa04"

# Every comparison keeps its own control. No synthetic bridge across cohorts.
STEPS = [
    ("Sage/CAKE attention", "Approximate", 27.493667, 25.706333, "Sage/CAKE with ragged-tail correction"),
    ("QK/V + four-chunk O", "Byte-exact", 25.761333, 25.135333, "Q/K preparation during V exchange; O return/projection pipeline"),
    ("Gate projection overlap", "Byte-exact", 25.117667, 24.340333, "Original BF16 gate GEMM on the side stream during Q/K exchange"),
    ("Views + Q + coarse/fine", "Byte-exact", 24.364667, 24.011667, "Remove layout copies; earlier Q preparation; parallel coarse/fine"),
    ("QKV projection split", "Byte-exact", 24.0268, 23.6654, "First complete QK + V split A/B; later scheduling refinements inherited"),
    ("Paired VAE windows", "Byte-exact", 23.6536, 22.6670, "Balance spatial tiles across ranks and overlap assembly"),
    ("Exact VAE B2 batching", "Byte-exact", 22.6670, 22.3662, "Batch paired tiles; preserve sensitive to_out at B1"),
    ("Fixed allocator segments", "Byte-exact", 22.5924, 21.7712, "expandable_segments:False"),
    ("RDMA across requests", "Byte-exact", 17.9214, 17.5780, "Retain registered communication resources"),
    ("SwiGLU / FC2 quantization", "Byte-exact", 17.5780, 17.4478, "Fuse the activation producer while retaining reference rounding"),
    ("Online MXFP8 full VAE", "Approximate", 17.4168, 15.8322, "Quantize the full H3 decoder online"),
    ("CAKE TMA descriptors", "Byte-exact", 15.8322, 15.6370, "Descriptor-by-value adaptation on the existing CAKE path"),
    ("MXFP8 scheduling bundle", "Byte-exact", 15.6370, 15.1336, "MXFP8 QK/V split + VAE B4 + audio/video decode overlap"),
    ("O producer lookahead", "Byte-exact", 15.1336, 14.9692, "Prequeue four O packs with per-chunk readiness events"),
]

def plots():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "axes.labelcolor":SLATE, "text.color":NAVY,
                         "xtick.color":SLATE, "ytick.color":NAVY, "svg.fonttype":"none",
                         "pdf.fonttype":42})
    fig, ax = plt.subplots(figsize=(8.2,7.9))
    for i, (label, kind, before, after, _) in enumerate(STEPS):
        col = AMBER if kind == "Approximate" else GREEN
        ax.plot([after,before],[i,i],lw=3,color=col,zorder=2)
        ax.scatter(before,i,s=36,color="white",edgecolors=col,zorder=3)
        ax.scatter(after,i,s=36,color=col,zorder=3)
        ax.text(28.45,i,f"{before-after:.4f}s",va="center",fontsize=9)
    ax.axvline(15,ls="--",lw=1,color=BLUE,alpha=.8)
    ax.set_yticks(range(len(STEPS)),[x[0] for x in STEPS],fontsize=9.2)
    ax.invert_yaxis()
    ax.set_xlim(14,30)
    ax.set_xticks([15,18,21,24,27])
    ax.set_xlabel("Complete-request latency (seconds); each row has its own control")
    ax.grid(axis="x",alpha=.15)
    ax.text(28.45,-.9,"Saved",fontsize=9,fontweight="bold")
    fig.subplots_adjust(left=.32,right=.98,top=.96,bottom=.10)
    for ext in ["png","svg"]:
        fig.savefig(FIG/f"adopted-comparisons.{ext}",dpi=190,facecolor="white")
    plt.close(fig)

    vals = [x["mean_s"] for x in DATA["cumulative"]]
    labs = ["Existing\nbaseline", "Full VAE\nMXFP8", "CAKE TMA\ndescriptors",
            "QK/V + B4\n+ audio", "O producer\nlookahead"]
    fig, ax = plt.subplots(figsize=(8.2,4.6))
    ax.plot(range(5),vals,color=BLUE,lw=2.5,marker="o",ms=7)
    ax.axhline(15,color=GREEN,lw=1.3,ls="--",label="15-second target")
    for i,v in enumerate(vals):
        ax.annotate(f"{v:.4f}s",(i,v),xytext=(0,12),textcoords="offset points",
                    ha="center",fontsize=11,fontweight="bold")
        if i:
            ax.text(i-.5,(vals[i-1]+v)/2-.13,f"-{vals[i-1]-v:.4f}s",
                    ha="center",fontsize=9,color=GREEN,
                    bbox={"facecolor":"white","edgecolor":"none","pad":2})
    ax.set_ylim(14.62,17.9)
    ax.set_xlim(-.35,4.35)
    ax.set_xticks(range(5),labs)
    ax.set_ylabel("Complete-request latency (s)")
    ax.grid(axis="y",alpha=.18)
    ax.legend(frameon=False,loc="upper right")
    fig.tight_layout()
    for ext in ["png","svg"]:
        fig.savefig(FIG/f"final-cumulative.{ext}",dpi=200,facecolor="white")
    plt.close(fig)

    samples = DATA["current"]["formal_s"]
    fig, ax = plt.subplots(figsize=(8.2,3.2))
    ax.scatter(range(1,6),samples,color=GREEN,s=75,zorder=3)
    ax.axhline(15,ls="--",color=AMBER,label="15s target")
    ax.axhline(np.mean(samples),lw=1,color=BLUE,label="Mean 14.9692s")
    for i,v in enumerate(samples,1):
        ax.annotate(f"{v:.3f}",(i,v),xytext=(0,-19),textcoords="offset points",ha="center")
    ax.set_ylim(14.94,15.008)
    ax.set_xlim(.5,5.5)
    ax.set_xticks(range(1,6))
    ax.set_xlabel("Formal request (one separate warmup excluded)")
    ax.set_ylabel("Complete-request latency (s)")
    ax.grid(axis="y",alpha=.15)
    ax.legend(frameon=False,loc="lower left",fontsize=8)
    fig.tight_layout()
    for ext in ["png","svg"]:
        fig.savefig(FIG/f"formal-requests.{ext}",dpi=200,facecolor="white")
    plt.close(fig)

    # Dependency diagram: horizontal position is NOT measured time.
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(8.5,3.6))
    ax.set_xlim(-.8,4);ax.set_ylim(-.6,3.1);ax.axis("off")
    def box(x,y,t,c):
        patch=FancyBboxPatch((x-.31,y-.22),.62,.44,boxstyle="round,pad=.04",
                            facecolor=c,edgecolor="none")
        ax.add_patch(patch);ax.text(x,y,t,ha="center",va="center",fontsize=10)
    def arrow(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=10,
                                    color=SLATE,lw=1))
    for i in range(4):
        box(i,2.5,f"Pack {i}","#dce9ff")
        box(i,1.35,f"RDMA {i}","#d5f2e9")
        box(i,.2,f"Gate / O{i}","#e9e5ff")
        arrow(i,2.23,i,1.63);arrow(i,1.08,i,.48)
        if i<3:
            for y in [2.5,1.35,.2]:arrow(i+.37,y,i+.63,y)
    for y,t in [(2.5,"Producer"),(1.35,"Transport"),(.2,"Consumer")]:
        ax.text(-.5,y,t,ha="right",va="center",fontsize=9,color=SLATE)
    ax.text(1.5,-.4,"Arrows show dependencies and stream order, not elapsed time.",
            ha="center",fontsize=9,color=SLATE)
    fig.tight_layout(pad=.5)
    for ext in ["png","svg"]:
        fig.savefig(FIG/f"o-dependencies.{ext}",dpi=200,facecolor="white")
    plt.close(fig)

class Report:
    def __init__(self, path):
        self.c=canvas.Canvas(str(path),pagesize=A4,invariant=1)
        self.c.setTitle("MiniMax-H3: From 653.838 Seconds to a Full-VAE Sub-15s Profile")
        self.c.setAuthor("MiniMax-H3 optimization study")
        self.n=0;self.y=0
        self.titles=[]
    def p(self,text,size=10,leading=14.5,gap=10,color=NAVY,font="Body",width=None,x=M):
        style=ParagraphStyle("p",fontName=font,fontSize=size,leading=leading,
                             textColor=colors.HexColor(color),splitLongWords=True)
        obj=Paragraph(text,style)
        _,h=obj.wrap(width or CW,800)
        assert self.y-h>=53,(self.n,text[:80],self.y,h)
        obj.drawOn(self.c,x,self.y-h)
        self.y-=h+gap
    def start(self,title,kicker="REVISED EDITION · 14 SEPTEMBER 2026"):
        if self.n:self.c.showPage()
        self.n+=1;self.titles.append(title)
        self.c.bookmarkPage(f"page-{self.n}")
        self.c.addOutlineEntry(title,f"page-{self.n}",level=0)
        self.c.setFillColor(colors.HexColor(BLUE))
        self.c.rect(M,H-33,32,3,fill=1,stroke=0)
        self.c.setFont("Bold",7.5);self.c.setFillColor(colors.HexColor(SLATE))
        self.c.drawString(M,H-47,kicker)
        self.y=H-66
        self.p(title,23,28,17,font="Bold")
        self.c.setStrokeColor(colors.HexColor(LINE));self.c.line(M,42,W-M,42)
        self.c.setFont("Body",7);self.c.setFillColor(colors.HexColor(SLATE))
        self.c.drawString(M,27,"MiniMax-H3 · full H3 VAE · selected mean 14.9692s")
        self.c.drawRightString(W-M,27,f"{self.n} / {TOTAL}")
    def table(self,headers,rows,widths,size=8.5):
        ps=ParagraphStyle("cell",fontName="Body",fontSize=size,leading=size+3,
                          textColor=colors.HexColor(NAVY))
        ph=ParagraphStyle("head",parent=ps,fontName="Bold",textColor=colors.white)
        data=[[Paragraph(escape(str(v)),ph) for v in headers]]
        data.extend([[Paragraph(str(v),ps) for v in row] for row in rows])
        t=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor(PALE)]),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),
            ("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),
            ("BOTTOMPADDING",(0,0),(-1,-1),7),
            ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor(LINE))]))
        _,h=t.wrap(CW,800)
        assert self.y-h>=53,("table overflow",self.n,self.y,h)
        t.drawOn(self.c,M,self.y-h);self.y-=h+13
    def image(self,path,maxh=300):
        iw,ih=Image.open(path).size
        w=min(CW,maxh*iw/ih);h=w*ih/iw
        assert self.y-h>=53,("image overflow",self.n,self.y,h)
        self.c.drawImage(str(path),M+(CW-w)/2,self.y-h,w,h)
        self.y-=h+10
    def cards(self,items):
        gap=12;w=(CW-gap*(len(items)-1))/len(items);h=76
        assert self.y-h>=53
        for i,(value,label) in enumerate(items):
            x=M+i*(w+gap)
            self.c.setFillColor(colors.HexColor(PALE))
            self.c.roundRect(x,self.y-h,w,h,7,fill=1,stroke=0)
            self.c.setFont("Bold",22);self.c.setFillColor(colors.HexColor(GREEN))
            self.c.drawString(x+12,self.y-30,value)
            self.c.setFont("Body",8);self.c.setFillColor(colors.HexColor(SLATE))
            self.c.drawString(x+12,self.y-52,label)
        self.y-=h+17
    def save(self):
        assert self.n==NEW_PAGES,self.n
        self.c.save()

def make_revision():
    for name,file in [("Body","DejaVuSans.ttf"),("Bold","DejaVuSans-Bold.ttf"),
                      ("Mono","DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name,"/usr/share/fonts/truetype/dejavu/"+file))
    pdfmetrics.registerFontFamily("Body",normal="Body",bold="Bold",italic="Body",boldItalic="Bold")
    r=Report(HERE/"revision-pages.pdf")

    r.start("MiniMax-H3<br/>From 653.838 Seconds<br/>to a Full-VAE Sub-15s Profile")
    r.p("Cumulative optimization · revised from the original 71-page report",11,16,18,color=SLATE)
    r.cards([("14.9692s","Mean complete request"),("5 / 5","Formal requests below 15s"),("Full H3 VAE","Original decoder architecture")])
    r.p("<b>The new starting point after rollback was 27.4937 seconds.</b> Four-step FastH3, VSA and existing exact engineering stayed in place. The later E4M3-wire / lightweight-decoder branch was retired from the selected path.")
    r.p("The revised story follows the profiling work that came next: gate projection overlap, QKV splitting, Q/K preparation during communication, four-chunk output pipelining, decoder scheduling and allocator changes. These changes are shown separately from new low-precision compute.")
    r.p("<b>Selected result:</b> BF16 communication, MXFP8 DiT and full video VAE, Sage/CAKE attention, and the validated scheduling changes. Timing ends after the complete H.264/AAC MP4 has been saved.")
    r.p("The historical 653.838s result, the old 14.561s endpoint and this revision have different request/configuration contracts. No matched 653.838 / 14.9692 speedup is claimed.",9,13,color=SLATE)
    r.p("Reading guide",13,18,font="Bold")
    r.table(["Pages","Contents"],[
        ("2–4","Rollback, retained foundations and the revised cumulative tables"),
        ("5–9","Measured gains, profile findings and dependency changes"),
        ("10–12","Quality, final samples, scope and evidence"),
        (f"13–{TOTAL}","Retained technical pages from the original report, labeled historical")
    ],[58,CW-58])

    r.start("What changed after the old report")
    r.p("The old 14.561s result inherited E4M3 QKV transport. Subsequent complete-video human review rejected that transport state. Its timing remains historical evidence, and its recommendation is superseded.")
    r.table(["Old choice","Revised selected choice"],[
        ("E4M3 QKV wire","BF16 QKV/O transport. Quantized GEMMs do not imply quantized wire."),
        ("TAEH3 lightweight video decoder","Full H3 VAE, patch-parallel on eight ranks; online MXFP8 added later. TAEH3 was retired from this profile, not declared to have failed its own quality gate."),
        ("Earlier FP8 linear recipe","MXFP8 E4M3 values and E8M0 block-32 scales in the selected DiT/VAE paths."),
        ("Earlier sparse attention kernel","Sage/CAKE approximate attention, with a separately measured later TMA descriptor update."),
        ("TAEH3 rank 0 / Audio VAE rank 1","Full video VAE across eight ranks; rank-0 auxiliary-stream Audio VAE joined by an event.")
    ],[147,CW-147],8.5)
    r.p("Foundations retained from the original work",13,18,font="Bold")
    r.table(["Historical checkpoint","E2E / s"],[
        ("Original H3, 49 diffusion forwards","653.838"),
        ("AdaLN TP1/SP8 + RDMA Ulysses + direct Q/K layout","590.562"),
        ("Dense FastH3, four forwards","66.785"),
        ("FastH3 + VSA + FlashInfer","43.420"),
        ("GPU pixel packing, pinned D2H/encode pipeline, persistent communication and exact VAE/layout cleanup","28.236")
    ],[CW-85,85])
    r.p("These are historical checkpoints under the original prompt. The rollback-era BF16 VSA control was separately measured at 27.493667s. Restoring precision did not discard the existing exact engineering work.",9,13,color=SLATE)

    r.start("Revised cumulative table · part 1")
    r.p("Start from the restored BF16 VSA / full-VAE control. Each arrow is its own measured comparison; restarted controls are preserved.",10,14.5)
    rows=[("Rollback-era baseline","BF16 VSA / linears / wire; full H3 VAE","27.4937","—")]
    for label,kind,b,a,desc in STEPS[:8]:
        rows.append((label,desc+("<br/><b>Approximate compute</b>" if kind=="Approximate" else "<br/>Byte-exact against that reference"),f"{b:.4f}<br/>→ {a:.4f}",f"{b-a:.4f}"))
    r.table(["Step","Adopted change","E2E / s","Saved / s"],rows,[107,237,88,CW-432],8)
    r.p("The 0.3614s split result is the first complete QK + V comparison. Later V-first, gate-after-V and Q/K submission refinements continue that mechanism; their comparisons against fused QKV are not added again.",8.7,12.5)
    r.p("The earlier 24.0117s endpoint was not reproduced unchanged in every later same-feature cohort. Historical means describe their own runs, not a guaranteed stable staircase.",8.7,12.5,color=SLATE)

    r.start("Revised cumulative table · part 2")
    r.p("Introduce the new DiT precision baseline, then continue through decoder quantization and the final scheduling improvements.",10,14.5)
    rows=[("DiT MXFP8 baseline","New low-precision compute and quantization call boundary; BF16 wire and full VAE retained.","17.9240","Unpaired")]
    for idx in [8,9]:
        label,kind,b,a,desc=STEPS[idx]
        rows.append((label,desc,f"{b:.4f}<br/>→ {a:.4f}",f"{b-a:.4f}"))
    rows.append(("Existing later baseline","Qualified inherited configuration; smaller inherited changes are not separate gain rows.","17.4168","—"))
    for label,kind,b,a,desc in STEPS[10:]:
        rows.append((label,desc+("<br/><b>Approximate compute</b>" if kind=="Approximate" else "<br/>Byte-exact against quantized reference"),f"{b:.4f}<br/>→ {a:.4f}",f"{b-a:.4f}"))
    r.table(["Step","Adopted change","E2E / s","Saved / s"],rows,[107,237,88,CW-432],8)
    r.p("<b>No invented precision delta.</b> The 17.9240s DiT MXFP8 experiment changed the quantization call boundary and disabled the old BF16 split. It did not include a matched BF16 control. The difference from the earlier 21.7712s run is not assigned entirely to quantization.",9,13)
    r.p("Only adopted additions saving more than 0.100s appear as gain rows. Existing baseline code is not represented as removed or remeasured after removal. The selected endpoint is model-oproducer-r2.",9,13,color=SLATE)

    r.start("Measured gains across the adopted path")
    r.p("Hollow point: that round's control. Filled point: candidate. Green preserves the selected reference bytes; amber changes numerical precision.",9.5,14)
    r.image(FIG/"adopted-comparisons.png",maxh=510)
    r.p("This comparison chart preserves the actual controls instead of fabricating one uninterrupted sweep. The separate DiT MXFP8 checkpoint has no matched BF16 delta and is intentionally absent from the gain plot.",9,13,color=SLATE)

    r.start("The final cumulative descent")
    r.p("From the existing MXFP8 DiT / full H3 VAE baseline to the selected complete-request endpoint.",10,14.5)
    r.image(FIG/"final-cumulative.png",maxh=300)
    r.cards([("2.4476s","Saved from 17.4168s"),("0.8630s","Byte-exact tail after VAE MXFP8")])
    r.p("<b>Online full-VAE MXFP8: 1.5846s.</b> Original FP32 weights are quantized online to E4M3 values with E8M0 block-32 scales. Dynamic activations, FP16 output and 144 converted layers per worker retain the full decoder architecture.")
    r.p("<b>CAKE descriptors: 0.1952s.</b> The measured change is the TMA descriptor ABI on an already specialized CAKE path. The H3 specialization itself has no isolated gain in this table.")
    r.p("<b>Scheduling bundle: 0.5034s.</b> MXFP8-aware QK/V splitting, B4 decoder scheduling and audio/video overlap were validated together. Their individual contributions were not isolated.")
    r.p("<b>O producer lookahead: 0.1644s.</b> Prequeuing the four pack operations removes host submission dependencies while preserving per-chunk readiness and the original transport protocol.")
    r.p("Five-sample sequential cohorts support these observations. The plot is not a randomized interleaved study, and local kernel savings are not added to the complete-request measurements.",9,13,color=SLATE)

    r.start("What the profiles revealed")
    r.p("The pipeline changes were developed from actual Nsight timelines and data dependencies. A faster attention kernel alone would leave these serialized operations in place.",10,14.5)
    r.table(["Observed serialization","Implemented schedule","Measured evidence"],[
        ("Q/K preparation waited for V exchange.","Prepare Q/K on the side stream while communication continues with V.","QK/V + four-chunk O bundle: 0.6260s E2E."),
        ("The VSA gate projection ran before QKV exchange.","Run the original BF16 gate GEMM on the side stream during Q/K communication.","Gate-only A/B: 0.7773s E2E."),
        ("All O data returned before projection.","Return four 2998-row chunks; gate/project each ready chunk during later transfers.","GPU7 reverse-path example: 8.626 → 6.754ms."),
        ("Q/K/V layout materialization, late Q work and serialized coarse attention.","Use strided views, prepare Q earlier and run coarse/fine concurrently with an explicit join.","Three-change bundle: 0.3530s E2E.")
    ],[153,193,CW-346],8.8)
    r.p("Gate projection: a concrete profile example",13,18,font="Bold")
    r.p("On GPU7, step 3 / layer 0, the original gate GEMM took about <b>4.536ms</b>; <b>4.473ms</b> overlapped the Q exchange interior after rescheduling. The QKV-start-to-V-closing-start span shortened <b>29.115 → 25.997ms</b>.")
    r.p("That gate depends on the original hidden input, not on remote Q/K/V. Its shape and BF16 rounding remain unchanged. The output-side elementwise gate and the MLP SwiGLU are different operations.")
    r.p("Exchange brackets include synchronization and scheduling waits. These overlaps are not measurements of NIC payload-active time. The final O projection tail remains exposed; the report does not claim all communication or computation is hidden.",9,13,color=SLATE)
    r.p('<link href="'+BASE+'update-20260914/evidence/pipeline-profile-breakdown.md" color="'+BLUE+'">Source: original pipeline profile breakdown and per-layer evidence.</link>',8.5,12.5)

    r.start("QKV splitting opens another overlap window")
    r.p("<b>Preparation overlap and projection splitting are different changes.</b> The first moves already available Q/K processing under V communication. The second splits the producer GEMM into QK and V so V computation can overlap Q/K transport.")
    r.image(FIG/"qkv-split-actual-profile.png",maxh=255)
    r.p("Actual historical Nsight figure: GPU7, step 3 / layer 0, fused QKV versus the first QK + delayed-V split. This is a captured timeline, not a schematic or the final MXFP8 trace.",8.5,12.5,color=SLATE)
    r.table(["First split qualification","Result"],[
        ("Complete requests","24.0268 → 23.6654s; 0.3614s observed mean saving"),
        ("Numerical/trace evidence","Ten formal MP4s match; 400 rank/block dependency checks in each trace"),
        ("Remaining exposure","GPU7 V averaged 4.5626ms; 2.7565ms overlapped K interior, with 1.7297ms of V activity after K close")
    ],[160,CW-160],8.6)
    r.p("The subsequent schedule launched V before Q exchange, made the gate wait for V, and moved Q/K preparation into the V transfer window. This refines the same mechanism and is not an extra additive split gain.")
    r.p("The old BF16 split bypassed the newer quantization interface. The final path therefore reimplemented QK/V splitting for MXFP8 and checked actual projection bytes and full videos before admission.",9,13)

    r.start("Remove false dependencies; preserve real ones")
    r.p("The final O improvement keeps four original chunks and four original exchanges. It changes when pack operations enter the producer stream.",10,14.5)
    r.image(FIG/"o-dependencies.png",maxh=228)
    r.p("<b>Before:</b> return chunk 0, then submit pack 1; return chunk 1, then submit pack 2. Those host-order edges delayed work that did not depend on the previous return.")
    r.p("<b>After:</b> queue all four packs before the first blocking exchange. Each RDMA still waits for its own pack-ready event. Separate slots, original exchange order and the consumer join preserve data visibility and buffer lifetime.")
    r.p("Complete-request result: <b>15.1336 → 14.9692s</b>. The gain is 164.4ms; it is not converted into an inferred per-stage overlap percentage.")
    r.p("The decoder and output pipelines",13,18,font="Bold")
    r.table(["Adopted mechanism","Purpose"],[
        ("Paired temporal windows + exact B2","Balance the eight VAE ranks and reduce repeated decode overhead while preserving reference bytes."),
        ("MXFP8 B4 + Audio VAE overlap","Use the qualified B4/B1 plan; launch rank-0 audio on an auxiliary stream before video decode and join before its consumer."),
        ("Pinned D2H / CPU encoding","Overlap output copies and encoding; the request still waits for the complete MP4."),
        ("Resident RDMA + fused SwiGLU/quantization","Avoid repeated registration and an intermediate activation producer without changing its reference values.")
    ],[182,CW-182],8.7)

    r.start("Quality: separate approximation from identity")
    q=DATA["vae_quality"]
    r.cards([(f"{q['ssim_all']:.5f}","SSIM, all 362 frames"),(f"{q['psnr_average']:.4f} dB","PSNR, all 362 frames")])
    r.image(FIG/"vae-comparison.jpg",maxh=230)
    r.p("Same decoded frames at 1.0, 4.5, 9.0 and 14.0s. Top: full H3 VAE before MXFP8. Bottom: full H3 VAE with MXFP8; this candidate MP4 is byte-identical to the final selected formal video.",8.5,12.5,color=SLATE)
    r.table(["Check","Scope and result"],[
        ("VAE precision boundary","All 362 decoded MP4 frames, including codec effects; audio PCM identical. The metrics describe one formal-seed video, not all prompts."),
        ("Scheduling output identity","Six complete MP4s match their corresponding references: one warmup plus five formal requests. These are not six independent prompts/seeds."),
        ("Actual execution checks","400 QKV byte gates, 400 O byte gates, 48 gather markers and 12 audio start/join events."),
        ("Whole-path precision","FastH3, VSA, Sage/CAKE and MXFP8 are approximate. Byte identity applies to the validated engineering changes after selecting the numerical reference.")
    ],[151,CW-151],8.8)
    r.p("Full VAE and BF16 wire remove the old decoder/wire choices. Neither PSNR/SSIM nor the screenshots establish universal perceptual equivalence to the original dense BF16 model.",9,13,color=SLATE)
    r.p('<link href="'+PUBLIC+'fasth3-vae-mxfp8-20260914/videos/baseline.mp4" color="'+BLUE+'">Reference MP4</link> · <link href="'+PUBLIC+'fasth3-vae-mxfp8-20260914/videos/candidate.mp4" color="'+BLUE+'">Selected-output MP4</link>',9,13)

    r.start("Final requests and measurement contract")
    r.image(FIG/"formal-requests.png",maxh=210)
    r.cards([("14.9692s","Mean; sample SD 0.00904s"),("14.976s","Slowest of five requests"),("30.8ms","Mean margin to 15s")])
    w=DATA["current"]["workload"]
    r.table(["Contract","Observed configuration"],[
        ("Timing","Client HTTP POST start to complete MP4 saved; model load, warmup and post-run validation excluded. Concurrency 1; profiler disabled."),
        ("Media","362 frames, 24 fps, effective 1280×704 from a 1280×720 request; H.264 + stereo 32kHz AAC; formal seed 1101."),
        ("Model / parallelism","FastH3 VSA/Data-Free, four forwards, top-k 162, tile 64. DiT TP1/Ulysses8/Ring1; text encoder TP8; full video VAE parallel8."),
        ("GPU / network","8× SM120; recorded device name NVIDIA Graphics Device, 73415MiB each. Four ConnectX-8 groups / eight active 400Gb/s mlx5 RDMA devices."),
        ("Runtime","Driver 580.95.05; PyTorch 2.13.0+cu132; vLLM 0.28.0; OMP 28; model resident; node-default unlocked clocks."),
        ("Run integrity","4,784 source files pinned; source unchanged; zero formal foreign-process samples. Five sequential requests, not an independent interleaved replication.")
    ],[115,CW-115],8.35)
    r.p("The measured five requests meet the strict target. The 30.8ms margin is specific to this fixed workload and node; it is not a latency guarantee for other prompts, loads or two-NIC machines.",8.8,12.5,color=SLATE)

    r.start("Evidence, provenance and retained pages")
    r.p("The revised PDF replaces the original download at the same path. The original 71-page edition is archived unchanged. Its E4M3-dependent 14.561s recommendation is superseded by the later quality review.")
    r.table(["Artifact","What it supports"],[
        ("adopted-cumulative.zh.md","Chinese cumulative table, stage descriptions and source paths"),
        ("data.json + evidence/source-manifest.json","Selected request/quality values, original source hashes and six historical evidence snapshots"),
        ("evidence/pipeline-profile-breakdown.md","Actual gate, QK/V, O and coarse/fine timeline findings"),
        ("evidence/qkv-split-results.md","Initial QK + V full-model timings and trace qualification"),
        ("figures/*.svg and *.png","Exportable charts generated from the recorded values"),
        ("../quality/moon-teahouse-seed1101/README.md","Subsequent complete-video rejection of E4M3 QKV transport"),
        ("../archive-20260907/","Unchanged original PDF, data and generator; historical archive")
    ],[208,CW-208],8.6)
    r.p("Selected formal MP4 SHA-256",10,15,font="Bold")
    r.p(DATA["vae_quality"]["candidate_mp4_sha256"],7.5,11,font="Mono")
    r.p("Selected source-map SHA-256",10,15,font="Bold")
    r.p(DATA["current"]["source_map_sha256"],7.5,11,font="Mono")
    r.p("Historical prompt: "+DATA["historical"]["prompt_sha256"],7.3,10.5,font="Mono")
    r.p("Current prompt: "+DATA["current"]["prompt_sha256"],7.3,10.5,font="Mono")
    r.p("Retained original technical pages",13,18,font="Bold")
    r.p("Pages 13–24 retain original pages 8, 15–24 and 26: original architecture, AdaLN placement, RDMA Ulysses, direct layouts, FastH3, VSA, output processing, persistent communication, packing and exact VAE cleanup. Their values and any quality-section references belong to the historical edition.")
    r.p("The old rejected endpoint, unrelated unsuccessful branches and sub-100ms additions are excluded from this edition's adopted-gain narrative. No GPU experiments were run to create this revision.",9,13,color=SLATE)
    r.p('<link href="'+BASE+'index.html" color="'+BLUE+'">Report landing page</link> · <link href="'+BASE+'update-20260914/data.json" color="'+BLUE+'">Current data</link> · <link href="'+BASE+'archive-20260907/'+PDF_NAME+'" color="'+BLUE+'">Original archive</link>',9,13)
    r.save()
    return r.titles

def append_foundations(titles):
    original=PdfReader(ARCHIVE/PDF_NAME)
    new=PdfReader(HERE/"revision-pages.pdf")
    writer=PdfWriter()
    for i,p in enumerate(new.pages):
        writer.add_page(p)
        writer.add_outline_item(titles[i],i)
    for idx,n in enumerate(LEGACY_PAGES,NEW_PAGES):
        page=writer.add_page(original.pages[n-1])
        content=ContentStream(page.get_contents(),writer)
        filtered=[];block=[];inside=False
        for operands,op in content.operations:
            if op==b"BT":
                inside=True;block=[(operands,op)]
            elif inside:
                block.append((operands,op))
                if op==b"ET":
                    text=" ".join(str(arg) for args,oper in block if oper in [b"Tj",b"TJ"] for arg in args)
                    if "D6 final" not in text and not re.search(r"\bPage \d+\b",text):
                        filtered.extend(block)
                    inside=False;block=[]
            else:filtered.append((operands,op))
        assert not inside
        content.operations=filtered
        page.replace_contents(content)
        stamp=io.BytesIO();c=canvas.Canvas(stamp,pagesize=A4)
        c.setFont("Helvetica",8);c.setFillColor(colors.HexColor(SLATE))
        c.drawString(M,H-22,f"RETAINED HISTORICAL FOUNDATION  ·  Original page {n}  ·  Original request contract")
        c.drawString(M,23,"MiniMax-H3 · revised full-VAE report · historical technical appendix")
        c.drawRightString(W-M,23,f"{idx+1} / {TOTAL}")
        c.save()
        page.merge_page(PdfReader(stamp).pages[0])
        writer.add_outline_item(f"Historical foundation — original page {n}",idx)
    writer.add_metadata({"/Title":"MiniMax-H3: From 653.838 Seconds to a Full-VAE Sub-15s Profile",
                         "/Author":"MiniMax-H3 optimization study",
                         "/Subject":"Revised cumulative optimization, profiling and quality evidence; selected mean 14.9692s"})
    with (REPORT/PDF_NAME).open("wb") as f:writer.write(f)
    return len(writer.pages)

def checksums():
    files=[REPORT/PDF_NAME, REPORT/"index.html", REPORT/"README.md", REPORT/"report_data.json"]
    for line in (ARCHIVE/"SHA256SUMS.original.txt").read_text().splitlines():
        _, relative = line.split("  ", 1)
        files.append(REPORT/relative)
    for folder in [HERE,ARCHIVE]:
        files.extend(p for p in folder.rglob("*") if p.is_file()
                     and p.name not in ["SHA256SUMS.txt","revision-pages.pdf"]
                     and "__pycache__" not in p.parts)
    (REPORT/"SHA256SUMS.txt").write_text("".join(
        hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p.relative_to(REPORT))+"\n"
        for p in sorted(set(files)) if p.exists()))

def main():
    FIG.mkdir(exist_ok=True)
    assert hashlib.sha256((ARCHIVE/PDF_NAME).read_bytes()).hexdigest() == ORIGINAL_PDF_SHA256
    for _,_,before,after,_ in STEPS:assert before-after>.100
    plots()
    for svg in FIG.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    titles=make_revision()
    count=append_foundations(titles)
    assert count==TOTAL
    reader=PdfReader(REPORT/PDF_NAME)
    text="\n".join(p.extract_text() for p in reader.pages)
    for required in ["27.4937","14.9692","0.7773","0.3614","0.5034","43.0414","0.97892"]:
        assert required in text,required
    assert "D6 final" not in text
    assert "14.9300" not in text
    (HERE/"revision-pages.pdf").unlink()
    checksums()
    print(json.dumps({"pdf":str(REPORT/PDF_NAME),"pages":count,"selected_mean_s":14.9692,
                      "source_archive_sha256":hashlib.sha256((ARCHIVE/PDF_NAME).read_bytes()).hexdigest()}))

if __name__=="__main__":
    main()

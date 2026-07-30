#!/usr/bin/env python3
"""v5 cards: capability/architecture explainer (model-card-referenced) + attributed official material."""
from PIL import Image, ImageDraw, ImageFont
import os
G = "/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_edge_gen/cards5"
OFF = "/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_official"
os.makedirs(G, exist_ok=True)
FD = "/usr/share/fonts/opentype/inter"
def F(n, s): return ImageFont.truetype(f"{FD}/{n}", s)
BLACK=(11,13,16); GREEN=(118,185,0); WHITE=(238,240,243); GREY=(150,158,168); DGREY=(92,98,106); PANEL=(24,27,31)
W,H=1280,720
def base():
    im=Image.new("RGB",(W,H),BLACK); d=ImageDraw.Draw(im)
    d.rectangle([0,0,W,6],fill=GREEN); d.rectangle([0,H-6,W,H],fill=GREEN); return im,d
def ctr(d,t,f,y,fill,cx=W//2): d.text((cx,y),t,font=f,fill=fill,anchor="mm")
def attribution(d, text):
    d.text((W-30,H-26),text,font=F("Inter-Medium.otf",18),fill=DGREY,anchor="rm")

# 1 TITLE
im,d=base()
ctr(d,"NVIDIA  ·  Physical AI",F("Inter-SemiBold.otf",26),252,GREEN)
ctr(d,"Cosmos3-Edge",F("Inter-Black.otf",100),330,WHITE)
ctr(d,"Compact 4B world foundation model",F("Inter-Medium.otf",32),412,GREY)
ctr(d,"understand · simulate · interact with the physical world",F("Inter-Medium.otf",24),462,DGREY)
im.save(f"{G}/s1_title.png")

# 2 CAPABILITIES (omni-modal I/O)
im,d=base()
ctr(d,"One omni-modal world model",F("Inter-Bold.otf",52),96,WHITE)
ctr(d,"a single Mixture-of-Transformers checkpoint",F("Inter-Medium.otf",26),150,GREEN)
# in -> out
box=lambda x0,y0,x1,y1,c: d.rounded_rectangle([x0,y0,x1,y1],radius=16,fill=c)
box(120,240,560,470,PANEL); box(720,240,1160,470,PANEL)
ctr(d,"INPUTS",F("Inter-Bold.otf",24),278,GREY,cx=340)
ctr(d,"text · image · video · action",F("Inter-SemiBold.otf",30),340,WHITE,cx=340)
ctr(d,"trajectories",F("Inter-SemiBold.otf",30),380,WHITE,cx=340)
ctr(d,"OUTPUTS",F("Inter-Bold.otf",24),278,GREEN,cx=940)
ctr(d,"text · image · video",F("Inter-SemiBold.otf",30),340,WHITE,cx=940)
ctr(d,"audio · robot action",F("Inter-SemiBold.otf",30),380,WHITE,cx=940)
d.text((W//2,355),"→",font=F("Inter-Black.otf",64),fill=GREEN,anchor="mm")
ctr(d,"world understanding  ·  simulation  ·  future prediction  ·  embodied policy",F("Inter-Medium.otf",23),520,GREY)
attribution(d,"Capabilities: NVIDIA Cosmos3-Edge model card")
im.save(f"{G}/s2_caps.png")

# 3 ARCHITECTURE (MoT two towers)
im,d=base()
ctr(d,"Mixture-of-Transformers",F("Inter-Bold.otf",52),96,WHITE)
ctr(d,"two complementary towers, one model",F("Inter-Medium.otf",26),150,GREEN)
d.rounded_rectangle([130,230,600,470],radius=16,fill=PANEL)
d.rounded_rectangle([680,230,1150,470],radius=16,fill=PANEL)
ctr(d,"Autoregressive tower",F("Inter-Bold.otf",30),278,WHITE,cx=365)
ctr(d,"next-token decoding",F("Inter-Medium.otf",24),322,GREY,cx=365)
ctr(d,"→ text",F("Inter-SemiBold.otf",28),372,GREEN,cx=365)
ctr(d,"Diffusion tower",F("Inter-Bold.otf",30),278,WHITE,cx=915)
ctr(d,"iterative denoising",F("Inter-Medium.otf",24),322,GREY,cx=915)
ctr(d,"→ image · video · audio · action",F("Inter-SemiBold.otf",25),372,GREEN,cx=915)
ctr(d,"heterogeneous modalities, each with its best-suited mechanism",F("Inter-Medium.otf",23),520,GREY)
attribution(d,"Architecture: NVIDIA Cosmos3-Edge model card")
im.save(f"{G}/s3_arch.png")

# 4 FAMILY / positioning
im,d=base()
ctr(d,"Edge = the compact one",F("Inter-Bold.otf",52),96,WHITE)
ctr(d,"same architecture, edge-ready footprint",F("Inter-Medium.otf",26),150,GREEN)
cols=[("Edge","4B","480p",GREEN),("Nano","16B","720p",GREY),("Super","64B","720p",DGREY)]
x0=210; cw=300
for i,(name,size,res,col) in enumerate(cols):
    cx=x0+i*cw+cw//2
    d.rounded_rectangle([x0+i*cw+20,230,x0+i*cw+cw-20,480],radius=16,fill=PANEL if i==0 else (18,20,23))
    if i==0: d.rounded_rectangle([x0+i*cw+20,230,x0+i*cw+cw-20,236],radius=3,fill=GREEN)
    ctr(d,name,F("Inter-Bold.otf",34),290,col,cx=cx)
    ctr(d,size,F("Inter-Black.otf",56),360,WHITE if i==0 else GREY,cx=cx)
    ctr(d,"params",F("Inter-Medium.otf",20),400,DGREY,cx=cx)
    ctr(d,res+" native",F("Inter-Medium.otf",22),445,GREY if i==0 else DGREY,cx=cx)
ctr(d,"one quarter of Nano  ·  runs on a single edge GPU",F("Inter-SemiBold.otf",24),530,GREEN)
attribution(d,"Sizes: NVIDIA Cosmos3-Edge model card")
im.save(f"{G}/s4_family.png")

# 5 ACTION background (official clip composited later; leave a framed box)
im,d=base()
ctr(d,"Physical-AI: action & world dynamics",F("Inter-Bold.otf",44),70,WHITE)
# framed box for the clip (centered), 460x460 at (410,120)
bx0,by0,bx1,by1=410,120,870,580
d.rounded_rectangle([bx0-4,by0-4,bx1+4,by1+4],radius=12,outline=GREEN,width=3)
ctr(d,"forward-dynamics rollout  ·  UMI manipulation",F("Inter-SemiBold.otf",26),620,GREEN)
ctr(d,"a first frame + an action trajectory  →  the model rolls out the video",F("Inter-Medium.otf",22),656,GREY)
attribution(d,"Official demo — Source: NVIDIA Cosmos3-Edge")
im.save(f"{G}/s5_action_bg.png")
print("ACTION_BOX", bx0,by0,bx1-bx0,by1-by0)

# 6 BENCHMARK (official image composited on card + attribution)
im,d=base()
ctr(d,"Best-in-class throughput",F("Inter-Bold.otf",48),80,WHITE)
ctr(d,"highest generation throughput at competitive quality",F("Inter-Medium.otf",25),135,GREEN)
bench=Image.open(f"{OFF}/benchmark-overall.png").convert("RGB")
bw=1120; bh=int(bench.height*bw/bench.width); bench=bench.resize((bw,bh))
# white card behind the (light-background) benchmark image
d.rounded_rectangle([(W-bw)//2-16,190,(W+bw)//2+16,190+bh+32],radius=14,fill=(245,246,248))
im.paste(bench,((W-bw)//2,206))
attribution(d,"Source: NVIDIA Cosmos3-Edge model card")
im.save(f"{G}/s6_bench.png")

# 7 FP8 split overlay (real, our output)
ov=Image.new("RGBA",(W,H),(0,0,0,0)); dr=ImageDraw.Draw(ov)
for i,y in enumerate(range(520,H)):
    a=int(220*(y-520)/(H-520)); dr.line([(0,y),(W,y)],fill=(0,0,0,a))
dr.line([(W//2,0),(W//2,H-130)],fill=(255,255,255,120),width=2)
for cx,txt in [(W//4,"DENSE (BF16)"),(3*W//4,"FP8")]:
    tb=dr.textbbox((0,0),txt,font=F("Inter-Bold.otf",26)); tw=tb[2]-tb[0]
    dr.rectangle([cx-tw//2-16,24,cx+tw//2+16,66],fill=(0,0,0,150))
    dr.text((cx,45),txt,font=F("Inter-Bold.otf",26),fill=WHITE+(255,),anchor="mm")
dr.rectangle([64,H-118,70,H-58],fill=GREEN+(255,))
dr.text((90,H-108),"FP8 on vLLM-Omni  —  visually identical, faster & lighter",font=F("Inter-Bold.otf",32),fill=WHITE+(255,),anchor="lm")
dr.text((90,H-66),"real Cosmos3-Edge output  ·  same seed & scene  ·  LPIPS 0.094  ·  --quantization fp8",font=F("Inter-Medium.otf",21),fill=GREEN+(255,),anchor="lm")
ov.save(f"{G}/s7_fp8.png")

# 8 CLOSE / where it runs
im,d=base()
ctr(d,"Cosmos3-Edge",F("Inter-Black.otf",68),150,WHITE)
ctr(d,"4B  ·  omni-modal  ·  edge-ready",F("Inter-SemiBold.otf",28),210,GREEN)
ctr(d,"Runs on  Jetson AGX Orin · Thor  ·  RTX Pro 6000",F("Inter-Medium.otf",28),330,GREY)
ctr(d,"Supported today on vLLM-Omni",F("Inter-Bold.otf",34),400,WHITE)
ctr(d,"text · image · video · audio · action  —  in one model",F("Inter-Medium.otf",22),460,DGREY)
im.save(f"{G}/s8_close.png")
print("cards5 written")
for f in sorted(os.listdir(G)): print(" ",f)

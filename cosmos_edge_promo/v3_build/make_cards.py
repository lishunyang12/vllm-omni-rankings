#!/usr/bin/env python3
"""Generate title/close cards + per-segment caption overlays (1280x720) for the Edge promo v3."""
from PIL import Image, ImageDraw, ImageFont
import os
G = "/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_edge_gen/cards"
os.makedirs(G, exist_ok=True)
FD = "/usr/share/fonts/opentype/inter"
def F(name, sz): return ImageFont.truetype(f"{FD}/{name}", sz)
BLACK=(11,13,16); GREEN=(118,185,0); WHITE=(238,240,243); GREY=(150,158,168); DGREY=(90,96,104)
W,H=1280,720

def center(d, text, font, y, fill, cx=W//2, anchor="mm"):
    d.text((cx,y), text, font=font, fill=fill, anchor=anchor)

def scrim(img, top=560, alpha0=0, alpha1=210):
    """Bottom gradient scrim for legibility."""
    ov=Image.new("RGBA",(W,H),(0,0,0,0)); dr=ImageDraw.Draw(ov)
    for i,y in enumerate(range(top,H)):
        a=int(alpha0+(alpha1-alpha0)*(y-top)/(H-top))
        dr.line([(0,y),(W,y)],fill=(0,0,0,a))
    return Image.alpha_composite(img.convert("RGBA"),ov)

# ---------- TITLE CARD ----------
img=Image.new("RGB",(W,H),BLACK); d=ImageDraw.Draw(img)
d.rectangle([0,0,W,6],fill=GREEN)
center(d,"NVIDIA  ·  Physical AI", F("Inter-SemiBold.otf",26), 250, GREEN)
center(d,"Cosmos3-Edge", F("Inter-Black.otf",96), 330, WHITE)
center(d,"Compact world foundation model  ·  T2I · T2V · I2V · Action", F("Inter-Medium.otf",30), 410, GREY)
center(d,"Every clip generated on vLLM-Omni  ·  nvidia/Cosmos3-Edge", F("Inter-Medium.otf",24), 470, DGREY)
d.rectangle([0,H-6,W,H],fill=GREEN)
img.save(f"{G}/title.png")

# ---------- CLOSE / SPEC CARD ----------
img=Image.new("RGB",(W,H),BLACK); d=ImageDraw.Draw(img)
d.rectangle([0,0,W,6],fill=GREEN)
center(d,"Cosmos3-Edge", F("Inter-Black.otf",64), 96, WHITE)
center(d,"4B params  ·  edge-ready  ·  supported today on vLLM-Omni", F("Inter-Medium.otf",26), 150, GREEN)
# spec table
rows=[("","Edge","Nano / Super"),
      ("T2V / I2V","480×832","1280×720"),
      ("guidance_scale","5.0","6.0"),
      ("flow_shift","3.0","10.0"),
      ("FP8 speedup","1.15× · LPIPS 0.169","—")]
x0,x1,x2=300,720,980; y=250; rh=64
for i,(a,b,c) in enumerate(rows):
    fa=F("Inter-SemiBold.otf",28) if i==0 else F("Inter-Medium.otf",27)
    col=GREEN if i==0 else WHITE
    d.text((x0,y),a,font=F("Inter-Medium.otf",26),fill=GREY,anchor="lm")
    d.text((x1,y),b,font=fa,fill=(col if i>0 else GREEN),anchor="mm")
    d.text((x2,y),c,font=F("Inter-Medium.otf",26),fill=DGREY,anchor="mm")
    if i==0: d.line([(x0-20,y+rh//2),(x2+120,y+rh//2)],fill=(40,44,50),width=2)
    y+=rh
center(d,"Runs on  Jetson AGX Orin · Thor  ·  RTX Pro 6000", F("Inter-SemiBold.otf",26), y+40, GREY)
d.rectangle([0,H-6,W,H],fill=GREEN)
img.save(f"{G}/close.png")

# ---------- CAPTION OVERLAYS (transparent, bottom lower-third) ----------
def caption(name, title, tag):
    base=Image.new("RGBA",(W,H),(0,0,0,0))
    base=scrim(base, top=520, alpha1=215)
    d=ImageDraw.Draw(base)
    d.rectangle([64,H-118,70,H-58],fill=GREEN+(255,))  # green accent bar
    d.text((90,H-108),title,font=F("Inter-Bold.otf",34),fill=WHITE+(255,),anchor="lm")
    d.text((90,H-66),tag,font=F("Inter-Medium.otf",22),fill=GREY+(255,),anchor="lm")
    base.save(f"{G}/{name}.png")

caption("cap_hero","Image-to-Video:  one dashcam frame → full driving rollout",
        "nvidia/Cosmos3-Edge  ·  I2V  ·  480×832  ·  gs 5.0  ·  flow_shift 3.0  ·  vLLM-Omni")
caption("cap_sort","Text-to-Video:  robotic sorting, straight from a prompt",
        "T2V  ·  93 frames  ·  480×832  ·  generated on vLLM-Omni")
caption("cap_action","Physical-AI:  first-person manipulation rollout",
        "I2V action-conditioned  ·  480×832  ·  generated on vLLM-Omni")

# ---------- FP8 SPLIT OVERLAY (divider + labels + metrics) ----------
base=Image.new("RGBA",(W,H),(0,0,0,0))
base=scrim(base, top=520, alpha1=220)
d=ImageDraw.Draw(base)
d.line([(W//2,0),(W//2,H-130)],fill=(255,255,255,120),width=2)
# top-corner labels
for (cx,txt) in [(W//4,"DENSE (BF16)"),(3*W//4,"FP8")]:
    tb=d.textbbox((0,0),txt,font=F("Inter-Bold.otf",26)); tw=tb[2]-tb[0]
    d.rectangle([cx-tw//2-16,24,cx+tw//2+16,66],fill=(0,0,0,150))
    d.text((cx,45),txt,font=F("Inter-Bold.otf",26),fill=WHITE+(255,),anchor="mm")
d.rectangle([64,H-118,70,H-58],fill=GREEN+(255,))
d.text((90,H-108),"FP8 quantization  —  same quality, 1.15× faster",font=F("Inter-Bold.otf",34),fill=WHITE+(255,),anchor="lm")
d.text((90,H-66),"7.74s → 6.71s   ·   1.15×   ·   LPIPS 0.169   ·   --quantization fp8   ·   832×480 · 49f",
       font=F("Inter-Medium.otf",22),fill=GREEN+(255,),anchor="lm")
base.save(f"{G}/cap_fp8.png")
print("cards + overlays written to", G)
for f in sorted(os.listdir(G)): print(" ", f)

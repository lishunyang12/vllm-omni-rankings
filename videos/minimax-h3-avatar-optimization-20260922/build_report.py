"""Build exportable charts and standalone HTML. No GPU or network use.

Dependencies: matplotlib and markdown-it-py.
"""
import hashlib
import html
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from markdown_it import MarkdownIt

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'
BLUE='#2563ad'; TEAL='#087f75'; ORANGE='#ba5b28'; INK='#172c3a'; GRAY='#7790a2'

def load(name):return json.loads((ROOT/'evidence'/name).read_text())
def save(fig,name):
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight',dpi=180)
    plt.close(fig)

def charts():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':INK,
                         'text.color':INK,'axes.titleweight':'bold','axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#b5c2cc',
                         'grid.color':'#dde5eb','svg.fonttype':'none','svg.hashsalt':'h3-avatar-report-20260922'})
    live=load('current-live-observation.json');m=load('measurements.json')
    budget=119/24; totals=live['all']
    fig,(a,b)=plt.subplots(2,1,figsize=(10,6.4),gridspec_kw={'height_ratios':[1,2]},layout='constrained')
    x=totals['http_seconds']['mean'];outside=totals['outside_http_seconds']['mean']
    a.barh([0],[x],color=BLUE,height=.46,label='Backend HTTP')
    a.barh([0],[outside],left=x,color=ORANGE,height=.46,label='Application / checks')
    a.barh([0],[budget-x-outside],left=x+outside,color='#dce9e5',height=.46,label='Mean spare budget')
    a.text(x/2,0,f'Backend {x:.3f} s',ha='center',va='center',color='white',weight='bold')
    a.text(x+outside+(budget-x-outside)/2,0,f'{budget-x-outside:.3f} s spare',ha='center',va='center',fontsize=9)
    a.annotate(f'Outside HTTP {outside:.3f} s',xy=(x+outside/2,.23),xytext=(3.25,.64),arrowprops={'arrowstyle':'-','color':ORANGE},fontsize=9)
    a.set(yticks=[],xlim=(0,5.4),ylim=(-.7,.9),xlabel='Seconds per newly published continuation')
    a.set_title('Current mean producer budget · 102 accepted continuations',loc='left')
    names=['All','Quiet rest','Speaking','Page turn']
    groups=[live['all'],*[live['by_action'][k] for k in ['rest','speak','turn_page']]]
    ys=np.arange(len(names));p50=[g['ready_seconds']['p50'] for g in groups];p95=[g['ready_seconds']['p95'] for g in groups]
    b.barh(ys,p50,height=.5,color=BLUE,label='P50 publication')
    b.scatter(p95,ys,color=ORANGE,marker='D',label='P95 publication',zorder=3)
    for y,g in zip(ys,groups):
        s=g['ready_seconds'];b.text(.10,y,f"n={s['count']}  |  P50 {s['p50']:.3f} s",va='center',color='white',fontsize=9)
    b.axvline(budget,color=TEAL,linestyle='--',label=f'Playback budget {budget:.3f} s')
    b.set(yticks=ys,yticklabels=names,xlim=(0,5.4),xlabel='Seconds to accepted media publication')
    b.invert_yaxis();b.legend(loc='lower right',fontsize=8);b.grid(axis='x',alpha=.6);b.set_axisbelow(True)
    fig.suptitle('A bounded live observation, not a viewer-response SLO',fontsize=13)
    save(fig,'current-latency')

    cs=m['short_windows']['candidates'];fig,a=plt.subplots(figsize=(10,4.5),layout='constrained')
    x=np.arange(len(cs));means=[c['timing']['mean']/c['content_seconds'] for c in cs];p95=[c['timing']['p95']/c['content_seconds'] for c in cs]
    a.bar(x-.17,means,.32,color=BLUE,label='Mean producer / media')
    a.bar(x+.17,p95,.32,color=ORANGE,label='P95 producer / media')
    a.axhline(1,color=TEAL,linestyle='--',label='Playback deadline')
    for i,c in enumerate(cs):
        t=c['timing'];a.text(i,max(means[i],p95[i])+.035,f"{t['over_playback_budget']}/{t['count']} misses",ha='center',fontsize=9)
    a.set(xticks=x,xticklabels=[f"{c['window_frames']} frames\n{c['content_seconds']:.3f} s new media" for c in cs],ylabel='Normalized generation cost (lower is better)',ylim=(0,1.38))
    a.set_title('Short windows reduce wait but also shrink the compute budget',loc='left')
    a.legend(loc='lower left',fontsize=9);a.grid(axis='y',alpha=.6);a.set_axisbelow(True)
    save(fig,'window-budgets')

    aac=m['long_form_aac'];output={x['name']:x for x in m['long_form_output']}
    pairs=[('Cached constants + early AAC',aac['baseline_seconds'],aac['candidate_seconds']),
           ('GPU blend / RGB fusion',output['baseline']['samples_seconds'],output['output_fused']['samples_seconds']),
           ('Encoder GPU-output residency',output['baseline']['samples_seconds'],output['encoder_resident']['samples_seconds'])]
    fig,a=plt.subplots(figsize=(10,4.3),layout='constrained')
    for i,(name,before,after) in enumerate(pairs):
        y=2-i;p=statistics.median(before);q=statistics.median(after)
        a.plot([p,q],[y,y],color=GRAY,linewidth=3,zorder=1)
        a.scatter(before,[y+.065]*len(before),marker='|',color=BLUE,alpha=.5)
        a.scatter(after,[y-.065]*len(after),marker='|',color=TEAL,alpha=.5)
        a.scatter([p],[y],s=70,color=BLUE,label='Parent median' if i==0 else None,zorder=3)
        a.scatter([q],[y],s=70,color=TEAL,label='Candidate median' if i==0 else None,zorder=3)
        a.text(13.19,y+.23,f'{p:.3f} → {q:.3f} s; change {(q-p)*1000:+.0f} ms',fontsize=9)
    a.set(yticks=[2,1,0],yticklabels=[x[0] for x in pairs],xlabel='Complete HTTP → MP4 seconds (truncated comparison axis)',xlim=(13.18,13.64),ylim=(-.5,2.55))
    a.set_title('Separate matched 15-second campaigns · three formal samples per arm',loc='left',fontsize=12)
    a.legend(loc='lower right',fontsize=9);a.grid(axis='x',alpha=.5);a.set_axisbelow(True)
    save(fig,'matched-campaigns')

    w=m['weight_residency'];fig,(a,b)=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',gridspec_kw={'width_ratios':[1,1.5]})
    for i,(key,col) in enumerate([('baseline',BLUE),('resident',ORANGE)]):
        vals=w[key]['samples_seconds'];med=statistics.median(vals)
        a.scatter([i+(j-4)*.026 for j in range(len(vals))],vals,color=col,s=24,alpha=.7)
        a.hlines(med,i-.25,i+.25,color=col,linewidth=3);a.text(i,5.86,f'{med:.3f} s',ha='center',color=col,weight='bold')
    a.set(xticks=[0,1],xticklabels=['MXFP8 resident\n+ BF16 prefetch','BF16 resident\n+ online packing'],ylabel='Unprofiled HTTP → MP4 seconds',ylim=(5.52,5.92))
    a.set_title('Nine requests per arm',loc='left',fontsize=11)
    phases=[f'h3.profile.dit.step_{i:02d}' for i in range(1,5)];x=np.arange(4)
    for off,key,col,label in [(-.18,'baseline',BLUE,'Original prefetch'),(.18,'resident',ORANGE,'BF16 resident')]:
        vals=[w[key]['nvtx_cpu_phases'][k]['median_cpu_ms'] for k in phases]
        b.bar(x+off,vals,.34,color=col,label=label)
    b.set(xticks=x,xticklabels=['Step 1','Step 2','Step 3','Step 4'],ylabel='CPU NVTX range across-rank median (ms)',ylim=(0,1800))
    b.set_title('Separate diagnostic capture · scopes include waits',loc='left',fontsize=11)
    b.legend(fontsize=8);b.grid(axis='y',alpha=.5);b.set_axisbelow(True)
    fig.suptitle('Removing weight copies did not make the complete request faster',fontsize=13)
    save(fig,'weight-residency')

    fig,a=plt.subplots(figsize=(11,8.5),layout='constrained');a.set(xlim=(0,11),ylim=(0,9));a.axis('off')
    def box(x,y,w,h,title,detail,color='#edf4f7'):
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.10',edgecolor='#bfd1dc',facecolor=color))
        a.text(x+w/2,y+h-.19,title,ha='center',va='top',weight='bold',fontsize=11)
        a.text(x+w/2,y+h-.55,detail,ha='center',va='top',fontsize=9,linespacing=1.7)
    def arrow(p,q,color=GRAY,style='-'):
        a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=13,color=color,linewidth=1.5,linestyle=style,connectionstyle='angle3'))
    box(.3,7.35,3.0,1.15,'Public chat / !ask','Timestamped inbox · deduplication')
    box(.3,5.6,3.0,1.25,'Control and dialogue','Direct supported actions OR\nlocal Qwen3-4B planner')
    box(.3,3.75,3.0,1.3,'Next-window scheduler','Whole-answer queue + scene facts\nOne-clip lookahead')
    box(4.0,7.35,6.6,1.15,'Persistent model service · vLLM-Omni','Exact reference/conditioning reuse · fused Turbo adapter · exact AdaLN')
    box(4.0,5.5,6.6,1.35,'Four-step joint audio/video DiT','TP1 / SP8 · dense Sage · resident MXFP8 projections\nBF16 RDMA transport · fused SwiGLU / FC2')
    box(4.0,3.55,6.6,1.4,'Decode and package','Video VAE PP8 / NVFP4 + FP32 audio VAE\nGPU RGB packing → pinned D2H → MP4 + original PCM')
    box(4.0,1.65,6.6,1.3,'Accepted media and state','Decode / drift / seam / speech checks → atomic publication\nAccepted raw AV tail; one same-parent retry on rejection','#edf7f3')
    box(.3,.05,10.3,1.0,'Persistent FFmpeg → Twitch → viewer','Continuous frame/sample clock · native PCM input · final H.264 / AAC broadcast')
    arrow((1.8,7.3),(1.8,6.9));arrow((1.8,5.55),(1.8,5.1))
    arrow((3.36,4.4),(3.8,7.85));arrow((7.3,7.3),(7.3,6.9));arrow((7.3,5.45),(7.3,5.0));arrow((7.3,3.5),(7.3,3.0));arrow((7.3,1.6),(7.3,1.1))
    arrow((3.94,2.3),(1.8,3.7),TEAL,'--')
    a.text(.35,2.7,'Accepted state only',color=TEAL,fontsize=10,weight='bold')
    a.text(.35,2.22,'No hidden image reset;\nfailed candidates are withheld.',fontsize=9,color=TEAL)
    a.set_title('Generation, playback and conversation advance on separate clocks',loc='left',fontsize=13,pad=16)
    save(fig,'architecture')

CSS='''
:root{--ink:#172c3a;--muted:#566b7a;--blue:#205b93;--line:#dbe3e9;--paper:#fff;--soft:#f2f6f8;--accent:#087f75}
*{box-sizing:border-box}[hidden]{display:none!important}html{scroll-behavior:smooth;scroll-padding-top:28px}body{margin:0;background:var(--soft);color:var(--ink);font:16px/1.7 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--blue);text-underline-offset:3px}a:hover{color:var(--accent)}
.top{padding:48px max(32px,calc((100vw - 1360px)/2));background:#142c3d;color:white;border-bottom:5px solid #30ac9b}.top .eyebrow{font:600 12px/1.5 system-ui;letter-spacing:.17em;text-transform:uppercase;color:#a7d9d2}.top h1{font-size:clamp(30px,4vw,49px);line-height:1.14;max-width:960px;margin:18px 0}.top p{max-width:890px;color:#d4e1e9;margin:16px 0}.top a{color:#c5ece7}.top .links{display:flex;gap:22px;flex-wrap:wrap;font-size:14px}
.layout{max-width:1440px;margin:auto;display:grid;grid-template-columns:254px minmax(0,1fr);gap:30px;padding:30px 32px 70px}aside{position:sticky;top:20px;height:calc(100vh - 40px);overflow:auto;padding:10px 4px;font-size:13px}aside h2{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}aside a{display:block;padding:6px 10px;border-left:2px solid transparent;text-decoration:none;color:#365268;line-height:1.45}aside a:hover,aside a.active{background:#e3eef0;border-color:var(--accent)}aside input{width:100%;border:1px solid #c4d2dc;border-radius:6px;background:white;padding:9px;margin-bottom:14px;color:var(--ink)}
main{min-width:0;background:white;border:1px solid var(--line);border-radius:12px;padding:36px 42px;box-shadow:0 6px 24px #163c4f06}main>h1{font-size:28px;line-height:1.3;margin-top:0}h2{font-size:26px;line-height:1.3;margin:52px 0 20px;padding-top:24px;border-top:2px solid var(--line)}h3{font-size:19px;line-height:1.4;margin-top:30px;overflow-wrap:anywhere}p{margin:15px 0}strong{font-weight:700}.table-wrap{overflow:auto;margin:22px 0;border:1px solid var(--line);border-radius:7px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.5}th,td{padding:12px 13px;vertical-align:top;border-bottom:1px solid var(--line);min-width:90px}th{text-align:left;background:#eaf0f4;font-weight:650;color:#16354c}tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#f8fafb}td:first-child{font-weight:550}code{font:12.5px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;background:#eff3f6;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}pre{padding:20px;background:#132d3e;color:#e1f0f6;border-radius:7px;overflow:auto;font:12.5px/1.75 ui-monospace,Consolas,monospace}pre code{background:none;padding:0;color:inherit;overflow-wrap:normal}blockquote{border-left:4px solid var(--accent);margin:20px 0;padding:4px 20px;background:#eef7f4}li{margin:8px 0}img,svg{max-width:100%;height:auto}.figure{border:1px solid var(--line);padding:16px;margin:25px 0;background:white;border-radius:8px}.figure svg{display:block;width:100%}.figure figcaption{color:var(--muted);font-size:12px;line-height:1.5;margin-top:12px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:8px 0 28px}.metric{padding:18px 15px;border:1px solid #d2e2e5;border-radius:8px;background:#f6faf9}.metric b{display:block;font-size:27px;color:#176862}.metric span{display:block;font-size:12px;color:var(--muted);line-height:1.5}.scope{font-size:13px;color:var(--muted);padding:14px 18px;border-left:3px solid var(--accent);background:#f2f7f7}.footer{margin-top:50px;padding-top:18px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}button{font:inherit;padding:8px 12px;border:1px solid #b6c9d3;background:#fff;border-radius:5px;cursor:pointer}
@media(max-width:1000px){.layout{grid-template-columns:210px minmax(0,1fr);gap:18px;padding:22px 18px}main{padding:25px}.metrics{grid-template-columns:repeat(2,1fr)}}@media(max-width:760px){.layout{display:block;padding:14px}aside{position:static;height:auto;max-height:240px;margin-bottom:20px}main{padding:22px 17px;border-radius:8px}.top{padding:28px 24px}h2{font-size:23px}table{min-width:650px}.metrics{gap:8px}}
@media print{body{background:white;font-size:10pt}aside,.top,.metrics,button{display:none}.layout{display:block;max-width:none;padding:0}main{border:0;box-shadow:none;padding:0}h2{break-after:avoid;margin-top:26px;font-size:17pt}h3{break-after:avoid}.figure{break-inside:avoid;padding:5px}pre{white-space:pre-wrap;background:#f0f4f6;color:black}table{font-size:8pt;min-width:0}th,td{padding:6px}.table-wrap{overflow:visible;border:0}a{color:inherit}p{orphans:3;widows:3}}
'''

def render(mdname,outname,title,subtitle,main_report=False):
    raw=(ROOT/mdname).read_text();md=MarkdownIt('commonmark',{'html':False}).enable('table')
    tokens=md.parse(raw);toc=[];seen=set()
    for i,t in enumerate(tokens):
        if t.type=='heading_open':
            label=tokens[i+1].content
            slug=re.sub(r'[^a-z0-9]+','-',label.lower()).strip('-');base=slug;n=1
            while slug in seen:n+=1;slug=f'{base}-{n}'
            seen.add(slug);t.attrSet('id',slug)
            if t.tag=='h2':toc.append((slug,label))
    body=md.renderer.render(tokens,md.options,{})
    body=body.replace('href="history.md"','href="history.html"').replace('href="report.md"','href="index.html"')
    body=re.sub(r'<table>(.*?)</table>',r'<div class="table-wrap"><table>\1</table></div>',body,flags=re.S)
    def embed(match):
        path=ROOT/match.group(1);alt=match.group(2)
        svg=path.read_text();svg=svg[svg.index('<svg'):]
        # Matplotlib uses local IDs; prefix them to avoid cross-figure collisions.
        prefix=path.stem+'-'
        svg=re.sub(r'id="([^"]+)"',lambda m:'id="'+prefix+m.group(1)+'"',svg)
        svg=re.sub(r'(url\(#|href="#)([^)"\s]+)',lambda m:m.group(1)+prefix+m.group(2),svg)
        svg=svg.replace('<svg ','<svg role="img" aria-label="'+html.escape(html.unescape(alt),quote=True)+'" ',1)
        return '<figure class="figure">'+svg+'<figcaption>'+alt+'</figcaption></figure>'
    body=re.sub(r'<p><img src="(figures/[^\"]+\.svg)" alt="([^\"]*)"\s*/?></p>',embed,body)
    nav=''.join(f'<a href="#{slug}">{html.escape(label)}</a>' for slug,label in toc)
    cards=''
    if main_report:
        cards='''<div class="metrics"><div class="metric"><b>4.958 s</b><span>New media per continuation</span></div><div class="metric"><b>3.972 s</b><span>Publication P50 · 102 clips</span></div><div class="metric"><b>4.205 s</b><span>Publication P95 · same sample</span></div><div class="metric"><b>8 traces</b><span>Historical Nsight reanalysis</span></div></div><p class="scope">Snapshot: 22 Sep 2026, 12:05 UTC. Warm mixed production, including 50 spoken clips. These timings are not chat-to-viewer response latency. The previous stream stopped on brightness drift; indefinite coherence remains unresolved.</p>'''
    script='''<script>
const query=document.getElementById('toc-search');query.addEventListener('input',()=>{for(const a of document.querySelectorAll('nav a'))a.hidden=!a.textContent.toLowerCase().includes(query.value.toLowerCase());});
const observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){for(const a of document.querySelectorAll('nav a'))a.classList.toggle('active',a.hash==='#'+e.target.id)}},{rootMargin:'-10% 0px -75% 0px'});document.querySelectorAll('main h2').forEach(h=>observer.observe(h));
</script>'''
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Measured H3 avatar optimization history, current five-second technology stack, latency boundaries and Nsight evidence."><title>'+html.escape(title)+'</title><style>'+CSS+'</style></head><body><header class="top"><div class="eyebrow">H3 Engineering · measured systems report · September 2026</div><h1>'+html.escape(title)+'</h1><p>'+html.escape(subtitle)+'</p><div class="links"><a href="index.html">Current system</a><a href="history.html">All recorded history</a><a href="'+mdname+'">Markdown source</a><a href="evidence/nsys-reanalysis.json">Nsight evidence</a></div></header><div class="layout"><aside><h2>On this page</h2><label for="toc-search">Find a section</label><input id="toc-search" type="search" placeholder="Filter section titles"><nav>'+nav+'</nav></aside><main>'+cards+body+'<div class="footer">Generated from preserved local measurements. No external fonts, scripts, analytics or GPU work are required to open this report. <button onclick="window.print()">Print / save PDF</button></div></main></div>'+script+'</body></html>'
    (ROOT/outname).write_text(page)

def main():
    charts()
    render('report.md','index.html','From 15-second generation to a five-second interactive avatar','The current technology stack, the optimization decisions that shaped the demo, and the evidence behind both retained changes and rollbacks.',True)
    render('history.md','history.html','Every retained step of the optimization history','Dense H3, FastH3, VSA, MXFP8, native VAE, OpenVDN, five-second continuation and the live application: successful changes, rejected branches and unfinished experiments.')
    manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='bundle-manifest.json' and '__pycache__' not in p.parts}
    (ROOT/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'html':['index.html','history.html'],'figures':5,'files':len(manifest)}))

if __name__=='__main__':main()

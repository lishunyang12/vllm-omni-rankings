"""Compose the retained report and continuity analysis as one standalone HTML.

All figures, styles, scripts, and downloadable evidence are embedded. CPU only.
"""
import base64
import hashlib
import html
import json
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent
TITLE = 'From FastH3 Throughput to H3 Turbo Streaming'
SUBTITLE = 'Systems Design and Evaluation of a Five-Second Interactive Avatar'


def sections(name):
    raw = (ROOT / name).read_text()
    return {title: body.strip() for title, body in re.findall(
        r'^## ([^\n]+)\n(.*?)(?=^## |\Z)', raw, re.M | re.S)}


def numbered(mapping, n):
    return next(body for title, body in mapping.items() if title.startswith(f'{n}. '))


def shift(body, prefix, depth=3):
    """Renumber internal headings without touching code blocks."""
    count = 0
    def replace(match):
        nonlocal count
        count += 1
        label = re.sub(r'^\d+(?:\.\d+)*\.?\s+', '', match.group(1))
        return '#' * depth + f' {prefix}.{count} {label}'
    return re.sub(r'^### (.+)$', replace, body, flags=re.M)


def compose():
    r = sections('report.md')
    c = sections('continuity.md')
    h = sections('history.md')
    intro = numbered(r, 1)
    intro = '\n\n'.join(intro.split('\n\n')[:2]) + '\n\n' + intro[intro.index('### Three separate'):]
    intro = intro.replace('Yes in the bounded current sample', 'Adequate throughput in bounded observations; later deadline misses occurred')
    intro = intro.replace('### Three separate meanings of “real time”', '### 1.1 Evaluation objectives')
    abstract = '''We describe the conversion of an eight-GPU H3 audiovisual generation system from a long-clip throughput benchmark into an interactive streaming avatar. The model path changes from FastH3 T2VA with VSA to MiniMax-H3 Ref2VA with the LightX2V four-step Turbo adapter. The application combines a persistent clean image reference, an accepted audiovisual latent tail, and a new segment instruction. At 960×544 and 24 fps, each continuation publishes 119 new frames (4.958 s). In a fixed 102-continuation production observation, publication latency is 3.972 s at the median and 4.205 s at P95; a later 456-continuation checkpoint records two deadline misses and 1.833 s of relay waiting. A retrospective three-arm continuity experiment, with 60 segments per arm, yields median boundary RGB errors of 7.824 without prefix protection and 3.785 with video-prefix protection. Longer trials nevertheless exhibit appearance drift. The report separates exact serving optimizations from approximate arithmetic, production observations from historical Nsight captures, and boundary invariants from perceptual consistency. The evidence supports bounded real-time production under the evaluated conditions; it does not establish indefinite visual coherence or a viewer-end response-time guarantee.'''
    migration = numbered(r, 3)
    migration = migration.replace('| Attention |', '| Few-step model | FastH3 student/adapter path | H3 Ref2VA base + LightX2V Turbo four-step adapter |\n| Attention |', 1)
    model_paragraphs = numbered(c, 1).split('\n\n')
    migration = model_paragraphs[1] + '\n\n' + migration + '\n\n' + model_paragraphs[-1].replace('in this supplement', 'in this report')
    migration = '![Model and workload migration. The earlier FastH3 T2VA benchmark and the selected H3 Ref2VA Turbo stream use different task paths, geometry and conditioning. The shared runtime foundation does not make their latency ratio an isolated speedup.](figures/model-migration.svg)\n\n' + migration
    state = shift(numbered(c, 2), '4')
    state = '![Reference-conditioned continuation. Clean appearance inputs, the prior raw AV suffix and the next instruction have separate roles. Exact video overlap and independent screening constrain acceptance; failure returns to the same accepted parent.](figures/reference-continuation.svg)\n\n' + state
    method = numbered(c, 3)
    for old, new in [(4, 7), (3, 6), (2, 5), (1, 4)]:
        method = method.replace(f'### 3.{old} ', f'### 4.{new} ')
    infra = shift(numbered(r, 5), '5')
    infra = '![Fidelity contracts of the selected runtime. Exact reuse and scheduling coexist with explicitly approximate model arithmetic; correctness and performance require different evidence.](figures/fidelity-contracts.svg)\n\n' + infra
    for sub, original, label in [
        (6, 6, 'Weight residency and the critical path'),
        (7, 7, 'Video reconstruction and output scheduling'),
        (8, 8, 'Exact reuse of reference and semantic conditioning'),
    ]:
        infra += f'\n\n### 5.{sub} {label}\n\n' + shift(numbered(r, original), f'5.{sub}', 4)
    evaluation = '### 7.1 Profiling protocol and interpretation\n\n' + shift(numbered(r, 14), '7.1', 4)
    evaluation += '\n\n### 7.2 Production latency and response budget\n\n' + shift(numbered(r, 9), '7.2', 4)
    evaluation = evaluation.replace('#### 7.2.3 Viewer response is a different clock', '#### 7.2.3 Viewer response is a different clock\n\n![Schematic response timeline. Text planning overlaps the current video request, the next reply uses a later generation slot, and first meaningful response follows local playback and player delivery. Horizontal distances are illustrative and carry no measured duration.](figures/response-clocks.svg)')
    evaluation += '''\n\n#### 7.2.5 Operational dashboard snapshot

![User-provided Grafana dashboard screenshot. The dashboard exposes request-to-ready timing, GPU activity, playback buffer, broadcast throughput and chat-to-local-playback milestones. It is a separate operational view from the fixed receipt cohorts above.](figures/grafana-screenshot.png)

The supplied dashboard image displays generation P50/P95 values of 3.66/3.67 s, 961 generated clips, three playback waiting events and 4.33 s of accumulated waiting. Its latest chat-response legend shows 3.54 s to text queued, 10.09 s to video ready and 11.02 s to local playback. These are transcribed display values with different metric scopes; they are not recomputed cohort statistics, and the screenshot does not establish their complete sampling definitions. It also displays one active alert and a no-data metric tile. The figure therefore illustrates observability and remaining interruptions, rather than certifying an error-free run. [Original screenshot and hash](evidence/grafana-screenshot.json)
'''
    evaluation += '\n\n### 7.3 Generation-window selection\n\n' + numbered(r, 10)
    ablation = shift(numbered(c, 4), '7.4', 4)
    ablation = ablation.replace('for this supplement', 'for this report')
    ablation = ablation.replace('Figure 2 reports', 'The top-border time-series figure reports')
    ablation = ablation.replace('the [main report\'s longer-run evidence](report.md)', 'the production observations in Section 7.2')
    evaluation += '\n\n### 7.4 Continuity ablation and longer extensions\n\n' + ablation
    limits = numbered(c, 5)
    limits += '''\n\nThe preceding live session stopped after 871 accepted clips when both attempts at the next continuation failed luminance screening. An operator-authorized fresh opening restarted production; this is not seamless autonomous recovery. A historical 7.352 s retried request also caused 2.458 s of relay waiting. Thus quality screening and a one-clip queue trade continuity risk against interruption risk.\n\nThe current screen checks decoded validity and sampled luminance, chroma, saturation, texture and top black bands against a fixed reviewed frame. It can miss incorrect text, repeated actions, impossible reflections or disappearing props. An intentional clothing or background change can also trigger its global appearance thresholds. Reliable semantic state verification and a continuous recovery policy remain open requirements.'''
    ledger = numbered(r, 13)
    ledger = ledger.replace('The [full historical appendix](history.md) includes', 'The retained history inventory includes')
    conclusion = '''The implemented system makes five-second H3 audiovisual continuation practical through a coordinated model, runtime and application design. Ref2VA Turbo establishes the reference-conditioned generation path; exact reuse and scheduling reduce recurring serving work; accepted raw-state continuation and prefix preservation constrain segment transitions. The measured production and ablation results support this operating point, while retained failures identify its limits. Future improvement should be judged jointly by complete-request timing, actual viewer response onset, speech fidelity and long-run visual state, under explicitly matched configurations.'''
    reproducibility = numbered(r, 16)
    reproducibility = reproducibility.replace('- [Historical appendix](history.md): early stages, rollbacks, negative results and every indexed experiment family.\n', '')
    reproducibility = reproducibility.replace('and [history indexer](build_history.py)', '[history indexer](build_history.py), [continuity figure builder](build_continuity.py), [method diagram builder](build_diagrams.py), and [single-file paper builder](build_paper.py)')
    reproducibility += '''\n\nThe [continuity protocol manifest](evidence/continuity-design.json) records the selected model, reference roles, temporal geometry and inspected-source hashes. The [continuity ablation records](evidence/continuity-ablation.json) contain all 180 sanitized segment records and 177 measured boundaries. Unchecked prefix-equality fields are null, rather than recorded as failed checks.\n\nThis HTML embeds its vector figures, presentation code and evidence downloads. It can be read offline without adjacent files. Raw Nsight traces, model weights, media and latent checkpoints are not embedded. Rebuilding the paper requires the source modules and data; rerunning a historical experiment additionally requires its original artifacts and configuration.'''
    native = h['Every indexed native experiment family']
    families = re.findall(r'^### (.+)$', native, re.M)
    family_index = '| Index | Retained experiment family |\n|---:|---|\n' + '\n'.join(f'| {i} | `{name}` |' for i, name in enumerate(families, 1))
    architecture = shift(numbered(r, 4), '3')
    # The vector architecture already expresses this path; omit the duplicate ASCII block.
    architecture = re.sub(r'```text\nTwitch public chat.*?```\n*', '', architecture, flags=re.S)
    live_logic = (ROOT / 'live-logic.md').read_text().strip()
    live_logic += '\n\n' + (ROOT / 'harness.md').read_text().strip()
    live_logic = live_logic.replace('### 3.4 Live program and interaction policy', '### 3.4 Live program and interaction policy\n\n![Live control and scheduling. Active answer segments retain priority, queued interactions replace idle reading, and relay receipts bound generation lookahead. This diagram shows control policy, not guaranteed visual execution.](figures/live-control.svg)')
    audio = '![Native audio flow. H3 jointly generates audio and video; the FP32 audio decoder produces PCM that is preserved through validation and resampled for one final broadcast AAC encode. Generated speech content and transport distortion remain separate diagnoses.](figures/audio-path.svg)\n\n' + shift(numbered(r, 11), '6')
    parts = [
        ('Abstract', abstract),
        ('1. Introduction', intro),
        ('2. Workload and model selection', '### 2.1 From FastH3 to reference-conditioned H3 Turbo\n\n' + migration + '\n\n### 2.2 Design requirements and selection rationale\n\n' + numbered(r, 2)),
        ('3. System architecture', '''![User-provided Twitch screenshot of the running avatar demonstration. Zhiwei reads beside the lotus window, with incense, sweets, bamboo slips and tea on the table; live chat is visible alongside the generated scene. This presentation image is not a temporal-consistency or latency measurement.](figures/live-screenshot.png)

The demonstration presents an established character and scene through a public broadcast interface. The screenshot is included unmodified, with its original player and chat UI; it does not establish that every visible chat request was executed successfully. [Original screenshot and hash](evidence/live-screenshot.json)

''' + architecture + '\n\n' + live_logic),
        ('4. Reference and continuation method', state + '\n\n' + method),
        ('5. Infrastructure optimization', infra),
        ('6. Native audio generation and fidelity', audio),
        ('7. Experimental evaluation', evaluation),
        ('8. Discussion and limitations', limits),
        ('9. Future work', numbered(r, 15)),
        ('10. Conclusion', conclusion),
        ('Appendix A. Historical record', '### A.1 Early trajectory and rejected branches\n\n' + h['Early trajectory, including rejected branches'] + '\n\n### A.2 Scheduling overlap and negative retests\n\n' + h['Exact overlap branch and negative retests'] + '\n\n### A.3 Consolidated decision ledger\n\n' + ledger + '\n\n### A.4 Indexed experiment families\n\nThe full descriptions, document hashes and evidence status are retained in the embedded [history inventory](evidence/history-inventory.json). The index below identifies the scope without repeating the methods discussed in the main text.\n\n' + family_index),
        ('Appendix B. Reproducibility', reproducibility + '''\n\n### B.1 Inspectable prompt examples

The gallery distinguishes source policy templates from exact H3 prompt strings saved in transmitted request records. Original Chinese policy text is retained verbatim; explanatory text is in English. Each renderer example identifies its reference, geometry, seed and intended state changes. The transmitted Picture 2 suffix is included. These are private research requests, not proof that the actions succeeded or were broadcast. [Download all prompt examples](evidence/prompt-examples.json)

PROMPT_EXPLORER

'''),
    ]
    raw = '# ' + TITLE + '\n\n' + '\n\n'.join('## ' + title + '\n\n' + body for title, body in parts)
    raw = re.sub(r'\*\*Table \d+\. (.+?)\*\*', r'**TABLECAPTION: \1**', raw)
    raw = raw.replace('Figures 1–2', 'the continuity figures')
    raw = raw.replace('See the branch tables below.', 'See Appendix A.2.')
    (ROOT / 'paper.md').write_text(raw.rstrip() + '\n')
    return raw


CSS = '''
:root{--ink:#20242a;--muted:#59616b;--link:#235c85;--rule:#bbc2c9;--soft:#f6f7f8}*{box-sizing:border-box}[hidden]{display:none!important}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:#f1f2f3;color:var(--ink);font:17px/1.68 Georgia,"Times New Roman",serif}a{color:var(--link);text-underline-offset:3px}a:hover{color:#153c59}button,input,nav,.metadata,.tools{font-family:system-ui,-apple-system,sans-serif}button{background:white;border:1px solid #aeb7c1;border-radius:3px;padding:7px 11px;cursor:pointer;color:var(--ink)}.layout{display:grid;grid-template-columns:238px minmax(0,1040px);gap:28px;max-width:1366px;margin:0 auto;padding:32px 24px 70px}aside{position:sticky;top:22px;max-height:calc(100vh - 44px);overflow:auto;font:12px/1.5 system-ui,sans-serif;padding:8px 0}aside h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;margin:0 0 16px}aside input{width:100%;padding:8px;margin:9px 0 14px;border:1px solid #bac1c8;border-radius:3px;background:#fff}nav a{display:block;padding:6px 9px;border-left:2px solid transparent;text-decoration:none;color:#4d5864}nav a.sub{font-size:11px;padding-left:19px}nav a:hover,nav a.active{border-left-color:var(--link);background:#e3e9ed;color:#204c6c}main{min-width:0;background:#fff;border:1px solid #d7dbe0;padding:48px 58px;box-shadow:0 2px 14px #00000006}.paper-head{text-align:center;border-bottom:1px solid var(--ink);padding:0 0 25px;margin-bottom:32px}.paper-head h1{font-size:33px;line-height:1.22;letter-spacing:-.3px;font-weight:600;margin:0}.subtitle{font-size:22px;line-height:1.4;margin:12px 0 20px}.metadata{font-size:11px;color:var(--muted);line-height:1.7;letter-spacing:.02em}.tools{display:flex;gap:14px;align-items:center;justify-content:center;font-size:11px;margin-top:20px}h2{font-size:25px;line-height:1.3;font-weight:600;margin:46px 0 18px;padding-top:17px;border-top:1px solid #d5d9dd;overflow-wrap:anywhere}h3{font-size:19px;line-height:1.4;margin:30px 0 13px}h4{font-size:17px;line-height:1.4;margin:24px 0 11px}p{margin:13px 0}p,li{overflow-wrap:break-word}#abstract{margin-top:0;border:0;padding-top:0;font-size:20px;text-align:center}#abstract+p{font-size:15.5px;line-height:1.7;text-align:justify}.table-wrap{overflow-x:auto;margin:25px 0 29px}table{width:100%;border-collapse:collapse;font:12px/1.5 system-ui,sans-serif;border-top:1.5px solid #3d4650;border-bottom:1.5px solid #3d4650}caption{text-align:left;font:14px/1.45 Georgia,serif;padding-bottom:9px;color:#252a30}th,td{padding:9px 10px;text-align:left;vertical-align:top;min-width:70px;border-bottom:1px solid #dfe2e5}th{font-weight:600;background:#f6f7f8;border-bottom:1px solid #59636e}tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fafbfc}td:first-child{font-weight:500}figure{margin:26px 0 32px;padding:0;break-inside:avoid}figure img{display:block;width:100%;height:auto}figure svg{display:block;width:100%;height:auto;max-width:100%}figcaption{font:14px/1.5 Georgia,serif;margin-top:11px}code{font:11.5px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;background:#f1f3f5;padding:2px 3px;overflow-wrap:anywhere}pre{background:#f6f7f8;border-left:2px solid #798794;padding:16px 18px;overflow:auto;line-height:1.65}pre code{padding:0;background:none;overflow-wrap:normal}blockquote{margin:24px 14px;padding:9px 16px;background:#f8f9fa;border:1px solid #e1e5e8;text-align:center;font-size:17px}li{margin:7px 0}.paper-footer{font:11px/1.6 system-ui,sans-serif;color:var(--muted);margin-top:38px;border-top:1px solid var(--rule);padding-top:18px}.attachment-list{font:12px/1.6 ui-monospace,monospace;padding-left:20px}.attachment-list small{font-size:10px;color:var(--muted);overflow-wrap:anywhere}.caption-note{font-size:13px;color:var(--muted)}
@media(max-width:1100px){.layout{grid-template-columns:200px minmax(0,1fr);gap:20px;padding:22px 18px}main{padding:36px 32px}}@media(max-width:760px){body{font-size:16px}.layout{display:block;padding:12px}aside{position:static;max-height:260px;padding:8px 10px;margin-bottom:18px}main{padding:28px 18px}.paper-head h1{font-size:28px}.subtitle{font-size:19px}h2{font-size:23px}table{min-width:620px}.tools{flex-wrap:wrap}#abstract+p{text-align:left}}
@page{size:A4;margin:18mm 16mm}@media print{body{background:white;font-size:10pt;line-height:1.5}.layout{display:block;padding:0;max-width:none}aside,.tools{display:none}main{border:0;box-shadow:none;padding:0}.paper-head h1{font-size:22pt}.subtitle{font-size:15pt}h2{font-size:16pt;break-after:avoid;margin-top:25px}h3{font-size:12pt;break-after:avoid}h4{font-size:11pt;break-after:avoid}p{orphans:3;widows:3}table{font-size:7.5pt;min-width:0}th,td{padding:5px;min-width:0}thead{display:table-header-group}tr{break-inside:avoid}.table-wrap{overflow:visible}caption,figcaption{font-size:9pt}pre{white-space:pre-wrap;overflow:visible}code{font-size:8pt}a{color:inherit;text-decoration:none}#abstract+p{font-size:9.5pt}.attachment-list small{font-size:7pt}}
'''


CSS += '''
.prompt-card{margin:14px 0;border:1px solid #c4cdd5;border-radius:4px;padding:12px 16px;break-inside:avoid}.prompt-card summary{cursor:pointer;font:600 14px/1.5 system-ui,sans-serif}.prompt-meta{font:12px/1.6 system-ui,sans-serif;color:#59616b}.prompt-source{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.6}.prompt-source code{white-space:inherit;overflow-wrap:inherit}.prompt-tools{display:flex;gap:10px;margin:15px 0}
'''


def prompt_gallery():
    rows=json.loads((ROOT/'evidence/prompt-examples.json').read_text())
    content='<div class="prompt-tools"><button type="button" onclick="document.querySelectorAll(\'.prompt-card\').forEach(x=>x.open=true)">Expand all prompts</button><button type="button" onclick="document.querySelectorAll(\'.prompt-card\').forEach(x=>x.open=false)">Collapse all</button></div>'
    for i,row in enumerate(rows):
        content+='<details class="prompt-card" id="prompt-'+html.escape(row['id'],quote=True)+'"'+(' open' if i==0 else '')+'><summary>'+html.escape(row['title'])+' · '+html.escape(row['kind'])+'</summary>'
        meta=row['scope']+'. '+row['note']
        if 'geometry' in row:
            g=row['geometry']
            meta+=f" Reference: {row['reference_image']}; {row['reference_copies']} image blocks; {row['external_audio_references']} external audio references. {g['width']}×{g['height']}, {g['fps']} fps, {g['window_frames']} frames, {g['overlap_frames']} overlapping frames; seed {row['seed']}."
        else:meta+=' Source: '+row['source']+'.'
        # Preserve the exact captured text without literal trailing whitespace in HTML.
        prompt_html = re.sub(r'[ \t]+(?=\n|$)', lambda m: ''.join(f'&#{ord(c)};' for c in m[0]), html.escape(row['prompt']))
        content+='<p class="prompt-meta">'+html.escape(meta)+'</p><pre class="prompt-source"><code>'+prompt_html+'</code></pre>'
        if 'intended_state_delta' in row:
            content+='<p class="prompt-meta">Intended state change, not an observed-state commit:</p><pre class="prompt-source"><code>'+html.escape(json.dumps(row['intended_state_delta'],ensure_ascii=False,indent=2))+'</code></pre>'
        content+='<p class="prompt-meta">Prompt SHA256: '+row['prompt_sha256']+'</p></details>'
    return content


def main():
    global CSS
    font=base64.b64encode((ROOT/'fonts/prompt-cjk-subset.otf').read_bytes()).decode()
    CSS += '@font-face{font-family:H3PromptCJK;src:url(data:font/otf;base64,'+font+');font-display:swap}.prompt-source,.prompt-source code{font-family:ui-monospace,SFMono-Regular,Consolas,H3PromptCJK,monospace}'
    raw = compose()
    md = MarkdownIt('commonmark', {'html': False}).enable('table')
    tokens = md.parse(raw)
    toc, seen, active_section, table_number = [], set(), '', 0
    for i, t in enumerate(tokens):
        if t.type == 'heading_open':
            label = tokens[i + 1].content
            slug = re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')
            base, suffix = slug, 1
            while slug in seen:
                suffix += 1
                slug = f'{base}-{suffix}'
            seen.add(slug)
            t.attrSet('id', slug)
            if t.tag in ('h2', 'h3'):
                toc.append((slug, label, t.tag))
                active_section = re.sub(r'^(?:[A-Z0-9]+\.)+\s*', '', label)
        elif t.type == 'table_open':
            table_number += 1
            t.meta['number'] = table_number
            t.meta['caption'] = active_section
    def table_open(ts, idx, options, env):
        t = ts[idx]
        caption = f'Table {t.meta["number"]}. {t.meta["caption"]}.'
        return '<div class="table-wrap"><table><caption>' + html.escape(caption) + '</caption>\n'
    md.renderer.rules['table_open'] = table_open
    md.renderer.rules['table_close'] = lambda *args: '</table></div>\n'
    body = md.renderer.render(tokens, md.options, {})
    body = re.sub(r'^<h1.*?</h1>\s*', '', body, count=1, flags=re.S)
    # Move existing method-specific captions into their proper table caption.
    def explicit_caption(m):
        title, note, start, numbered_caption = m.groups()
        number = re.match(r'Table \d+\.', numbered_caption).group(0)
        note_html = '<p class="caption-note">' + note.strip() + '</p>' if note.strip() else ''
        return note_html + start + '<caption>' + number + ' ' + title + '</caption>'
    body = re.sub(r'<p><strong>TABLECAPTION: (.*?)</strong>(.*?)</p>\s*(<div class="table-wrap"><table>)<caption>(.*?)</caption>', explicit_caption, body, flags=re.S)
    figure_number = 0
    def embed_figure(m):
        nonlocal figure_number
        figure_number += 1
        path = ROOT / m.group(1)
        alt = re.sub(r'^Figure \d+\.\s*', '', html.unescape(m.group(2)))
        if path.suffix == '.png':
            payload = base64.b64encode(path.read_bytes()).decode()
            return f'<figure class="raster" id="figure-{figure_number}"><img loading="lazy" src="data:image/png;base64,{payload}" alt="' + html.escape(alt, quote=True) + f'"><figcaption><strong>Figure {figure_number}.</strong> ' + html.escape(alt) + '</figcaption></figure>'
        svg = path.read_text()
        svg = '\n'.join(line.rstrip() for line in svg.splitlines())
        svg = svg[svg.index('<svg'):]
        prefix = f'fig-{figure_number}-'
        svg = re.sub(r'id="([^"]+)"', lambda x: 'id="' + prefix + x.group(1) + '"', svg)
        svg = re.sub(r'(url\(#|href="#)([^)"\s]+)', lambda x: x.group(1) + prefix + x.group(2), svg)
        svg = svg.replace('<svg ', '<svg role="img" aria-label="' + html.escape(alt, quote=True) + '" ', 1)
        return f'<figure id="figure-{figure_number}">' + svg + f'<figcaption><strong>Figure {figure_number}.</strong> ' + html.escape(alt) + '</figcaption></figure>'
    body = re.sub(r'<p><img src="(figures/[^\"]+\.(?:svg|png))" alt="([^\"]*)"\s*/?></p>', embed_figure, body)
    body = body.replace('<p>PROMPT_EXPLORER</p>',prompt_gallery())
    attachments = {}
    def local_link(m):
        href = html.unescape(m.group(1))
        if href.startswith(('https:', 'http:', '#', 'mailto:')):
            return m.group(0)
        if href in ('report.md', 'index.html'):
            return 'href="#1-introduction"'
        if href in ('history.md', 'history.html'):
            return 'href="#appendix-a-historical-record"'
        path = ROOT / href
        if not path.is_file():
            raise FileNotFoundError(path)
        key = 'evidence-' + re.sub(r'[^a-z0-9]+', '-', href.lower()).strip('-')
        attachments[href] = key
        return 'href="#' + key + '"'
    body = re.sub(r'href="([^"]+)"', local_link, body)
    # Bundle the technical manuscript, harness builder and required font license.
    for name in ('paper.md', 'report.md', 'continuity.md', 'history.md', 'live-logic.md', 'harness.md', 'build_harness.py', 'fonts/NotoSansCJK-LICENSE.txt'):
        attachments[name] = 'evidence-' + name.replace('.', '-')
    body += '<h2 id="appendix-c-embedded-evidence">Appendix C. Embedded evidence</h2><p>Each link downloads a byte-for-byte evidence or source file embedded in this HTML. No network request is required. Hashes identify the bundled version, not a new experiment.</p><ul class="attachment-list">'
    for name, key in sorted(attachments.items()):
        payload = (ROOT / name).read_bytes()
        mime = 'application/json' if name.endswith('.json') else 'text/plain;charset=utf-8'
        uri = 'data:' + mime + ';base64,' + base64.b64encode(payload).decode()
        digest = hashlib.sha256(payload).hexdigest()
        body += '<li id="' + key + '"><a href="' + uri + '" download="' + html.escape(name.replace('/', '--'), quote=True) + '">' + html.escape(name) + '</a> · ' + f'{len(payload):,} bytes<br><small>SHA256 {digest}</small></li>'
    body += '</ul>'
    toc.append(('appendix-c-embedded-evidence', 'Appendix C. Embedded evidence', 'h2'))
    nav = ''.join('<a class="' + ('sub' if tag == 'h3' else 'section') + '" href="#' + slug + '">' + html.escape(label) + '</a>' for slug, label, tag in toc)
    script = '''<script>
const links=[...document.querySelectorAll('nav a')];
document.getElementById('toc-search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();for(const a of links)a.hidden=!a.textContent.toLowerCase().includes(q)});
if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting)for(const a of links)a.classList.toggle('active',a.hash==='#'+e.target.id)},{rootMargin:'-8% 0px -78% 0px'});document.querySelectorAll('main h2,main h3').forEach(h=>observer.observe(h))}
</script>'''
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="A self-contained systems paper on H3 Turbo streaming, performance engineering, reference conditioning and temporal consistency."><title>' + TITLE + ' — ' + SUBTITLE + '</title><style>' + CSS + '</style></head><body><div class="layout"><aside aria-label="Table of contents"><h2>Contents</h2><label for="toc-search">Find a section</label><input id="toc-search" type="search" placeholder="Filter headings"><nav>' + nav + '</nav></aside><main><header class="paper-head"><h1>' + TITLE + '</h1><p class="subtitle">' + SUBTITLE + '</p><div class="metadata">SYLAR · Systems technical report · 22 September 2026<br>Eight SM120 / GB202 GPUs · Retrospective experiments and bounded production observations</div><div class="tools"><span>Self-contained HTML · English</span><button type="button" onclick="window.print()">Print / save PDF</button><a href="#appendix-c-embedded-evidence">Embedded evidence</a></div></header>' + body + '<footer class="paper-footer">All figures and downloadable evidence are embedded. Performance statistics retain their stated September 22 cohorts. Additional action prompts and state-reference examples are explicitly marked as private research.</footer></main></div>' + script + '</body></html>'
    (ROOT / 'index.html').write_text(page)
    print(json.dumps({'single_html': 'index.html', 'bytes': len(page.encode()), 'tables': table_number, 'figures': figure_number, 'embedded_files': len(attachments)}))


if __name__ == '__main__':
    main()

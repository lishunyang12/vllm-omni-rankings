"""Build vector method diagrams; geometry is schematic, not measured timing."""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent
BLUE = '#eaf2fa'
GREEN = '#eaf5ef'
ORANGE = '#fff2e5'
GRAY = '#f2f4f6'


class Diagram:
    def __init__(self, title, subtitle, height=650):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#617283"/></marker></defs>',
            '<rect width="100%" height="100%" fill="white"/>']
        self.text(30, 36, title, 24, weight='bold', serif=True)
        self.text(30, 65, subtitle, 14, color='#586674')

    def text(self, x, y, text, size=17, color='#202c38', anchor='start', weight='normal', serif=False):
        family = 'Georgia,serif' if serif else 'Arial,sans-serif'
        self.parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="{family}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(text)}</text>')

    def box(self, x, y, w, h, title, lines=(), fill=BLUE, size=16):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="#b7c4cf"/>')
        self.text(x + w/2, y+29, title, size+1, anchor='middle', weight='bold')
        for i, line in enumerate(lines):
            self.text(x+w/2, y+56+i*23, line, size, anchor='middle')

    def arrow(self, points, label=None, label_at=None, dashed=False):
        coords = ' '.join(f'{x},{y}' for x, y in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ''
        self.parts.append(f'<polyline points="{coords}" fill="none" stroke="#617283" stroke-width="2" marker-end="url(#arrow)"{dash}/>')
        if label:
            self.text(*label_at, label, 14, color='#4b5c6d', anchor='middle')

    def save(self, name):
        (ROOT/'figures'/f'{name}.svg').write_text('\n'.join(self.parts + ['</svg>']))


def main():
    d = Diagram('Model migration and workload selection', 'Separate the change in generation task from the serving optimizations inherited by the new task.', 670)
    d.box(35,105,470,300,'Earlier: FastH3 T2VA',[
        'Four-step FastH3 student / adapter',
        'VSA attention in the selected benchmark',
        '1280 × 704 · 362 frames · 24 fps',
        '15.083 s of completed media',
        'Objective: complete a finite clip in real time',
        'No claim of a persistent interactive world'],BLUE)
    d.box(695,105,470,300,'Selected: H3 Ref2VA + Turbo',[
        'LightX2V four-step Ref2VA adapter',
        'Dense Sage · fixed reference + raw AV tail',
        '960 × 544 · 124 generated frames · 24 fps',
        '119 new frames = 4.958 s per continuation',
        'Objective: accept the next instruction sooner',
        'State and quality checked across segments'],GREEN)
    d.arrow([(520,240),(680,240)],'Task migration',(600,215))
    d.box(175,460,850,115,'Inherited runtime mechanisms',[
        'Persistent service · exact AdaLN · BF16 RDMA · MXFP8 packing fusion · native AV decode',
        'VSA-specific schedules require separate admission; they do not transfer automatically'],GRAY)
    d.arrow([(270,405),(270,445)])
    d.arrow([(930,405),(930,445)])
    d.text(600,620,'Different task, geometry and conditioning: the timing ratio is not an isolated infrastructure speedup.',16,anchor='middle')
    d.save('model-migration')

    d = Diagram('Live interaction and scheduling', 'Control flow for the selected reading profile; a completed request does not prove visual action success.',750)
    d.box(35,105,305,112,'Real viewer message',['Timestamp · deduplicate · expire','Conversation or supported action'])
    d.box(395,105,345,112,'Prepare next response',['Direct action OR local text model','Persona + scene + recent dialogue'])
    d.box(795,105,370,112,'Durable reply queue',['Reply identity · ordered segments','Small amount of prepared work'])
    d.arrow([(340,161),(395,161)])
    d.arrow([(740,161),(795,161)])
    d.box(795,283,370,165,'Next eligible generation slot',[
        '1  Continue an active segmented turn',
        '2  Select the oldest queued reply',
        '3  Otherwise: read / occasional page turn'],GREEN)
    d.arrow([(980,217),(980,283)])
    d.box(395,300,345,128,'Compile one segment',['Current words or remaining action','Keep attention across reply segments','Reference + accepted AV context'])
    d.arrow([(795,365),(740,365)])
    d.box(35,300,305,128,'Generate and validate',['H3 Turbo → admission checks','Retry from accepted parent once','See continuation protocol'],ORANGE)
    d.arrow([(395,365),(340,365)])
    d.box(35,520,305,115,'Accepted publication',['New media + original PCM','Raw-tail checkpoint + receipt'],GREEN)
    d.arrow([(185,428),(185,520)])
    d.box(395,520,345,115,'Persistent relay',['Each fresh clip once · frame/sample clock','Start / finish receipts → dialogue state'])
    d.arrow([(340,577),(395,577)])
    d.box(795,520,370,115,'Twitch and viewer',['Transport + player buffering','Viewer response occurs after local play'])
    d.arrow([(740,577),(795,577)])
    d.arrow([(570,635),(570,682),(1180,682),(1180,365),(1165,365)],dashed=True)
    d.text(850,712,'Relay progress permits the next slot; one-clip lookahead limits future footage.',15,anchor='middle')
    d.save('live-control')

    d = Diagram('Appearance reference, motion context and exact boundary', 'Three input roles plus an independent quality anchor; reference conditioning is not pixel replacement.',880)
    d.box(25,105,345,105,'Clean scene reference R',['One distinct image, supplied twice','Identity · composition · canonical props'],BLUE)
    d.box(435,105,330,105,'Segment instruction P',['Current reply span or action phase','Persona + intended scene facts'],BLUE)
    d.box(830,105,345,105,'Accepted raw AV tail T',['Recent pose, motion and sound','Can also carry inherited errors'],GREEN)
    d.box(830,265,345,115,'Two uses of the visual suffix',['Noised guide: τT + (1 − τ)ε, τ = 0.8','Exact target overlap: unmodified T','Audio tail supplies recent context'],GREEN,size=15)
    d.arrow([(1000,210),(1000,265)])
    d.box(435,280,330,100,'H3 Ref2VA Turbo',['Joint audiovisual generation','Video prefix held at every update'],BLUE)
    d.arrow([(197,210),(197,330),(435,330)])
    d.arrow([(600,210),(600,280)])
    d.arrow([(830,325),(765,325)])
    d.box(435,445,330,115,'Candidate admission',['Raw AV shape + exact video prefix','Decoded seam + sampled drift','Expected speech audibility'],ORANGE,size=15)
    d.arrow([(600,380),(600,445)])
    d.box(25,450,345,110,'Separate RGB quality anchor',['Reviewed fixed comparison frame','CPU screen only; not a model input'],GRAY)
    d.arrow([(370,505),(435,505)])
    d.box(830,450,345,110,'Rejected candidate',['Discard its backend branch','Restore the same accepted parent'],ORANGE)
    d.arrow([(765,505),(830,505)],'Fail',(798,483))
    d.arrow([(1000,450),(1000,404),(1185,404),(1185,88),(1000,88),(1000,105)],dashed=True)
    d.box(435,640,330,110,'Accepted candidate',['Publish the new 119 frames','Save bounded raw AV tail'],GREEN)
    d.arrow([(600,560),(600,640)],'Pass',(640,605))
    d.box(830,640,345,110,'Second rejection',['Stop publication','Continuous recovery remains open'],ORANGE)
    d.arrow([(1000,560),(1000,640)],'Retry also fails',(1080,605))
    d.arrow([(435,695),(400,695),(400,88),(1000,88),(1000,105)],dashed=True)
    d.text(600,807,'Exact latent-prefix equality constrains the boundary; it does not guarantee long-run identity or object fidelity.',15,anchor='middle')
    d.save('reference-continuation')

    d = Diagram('Response latency spans several independent clocks', 'Schematic event order only: horizontal distances are not measured durations.',710)
    labels=[('Input / text',140),('GPU producer',250),('Media ready',350),('Local playback',460),('Viewer',570)]
    for name,y in labels:
        d.text(25,y,name,16,weight='bold')
        d.arrow([(175,y+25),(1170,y+25)])
    d.box(220,97,215,64,'Prepare reply',[],BLUE,15)
    d.text(220,85,'Message received',13)
    d.box(185,207,270,64,'Current window already running',[],GRAY,14)
    d.box(490,207,180,64,'Generate reply A',[],BLUE,14)
    d.box(790,207,215,64,'Generate reply B',[],BLUE,14)
    d.box(690,307,100,64,'Check A',[],ORANGE,14)
    d.box(1035,307,100,64,'Check B',[],ORANGE,14)
    d.box(185,417,605,64,'Earlier accepted footage is playing',[],GRAY,14)
    d.box(790,417,345,64,'Answer A: remain engaged',[],GREEN,14)
    d.box(1135,417,50,64,'B',[],GREEN,14)
    d.arrow([(435,140),(470,140),(470,200)],dashed=True)
    d.arrow([(670,250),(680,250),(680,290),(730,290),(730,307)],dashed=True)
    d.arrow([(790,350),(790,417)],dashed=True)
    d.arrow([(845,485),(925,530)],'Twitch / player delay',(970,515),dashed=True)
    d.box(925,537,240,64,'Meaningful response visible',[],GREEN,14)
    d.text(845,502,'First response within A',12,anchor='middle')
    d.text(600,646,'A reply can span A → B without an idle reading reset; new messages wait for an eligible decision.',16,anchor='middle')
    d.text(600,679,'Production < playback budget does not imply message-to-viewer latency < one window.',16,anchor='middle',weight='bold')
    d.save('response-clocks')

    d = Diagram('Native audiovisual generation and the broadcast audio path', 'Ordinary dialogue uses prompt-defined voice and generated audio context; no per-line TTS stage is required.',675)
    d.box(30,105,300,100,'Joint H3 audiovisual DiT',['Four-step Ref2VA Turbo','Shared speech / motion generation'],BLUE)
    d.box(445,105,315,100,'Video reconstruction',['Native video VAE · NVFP4','Frames → persistent H.264 encoding'],ORANGE,size=15)
    d.box(865,105,305,100,'Broadcast mux / transport',['Continuous frame/sample timestamps','Twitch ingest → viewer'],BLUE,size=15)
    d.arrow([(330,155),(445,155)])
    d.arrow([(760,155),(865,155)])
    d.box(30,290,300,100,'Native audio VAE',['FP32 decoder','Generated 32 kHz float PCM'],GREEN)
    d.arrow([(180,205),(180,290)])
    d.box(435,290,335,100,'Preserve and inspect PCM',['Keep original waveform sidecar','Audibility / media validity checks'],GREEN)
    d.arrow([(330,340),(435,340)])
    d.box(865,290,305,100,'Relay audio processing',['Resample to 48 kHz stereo','One final AAC encode · 192 kb/s'],ORANGE,size=15)
    d.arrow([(770,340),(865,340)])
    d.arrow([(1017,290),(1017,205)])
    d.box(120,485,960,107,'Three distinct fidelity questions',[
        'Content and prosody: already determined in the generated waveform',
        'Boundary continuity: raw audio context and sample alignment   |   Transport fidelity: resampling and final codec'],GRAY,size=15)
    d.text(600,638,'Preserving PCM avoids an extra lossy relay input; it cannot remove words already generated by the model.',16,anchor='middle')
    d.save('audio-path')

    d = Diagram('Exact reuse and approximate computation in the selected stack', 'Precision changes and exact infrastructure work are evaluated against different reference contracts.',680)
    cols=[30,430,830]
    d.box(cols[0],105,340,165,'Exact reuse / scheduling',[
        'Validated AdaLN tables',
        'Fixed image / conditioning cache',
        'Persistent buffers and registrations',
        'Matched packing / output fusion'],GREEN,size=15)
    d.box(cols[1],105,340,165,'Approximate model execution',[
        'Four-step Turbo distillation',
        'All-step Sage attention',
        'MXFP8 main projections',
        'NVFP4 video-VAE projections'],ORANGE,size=15)
    d.box(cols[2],105,340,165,'Preserved precision / state',[
        'BF16 communication transport',
        'FP32 audio VAE',
        'Accepted raw AV suffix',
        'Exact protected video prefix'],BLUE,size=15)
    d.box(30,365,540,145,'Required performance evidence',[
        'Matched unprofiled complete requests',
        'Configuration and source identity',
        'Profile scopes explain overlap and waiting'],BLUE)
    d.box(630,365,540,145,'Required quality evidence',[
        'Equality where the contract is exact',
        'Paired media review for approximate changes',
        'Long-run speech / appearance / state evaluation'],GREEN)
    for x in (200,600,1000):
        d.arrow([(x,270),(x,317)])
    d.arrow([(200,317),(1000,317)],dashed=True)
    d.arrow([(300,317),(300,365)])
    d.arrow([(900,317),(900,365)])
    d.text(600,574,'A byte-equivalent optimization relative to MXFP8 does not make MXFP8 equivalent to BF16.',16,anchor='middle')
    d.text(600,613,'A successful HTTP request or a faster kernel does not establish complete-request benefit or action fidelity.',15,anchor='middle')
    d.save('fidelity-contracts')
    print('Built six vector method diagrams; all timelines are explicitly schematic.')


if __name__ == '__main__':
    main()

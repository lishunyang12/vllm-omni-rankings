"""Snapshot harness contracts and exact private renderer prompts for the paper."""
import ast
import hashlib
import json
from pathlib import Path

from build_diagrams import Diagram, BLUE, GREEN, ORANGE, GRAY

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
LAB=REPO/'experiments/food-tea-window-20260922'


def constant(path,name):
    tree=ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def main():
    examples=[]
    for ident,title,source,name in [
        ('chat-policy','Conversational voice and grounding policy','app/avatar_zhiwei_dialogue.py','DIALOGUE_SYSTEM'),
        ('singing-policy','Original singing and accompaniment policy','app/avatar_zhiwei_dialogue.py','SINGING_SYSTEM'),
        ('intent-extractor','Research planner: intent extraction','app/avatar_world/planner.py','EXTRACTOR'),
        ('action-mapper','Research planner: available-action mapping','app/avatar_world/planner.py','MAPPER'),
        ('grounded-writer','Research planner: bounded grounded writer','app/avatar_world/planner.py','WRITER')]:
        prompt=constant(REPO/source,name)
        examples.append(dict(id=ident,title=title,kind='Source policy template',source=source,
            scope='Public dialogue path' if ident in ('chat-policy','singing-policy') else 'Separate research planner',
            note='Verbatim source constant; runtime facts, language, history and viewer text are added separately. This is not a captured complete request.',
            prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
    cases=[('place-book','Place the book on the table','food_tea',2,'v3'),
           ('eat-candy','Take one bite and return the remainder','food_tea',3,'v3'),
           ('sip-tea','Sip from the existing cup and return it','food_tea',5,'v3'),
           ('pour-tea','Pour from the existing pot into the existing cup','pour_tea',3,'v3'),
           ('hold-state','Keep the resulting tabletop reading state','food_tea',6,'v3'),
           ('close-window','Experimental window closure','window',3,'window-v3'),
           ('hold-window','Experimental closed-window hold','window',4,'window-v3'),
           ('open-window','Experimental window reopening','window',6,'window-v3')]
    for ident,title,scenario,ordinal,review in cases:
        rows=json.loads((LAB/f'review-{review}'/scenario/'generation-evidence.json').read_text())
        row=next(r for r in rows if r['ordinal']==ordinal)
        source=Path(row['media']).with_name(f'request-{ordinal:04d}.json')
        wire=json.loads(source.read_text());extra=json.loads(wire['extra_params'])
        before=row['expected_before'];after=row['expected_after']
        examples.append(dict(id=ident,title=title,kind='Captured H3 request prompt',scope='Private GPU action research; not broadcast',
            note=('Window manipulation remains unqualified: other shutters may move or the closure may not persist.' if scenario=='window' else
                  'Finite private samples show the intended gross motion; no population action-success rate or automatic semantic verification is established.'),
            prompt=wire['prompt'],prompt_sha256=hashlib.sha256(wire['prompt'].encode()).hexdigest(),
            request_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_file=source.name,
            media_sha256=row['sha256'],reference_image=Path(row['reference_image']).name,
            seed=int(wire['seed']),reference_copies=row['reference_copies'],external_audio_references=row['external_audio_references'],
            geometry=dict(width=int(wire['width']),height=int(wire['height']),fps=int(wire['fps']),
                window_frames=extra['rolling_continuation']['window_frames'],overlap_frames=extra['rolling_continuation']['overlap_frames']),
            intended_state_delta={key:dict(before=before[key],after=after[key]) for key in before if before[key]!=after[key]},
            observed_state_committed=False))
    (ROOT/'evidence/prompt-examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    sources=['app/avatar_chat_inbox.py','app/avatar_text_worker.py','app/avatar_text.py','app/avatar_zhiwei_dialogue.py',
        'app/avatar_agent_replies.py','app/avatar_short_prompt.py','app/avatar_zhiwei_actions.py','app/avatar_continuous.py',
        'app/avatar_world/actions.py','app/avatar_world/planner.py','app/avatar_world/interaction_lab.py',
        'app/avatar_world/grounding.py','app/avatar_world/runtime.py','app/avatar_world/prompts.py',
        'app/benchmark_short_ref2va.py','scripts/probe_interaction_lab.py']
    manifest=dict(scope='Source inspection plus private action evidence; not a declaration of live deployment',
        broadcast=False,observed_world_updated=False,
        sources={name:dict(bytes=(REPO/name).stat().st_size,sha256=hashlib.sha256((REPO/name).read_bytes()).hexdigest()) for name in sources},
        routing=json.loads((LAB/'routing.json').read_text()),
        caveats=['Source policies and captured renderer requests have different provenance.',
                 'The research writer retains two-line/20-Chinese-character limits; the public dialogue path permits longer complete answers.',
                 'Photometric admission is not semantic action verification.',
                 'The first attempt to vary seeds was overridden by ContinuousGenerator; corrected trials record effective seeds.'])
    (ROOT/'evidence/harness-system.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')

    d=Diagram('Harness contracts: intention, evidence and playback',
        'The public reply queue and the separate research world ledger serve different deployment roles.',920)
    d.box(25,100,360,115,'Public production controller',[
        'Real chat → durable inbox → reply queue',
        'Whole-answer identity across windows'],BLUE)
    d.box(435,100,330,115,'Research action planner',[
        'Extract → map against available actions',
        'Snapshot state + context digest'],ORANGE,size=15)
    d.box(815,100,360,115,'Pure phase compiler',[
        'Before / after states + timed beats',
        'Predictions do not change memory'],ORANGE)
    d.arrow([(765,157),(815,157)])
    d.box(25,295,360,115,'H3 rendering contract',[
        'Reference image + accepted raw AV tail',
        'Exact words / once-only action prompt'],BLUE,size=15)
    d.arrow([(205,215),(205,295)])
    d.arrow([(995,215),(995,255),(205,255),(205,295)],dashed=True)
    d.text(590,245,'Private renderer adapter; not automatic public admission',14,anchor='middle')
    d.box(435,295,330,115,'Media and boundary checks',[
        'Decode · drift · AV overlap · seam',
        'Same-parent retry / withhold'],GREEN,size=15)
    d.arrow([(385,352),(435,352)])
    d.box(815,295,360,115,'Semantic action review',[
        'Explicit observed state + evidence',
        'Separate from pixel-quality checks'],ORANGE,size=15)
    d.arrow([(765,352),(815,352)],dashed=True)
    d.text(600,463,'Research receipt lifecycle',21,anchor='middle',weight='bold')
    for x,title,detail,color in [(25,'Prepared','Immutable phase request',BLUE),(325,'Generated','Media exists; world unchanged',BLUE),
                                  (625,'Verified','Review matches expected state',ORANGE),(925,'Played','Commit state and memory',GREEN)]:
        d.box(x,495,250,100,title,[detail],color,size=13)
    for x in (275,575,875):d.arrow([(x,545),(x+50,545)])
    d.box(35,690,535,120,'Reject, cancel or reconcile',[
        'Failed review → rollback; unknown playback → uncertain',
        'Cancel at a boundary, preserve completed effects',
        'A duplicated receipt must not repeat an action'],GRAY,size=15)
    d.box(625,690,535,120,'Scope of the guarantee',[
        'Ledger transitions can be checked deterministically',
        'They cannot make generated object motion correct',
        'A simulated observation cannot certify real footage'],GRAY,size=15)
    d.arrow([(750,595),(750,648),(300,648),(300,690)],dashed=True)
    d.text(600,868,'A requested action, generated candidate, accepted video and observed played effect are distinct records.',16,anchor='middle')
    d.save('harness-contracts')
    print(json.dumps(dict(prompts=len(examples),source_modules=len(sources),figures=1)))


if __name__=='__main__':main()

'use strict';
try {
const data=JSON.parse(document.getElementById('report-data').textContent);
const $=id=>document.getElementById(id), names=Object.fromEntries(data.modes.map(m=>[m.id,m.label]));
const tones={bf16:'#416773',pr4691:'#ae693a',pr4951:'#517853'};
const fmt=(v,n=3)=>Number.isFinite(Number(v))?Number(v).toFixed(n):String(v);
const signed=(v,n=2)=>(v>0?'+':'')+fmt(v,n);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const table=(id,headers,rows)=>{$(id).innerHTML='<thead><tr>'+headers.map(s=>'<th scope="col">'+esc(s)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(s=>'<td>'+esc(s)+'</td>').join('')+'</tr>').join('')+'</tbody>';};
const normal=m=>data.latencies.find(r=>r.mode===m&&r.measurement==='normal');
const dit=m=>data.stages.find(r=>r.mode===m&&r.stage==='DiT').seconds;
$('performance-conclusion').textContent=`相对 BF16，Sage 与本地修复版 Cake 的 DiT 耗时分别下降 ${fmt(100*(1-dit('pr4691')/dit('bf16')),2)}% 和 ${fmt(100*(1-dit('pr4951')/dit('bf16')),2)}%；本次实测未达到预期的 15% DiT 性能提升。`;
const traceVideoLink=m=>data.identities.some(r=>r.mode===m&&r.measurement==='nsys'&&!r.same_as_normal_video)?`<a href="${m}-nsys.mp4">Nsight 对应视频</a>`:'';
$('videos').innerHTML=data.modes.map(m=>`<article class="card" style="--tone:${tones[m.id]}"><div class="card-head"><h3>${esc(m.label)}</h3><span class="marker">${m.id==='bf16'?'参考':m.id==='pr4691'?'INT8 / FP8':'INT8 / FP8 · 已适配'}</span></div><video id="v-${m.id}" controls playsinline preload="metadata" poster="quality/${m.id}-poster.jpg" aria-label="${esc(m.label)} 茶屋视频"><source src="${m.id}.mp4" type="video/mp4">请下载视频播放。</video><div class="stats"><div><small>端到端 · 3 次均值</small><strong>${fmt(normal(m.id).mean_s)} <span>s</span></strong></div><div><small>DiT · 分阶段测量</small><strong>${fmt(dit(m.id))} <span>s</span></strong></div></div><div class="card-foot"><a href="${m.id}.mp4" download>下载完整视频</a><a href="${m.id}-warmed.nsys-rep">原始 Nsight</a>${traceVideoLink(m.id)}</div></article>`).join('');
table('latency-table',['方法','端到端均值 / s','中位数 / s','最小–最大 / s','DiT均值 / s'],data.modes.map(m=>{const r=normal(m.id);return[names[m.id],fmt(r.mean_s),fmt(r.median_s),fmt(r.min_s)+'–'+fmt(r.max_s),fmt(dit(m.id))];}));
const pairs=[['bf16','pr4691'],['pr4691','pr4951'],['bf16','pr4951']];
table('change-table',['变化方向','端到端耗时下降','DiT耗时下降'],pairs.map(([a,b])=>[names[a]+' → '+names[b],...['normal_e2e','dit_untraced'].map(k=>fmt(data.comparisons.find(r=>r.reference===a&&r.candidate===b&&r.measurement===k).latency_reduction_percent,2)+'%')]));
const stageColors={'Encoder':'#b7c6be','DiT':'#607d89','Decode':'#d6a16f','Other request time':'#d9dcd8'};
$('legend').innerHTML=Object.entries(stageColors).map(([k,c])=>`<span><i class="dot" style="background:${c}"></i>${esc(k)}</span>`).join('');
$('stage-bars').innerHTML=data.modes.map(m=>{const rows=data.stages.filter(r=>r.mode===m.id),total=rows.reduce((a,r)=>a+r.seconds,0);return`<div class="bar-row"><span>${esc(names[m.id])}</span><div class="bar">${rows.map(r=>`<span style="width:${r.share_percent}%;background:${stageColors[r.stage]}" title="${esc(r.stage)}: ${fmt(r.seconds)} s (${fmt(r.share_percent,2)}%)">${r.share_percent>8?fmt(r.share_percent,1)+'%':''}</span>`).join('')}</div><span class="bar-label">${fmt(total)} s</span></div>`;}).join('');
table('stage-table',['阶段',...data.modes.map(m=>names[m.id]+' 秒 / 占比')],Object.keys(stageColors).map(k=>[k,...data.modes.map(m=>{const r=data.stages.find(x=>x.mode===m.id&&x.stage===k);return fmt(r.seconds)+' / '+fmt(r.share_percent,2)+'%';})]));
table('quality-table',['对照','RGB PSNR / dB','RGB SSIM','YUV SSIM','音频 SNR / dB'],data.qualities.map(r=>[names[r.reference]+' ↔ '+names[r.candidate],fmt(r.rgb_psnr_db),fmt(r.rgb_ssim,6),fmt(r.yuv_ssim,6),fmt(r.audio_snr_db)]));
function phaseTable(){const[a,b]=$('phase-select').value.split(','),gpu=Number($('phase-gpu').value);table('phase-table',['类别',names[a]+' / s',names[b]+' / s','原占比','现占比','变化 / pp'],data.phase_changes.filter(r=>r.physical_gpu===gpu&&r.reference===a&&r.candidate===b).map(r=>[r.phase,fmt(r.reference_s),fmt(r.candidate_s),fmt(r.reference_share_percent,2)+'%',fmt(r.candidate_share_percent,2)+'%',signed(r.share_delta_pp)]));}
phaseTable();$('phase-select').addEventListener('change',phaseTable);$('phase-gpu').addEventListener('change',phaseTable);
table('kernel-table',['物理 GPU',...data.modes.map(m=>names[m.id]+' / ms'),'Sage → Cake 耗时下降'],Array.from({length:8},(_,gpu)=>[gpu,...data.modes.map(m=>fmt(data.kernels.find(r=>r.mode===m.id&&r.physical_gpu===gpu).ms_per_gpu_per_step,2)),fmt(data.kernel_changes.find(r=>r.reference==='pr4691'&&r.candidate==='pr4951'&&r.physical_gpu===gpu).latency_reduction_percent,2)+'%']));
table('clock-table',['物理 GPU',...data.modes.map(m=>names[m.id]+' max / mean')],Array.from({length:8},(_,gpu)=>[gpu,...data.modes.map(m=>{const r=data.clocks.find(x=>Number(x.physical_gpu)===gpu&&x.mode===m.id);return fmt(r.max_mhz,2)+' / '+fmt(r.mean_mhz,2);})]));
$('prompt').textContent=data.prompt;$('trace-links').innerHTML=data.modes.map(m=>`<a href="${m.id}-warmed.nsys-rep">${esc(m.label)} .nsys-rep</a>`).join('');
const videos=data.modes.map(m=>$('v-'+m.id));let playing=false,dragging=false;
function applyAudio(){videos.forEach((v,i)=>{v.muted=data.modes[i].id!==$('audio').value;});}
function pauseAll(){playing=false;videos.forEach(v=>v.pause());$('play').textContent='同步播放';}
function duration(){return Math.min(...videos.map(v=>Number.isFinite(v.duration)?v.duration:15.083333));}
function seekAll(t){t=Math.min(Math.max(0,t),duration());videos.forEach(v=>{if(v.readyState>0)v.currentTime=t;});$('seek').value=t;$('time').textContent=t.toFixed(2)+' / '+duration().toFixed(2)+' s';}
async function playAll(){applyAudio();seekAll(Number($('seek').value));try{await Promise.all(videos.map(v=>v.play()));playing=true;$('play').textContent='正在同步播放';$('status').textContent='同步播放中；切换音频可逐路听声音。';}catch(e){pauseAll();$('status').textContent='部分视频尚未就绪，请等待加载后重试，也可以单独播放或下载。';}}
$('play').addEventListener('click',playAll);$('pause').addEventListener('click',pauseAll);
$('audio').addEventListener('change',applyAudio);$('speed').addEventListener('change',()=>videos.forEach(v=>v.playbackRate=Number($('speed').value)));
$('seek').addEventListener('pointerdown',()=>{dragging=true;});$('seek').addEventListener('input',()=>seekAll(Number($('seek').value)));$('seek').addEventListener('change',()=>{dragging=false;});$('seek').addEventListener('pointerup',()=>{dragging=false;});
$('prev').addEventListener('click',()=>{pauseAll();seekAll(Number($('seek').value)-1/24);});$('next').addEventListener('click',()=>{pauseAll();seekAll(Number($('seek').value)+1/24);});
document.querySelectorAll('[data-time]').forEach(b=>b.addEventListener('click',()=>seekAll(Number(b.dataset.time))));
videos.forEach(v=>{v.addEventListener('loadedmetadata',()=>{$('seek').max=duration();});v.addEventListener('ended',pauseAll);v.addEventListener('error',()=>{$('status').textContent='视频加载失败，请使用该卡片的下载链接。';});});
function tick(){if(playing&&!dragging){const t=videos[0].currentTime;$('seek').value=t;$('time').textContent=t.toFixed(2)+' / '+duration().toFixed(2)+' s';videos.slice(1).forEach(v=>{if(v.readyState>=2&&Math.abs(v.currentTime-t)>.12)v.currentTime=t;});}requestAnimationFrame(tick);}applyAudio();tick();
}catch(error){const target=document.getElementById('load-error');target.style.display='block';target.textContent='页面数据未能加载：'+error.message+'。可从 README 和原始数据文件查看结果。';console.error(error);}

'use strict';
(() => {
  const $=id=>document.getElementById(id), rules=window.OmniTopics;
  const topicMap=new Map(rules.topics.map(t=>[t.id,t]));
  topicMap.set('unclassified',{id:'unclassified',name:'Unclassified',group:'other'});
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const safe=url=>/^https:\/\//.test(url||'')?esc(url):'#';
  const fmt=n=>new Intl.NumberFormat('en-US').format(n);
  const date=value=>new Date(value).toLocaleString('en-US',{year:'numeric',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
  const params=new URLSearchParams(location.search);
  const choice=(name,allowed,fallback)=>allowed.includes(params.get(name))?params.get(name):fallback;
  const state={
    data:null,prs:[],results:[],base:[],limit:40,
    selected:new Set((params.get('topics')||'').split(',').filter(id=>topicMap.has(id))),
    group:choice('group',['model','hardware','area'],'model'),
    match:choice('match',['all','any'],'all'),status:choice('status',['all','ready','draft'],'all'),
    sort:choice('sort',['updated','created','oldest'],'updated'),
    query:params.get('q')||'',exclude:params.get('exclude')==='committers',bots:params.get('bots')!=='0'
  };
  if(state.selected.has('unclassified'))state.selected=new Set(['unclassified']);
  const notify=(message='')=>{$('notice').hidden=!message;$('notice').textContent=message;};
  function syncURL() {
    const p=new URLSearchParams();
    if(state.selected.size)p.set('topics',[...state.selected].join(','));
    if(state.group!=='model')p.set('group',state.group);
    if(state.match!=='all')p.set('match',state.match);
    if(state.status!=='all')p.set('status',state.status);
    if(state.sort!=='updated')p.set('sort',state.sort);
    if(state.query)p.set('q',state.query);
    if(state.exclude)p.set('exclude','committers');
    if(!state.bots)p.set('bots','0');
    history.replaceState(null,'',location.pathname+(p.size?'?'+p:''));
  }
  function renderTopics() {
    const counts=new Map();
    state.base.forEach(p=>p.matches.forEach(m=>counts.set(m.id,(counts.get(m.id)||0)+1)));
    const query=$('topic-search').value.trim().toLowerCase();
    const topics=rules.topics.filter(t=>t.group===state.group && t.name.toLowerCase().includes(query) && (counts.get(t.id)||state.selected.has(t.id)))
      .sort((a,b)=>(counts.get(b.id)||0)-(counts.get(a.id)||0)||a.name.localeCompare(b.name));
    const max=Math.max(1,...topics.map(t=>counts.get(t.id)||0));
    $('topic-total').textContent=topics.length;
    $('group-hint').textContent=rules.groups.find(g=>g.id===state.group).hint;
    document.querySelectorAll('[data-group]').forEach(b=>{b.classList.toggle('active',b.dataset.group===state.group);b.setAttribute('aria-pressed',String(b.dataset.group===state.group));});
    $('topic-list').innerHTML=topics.map(t=>`<button class="topic-row" data-topic="${t.id}" aria-pressed="${state.selected.has(t.id)}"><span class="topic-name">${esc(t.name)}<span class="topic-bar"><span style="width:${(counts.get(t.id)||0)/max*100}%"></span></span></span><span class="topic-number">${fmt(counts.get(t.id)||0)}</span></button>`).join('')||'<div class="empty-state">No topics match these filters.</div>';
  }
  function renderPRs() {
    const visible=state.results.slice(0,state.limit);
    $('pr-list').innerHTML=visible.map(p=>`<article class="topic-pr-card" data-number="${p.number}">
      <div class="pr-meta"><a class="pr-number" href="${safe(p.url)}" target="_blank" rel="noopener noreferrer">#${p.number}</a><span class="badge ${p.draft?'draft':'open'}">${p.draft?'Draft':'Open'}</span><a class="pr-author" href="../#author=${encodeURIComponent(p.author)}">${esc(p.author)}</a>${p.committer?'<span class="committer-tag">Committer</span>':''}<time datetime="${esc(p.updated)}">Updated ${esc(date(p.updated))}</time></div>
      <a class="pr-title" href="${safe(p.url)}" target="_blank" rel="noopener noreferrer">${esc(p.title)} ↗</a>
      <div class="topic-tags">${p.matches.length?p.matches.map(m=>{const t=topicMap.get(m.id);return `<button class="topic-tag ${t.group}" data-topic="${t.id}" aria-pressed="${state.selected.has(t.id)}">${esc(t.name)}</button>`;}).join(''):'<button class="topic-tag" data-topic="unclassified">Unclassified</button>'}</div>
      <details class="evidence"><summary>Why these topics?</summary>${p.matches.length?`<ul>${p.matches.map(m=>`<li><strong>${esc(topicMap.get(m.id).name)}:</strong> ${m.evidence.map(esc).join(' · ')}</li>`).join('')}</ul>`:'<p>No title or label matched the current rules. This PR is still included in the full list.</p>'}</details>
    </article>`).join('')||'<div class="empty-state"><strong>No matching PRs</strong>Try Match any, remove a topic, or adjust your search.</div>';
    $('list-info').textContent=`Showing ${fmt(visible.length)} of ${fmt(state.results.length)} PRs`;
    $('load-more').hidden=visible.length>=state.results.length;
  }
  function render() {
    if(!state.data)return;
    const query=state.query.trim().toLowerCase().replace(/^#/,'');
    state.base=state.prs.filter(p=>(state.bots||!p.bot)&&(!state.exclude||!p.committer)&&
      (state.status!=='ready'||!p.draft)&&(state.status!=='draft'||p.draft)&&
      (!query||`${p.number} ${p.title} ${p.author} ${(p.labels||[]).join(' ')}`.toLowerCase().includes(query)));
    state.results=state.base.filter(p=>{
      if(!state.selected.size)return true;
      if(state.selected.has('unclassified'))return !p.matches.length;
      const ids=new Set(p.matches.map(m=>m.id));
      return [...state.selected][state.match==='all'?'every':'some'](id=>ids.has(id));
    }).sort((a,b)=>state.sort==='oldest'?a.created.localeCompare(b.created)||a.number-b.number:
      b[state.sort].localeCompare(a[state.sort])||b.number-a.number);
    $('stat-total').textContent=fmt(state.data.total);
    $('stat-matched').textContent=fmt(state.results.length);
    $('stat-authors').textContent=fmt(new Set(state.results.map(p=>p.author)).size);
    $('stat-unclassified').textContent=fmt(state.base.filter(p=>!p.matches.length).length);
    $('results-heading').textContent=state.selected.size?'Selected topics':'All open PRs';
    $('selected-topics').innerHTML=state.selected.size?[...state.selected].map(id=>`<button class="selected-chip" data-topic="${id}" aria-label="Remove ${esc(topicMap.get(id).name)}">${esc(topicMap.get(id).name)} ×</button>`).join(''):'<span class="small-note">Select topics to narrow the list.</span>';
    $('result-summary').textContent=`${fmt(state.results.length)} unique PRs · ${state.selected.size?`${state.selected.size} topic${state.selected.size===1?'':'s'} · match ${state.match}`:'no topic filter'}${state.exclude?' · committers excluded':''}`;
    document.querySelectorAll('[data-status]').forEach(b=>{b.classList.toggle('active',b.dataset.status===state.status);b.setAttribute('aria-pressed',String(b.dataset.status===state.status));});
    renderTopics();renderPRs();syncURL();
  }
  function toggleTopic(id) {
    if(!topicMap.has(id))return;
    if(state.selected.has(id))state.selected.delete(id);
    else {if(id==='unclassified')state.selected.clear();else state.selected.delete('unclassified');state.selected.add(id);}
    state.limit=40;render();
  }
  async function load() {
    $('refresh').disabled=true;notify();$('sync-label').textContent='Loading latest snapshot';
    try {
      const response=await fetch('../data.json?t='+Date.now(),{cache:'no-store',signal:AbortSignal.timeout(25000)});
      if(!response.ok)throw Error('HTTP '+response.status);
      const data=await response.json();
      if(data.schema!==1||!data.complete||!Array.isArray(data.pullRequests)||data.total!==data.pullRequests.length||
        new Set(data.pullRequests.map(p=>p.number)).size!==data.total||
        !Array.isArray(data.committers?.logins)||!Number.isFinite(Date.parse(data.updatedAt)))throw Error('Incomplete snapshot');
      const committers=new Set(data.committers.logins.map(login=>login.toLowerCase()));
      const prs=data.pullRequests.map(p=>({...p,matches:rules.classify(p),committer:committers.has(p.author.toLowerCase())}));
      state.data=data;state.prs=prs;render();
      $('snapshot-time').textContent=date(data.updatedAt);$('snapshot-time').dateTime=data.updatedAt;
      $('sync-label').textContent='Complete snapshot';
      $('coverage').textContent=`${data.pages} pages collected · ${fmt(data.total)} open PRs`;
      if(Date.now()-Date.parse(data.updatedAt)>3*3600000)notify('This snapshot is over three hours old. Updates may be delayed; check Update history.');
    } catch(error) {
      $('sync-label').textContent=state.data?'Previous complete snapshot':'Snapshot unavailable';
      notify(`Unable to load data: ${error.message}. ${state.data?'Showing the last complete snapshot.':'Try Refresh data again shortly.'}`);
    } finally {$('refresh').disabled=false;}
  }
  $('pr-search').value=state.query;$('match').value=state.match;$('sort').value=state.sort;
  $('exclude-committers').checked=state.exclude;$('bots').checked=state.bots;
  document.addEventListener('click',event=>{
    const topic=event.target.closest('button[data-topic]');if(topic)toggleTopic(topic.dataset.topic);
    const group=event.target.closest('[data-group]');if(group){state.group=group.dataset.group;$('topic-search').value='';if(state.data)renderTopics();syncURL();}
    const status=event.target.closest('[data-status]');if(status){state.status=status.dataset.status;state.limit=40;render();}
  });
  $('topic-search').addEventListener('input',()=>{if(state.data)renderTopics();});
  $('pr-search').addEventListener('input',()=>{state.query=$('pr-search').value;state.limit=40;render();});
  for(const [id,key] of [['match','match'],['sort','sort']])$(id).addEventListener('change',()=>{state[key]=$(id).value;state.limit=40;render();});
  for(const [id,key] of [['exclude-committers','exclude'],['bots','bots']])$(id).addEventListener('change',()=>{state[key]=$(id).checked;state.limit=40;render();});
  $('clear').addEventListener('click',()=>{state.selected.clear();state.limit=40;render();});
  $('view-unclassified').addEventListener('click',()=>{state.selected=new Set(['unclassified']);state.limit=40;render();});
  $('load-more').addEventListener('click',()=>{state.limit+=40;renderPRs();});
  $('refresh').addEventListener('click',load);
  $('export').addEventListener('click',()=>{
    if(!state.data)return;
    const cell=value=>'"'+String(value).replace(/^[=+@-]/,"'$&").replaceAll('"','""')+'"';
    const rows=[['number','author','title','draft','topics','created','updated','url','snapshot_at'],
      ...state.results.map(p=>[p.number,p.author,p.title,p.draft,p.matches.map(m=>topicMap.get(m.id).name).join('; ')||'Unclassified',p.created,p.updated,p.url,state.data.updatedAt])];
    const blob=new Blob(['\ufeff'+rows.map(row=>row.map(cell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});
    const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='vllm-omni-pr-topics.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  load();
})();

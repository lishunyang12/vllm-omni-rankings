'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const fmt = n => new Intl.NumberFormat('en-US').format(n);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const safeUrl = url => /^https:\/\//.test(url || '') ? esc(url) : '#';
  const state = {data:null, filter:'all', sort:'open', bots:true, excludeCommitters:false, query:'', selected:new URLSearchParams(location.hash.slice(1)).get('author'), limit:60, ranked:[], visible:[]};
  const age = date => {
    const minutes = Math.max(0, Math.floor((Date.now() - Date.parse(date)) / 60000));
    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`;
    return `${Math.floor(minutes / 1440)}d ago`;
  };
  const dateLabel = date => new Date(date).toLocaleString('en-US',{year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
  const avatar = (author, cls='') => `<img class="avatar ${cls}" src="${safeUrl(author.avatar)}" alt="" loading="lazy" referrerpolicy="no-referrer" width="32" height="32">`;
  const badge = (label, cls) => `<span class="badge ${cls}">${esc(label)}</span>`;
  const notify = (message='') => { $('notice').hidden = !message; $('notice').textContent = message; };

  function validate(data) {
    if (data.schema !== 1 || data.complete !== true || !Array.isArray(data.pullRequests) || data.total !== data.pullRequests.length || !Number.isFinite(Date.parse(data.updatedAt))) throw Error('Incomplete snapshot');
    if (new Set(data.pullRequests.map(p=>p.number)).size !== data.total) throw Error('Duplicate PRs in snapshot');
    if (new Set(data.pullRequests.map(p=>p.author)).size !== data.authors) throw Error('Contributor count mismatch');
  }

  function renderStats() {
    const {pullRequests:prs, updatedAt, total, authors, pages} = state.data;
    $('exclude-committers').disabled=!state.data.committers;
    $('committer-roster-count').textContent=state.data.committers?.logins.length??'—';
    const draft = prs.filter(p=>p.draft).length;
    const bots = new Set(prs.filter(p=>p.bot).map(p=>p.author)).size;
    $('stat-total').textContent=fmt(total);
    $('stat-authors').textContent=fmt(authors);
    $('stat-ready').textContent=fmt(total-draft);
    $('stat-draft').textContent=fmt(draft);
    $('stat-active').textContent=fmt(prs.filter(p=>Date.parse(updatedAt)-Date.parse(p.updated)<7*86400000).length);
    $('stat-humans').textContent=`${fmt(authors-bots)} people · ${bots} bot accounts`;
    $('ready-bar').style.width=`${total ? (total-draft)/total*100 : 0}%`;
    $('snapshot-time').textContent=dateLabel(updatedAt);
    $('snapshot-time').dateTime=updatedAt;
    $('sync-label').textContent=`Snapshot updated ${age(updatedAt)}`;
    if(state.data.recentSubmissions){const r=state.data.recentSubmissions;$('recent-window').textContent=`${dateLabel(r.since)} — ${dateLabel(r.until)} (your local time)`; }
    $('coverage').textContent=`${pages} pages collected · ${fmt(total)} open PRs`;
    const stale = Date.now()-Date.parse(updatedAt)>3*3600000;
    if (stale) notify(`Showing a complete snapshot from ${age(updatedAt)}. Updates may be delayed; see Update history below.`);
  }

  function rankAuthors() {
    if (!state.data) return;
    const groups = new Map();
    const committers=new Set((state.data.committers?.logins||[]).map(login=>login.toLowerCase()));
    for (const pr of state.data.pullRequests) {
      if ((state.excludeCommitters && committers.has(pr.author.toLowerCase())) || (!state.bots && pr.bot) || (state.filter==='ready' && pr.draft) || (state.filter==='draft' && !pr.draft)) continue;
      if (!groups.has(pr.author)) groups.set(pr.author,{login:pr.author, avatar:pr.avatar, url:pr.authorUrl, bot:pr.bot, recent:state.data.recentSubmissions ? (Object.hasOwn(state.data.recentSubmissions.byAuthor,pr.author) ? state.data.recentSubmissions.byAuthor[pr.author] : 0) : null, prs:[], ready:0, draft:0, updated:''});
      const a=groups.get(pr.author);
      a.prs.push(pr);
      if (pr.draft) a.draft++; else a.ready++;
      if (pr.updated>a.updated) a.updated=pr.updated;
    }
    const metric = a => state.sort==='updated' ? Date.parse(a.updated) : state.sort==='recent' ? (a.recent ?? 0) : a.prs.length;
    state.ranked = [...groups.values()].sort((a,b)=>metric(b)-metric(a) || a.login.localeCompare(b.login,'en',{sensitivity:'base'}));
    let previous, rank=0;
    state.ranked.forEach((a,i)=>{const value=metric(a);if(value!==previous) rank=i+1; a.rank=rank;previous=value;});
    state.visible=state.ranked.filter(a=>a.login.toLowerCase().includes(state.query.toLowerCase().trim()));
    if (!state.visible.some(a=>a.login===state.selected)) state.selected=state.visible[0]?.login || null;
    $('author-count').textContent=fmt(state.ranked.length);
    $('scope').textContent=`${fmt(state.ranked.reduce((s,a)=>s+a.prs.length,0))} open PRs · ${fmt(state.ranked.length)} contributors in scope`;
    renderAuthors();renderDetail();
  }

  function renderAuthors() {
    const authors=state.visible.slice(0,state.limit);
    const max=Math.max(1,...state.ranked.map(a=>a.prs.length));
    $('author-list').innerHTML=authors.map(a=>`<button class="author-row ${a.login===state.selected?'selected':''}" data-author="${esc(a.login)}" aria-pressed="${a.login===state.selected}" aria-label="Rank ${a.rank}, ${esc(a.login)}, ${a.prs.length} open PRs"><span class="rank ${a.rank<=3?'top-rank':''}">${String(a.rank).padStart(2,'0')}</span><span class="author-identity">${avatar(a)}<span><span class="login">${esc(a.login)}${a.bot?'<span class="bot-tag">BOT</span>':''}</span><span class="author-sub">${state.sort==='updated'?'Updated '+age(a.updated):`${a.ready} Non-draft · ${a.draft} Draft`}</span></span></span><span class="author-value">${fmt(a.prs.length)}<span class="mini-bar"><span style="width:${a.prs.length/max*100}%"></span></span></span><span class="recent-count" title="All PRs created in the last 14 days, including merged and closed">${a.recent===null?'—':fmt(a.recent)}</span><span class="draft-count ${a.draft?'':'zero'}">${a.draft||'—'}</span></button>`).join('') || '<div class="empty-state"><strong>No matching contributors</strong>Try another username or adjust the filters.</div>';
    $('list-info').textContent=`Showing ${fmt(authors.length)} of ${fmt(state.visible.length)} contributors${state.query?' · ranks preserved':''}`;
    $('load-more').hidden=state.limit>=state.visible.length;
  }

  function renderDetail() {
    const author=state.ranked.find(a=>a.login===state.selected);
    if (!author) {
      $('detail-heading').innerHTML='<div class="overline">PULL REQUESTS</div><h2>No contributor selected</h2>';
      $('pr-list').innerHTML='<div class="empty-state">Clear your search or adjust the filters to select a contributor.</div>';
      $('pr-info').textContent='0 PRs';$('author-github').hidden=true;return;
    }
    $('detail-heading').innerHTML=`<div class="overline">PULL REQUESTS</div><div class="profile-line">${avatar(author)}<div><h2><a href="${safeUrl(author.url)}" target="_blank" rel="noopener noreferrer">${esc(author.login)} ↗</a></h2><div class="profile-hint">Last updated ${age(author.updated)}</div></div><span class="profile-rank">RANK ${String(author.rank).padStart(2,'0')}</span></div><div class="profile-stats"><span><strong>${author.prs.length}</strong> Open PR</span><span><strong>${author.ready}</strong> Non-draft</span><span><strong>${author.draft}</strong> Draft</span><span title="Includes merged and closed PRs; unaffected by the Draft filter"><strong>${author.recent===null?'—':fmt(author.recent)}</strong> submitted in 14d</span></div>`;
    const query=$('pr-search').value.toLowerCase().trim();
    const prs=author.prs.filter(p=>{
      if(query && !`${p.number} ${p.title}`.toLowerCase().includes(query.replace(/^#/,''))) return false;
      return true;
    }).sort((a,b)=>b.updated.localeCompare(a.updated)||b.number-a.number);
    $('pr-list').innerHTML=prs.map(p=>`<article class="pr-card"><div class="pr-meta"><a href="${safeUrl(p.url)}" class="pr-number" target="_blank" rel="noopener noreferrer">#${p.number}</a>${badge(p.draft?'Draft':'Open',p.draft?'draft':'open')}<time datetime="${esc(p.updated)}" title="${esc(dateLabel(p.updated))}">Updated ${age(p.updated)}</time></div><a class="pr-title" href="${safeUrl(p.url)}" target="_blank" rel="noopener noreferrer">${esc(p.title)} <span aria-hidden="true">↗</span></a><div class="pr-created">Created ${esc(new Date(p.created).toLocaleDateString('en-US'))}</div></article>`).join('') || '<div class="empty-state"><strong>No matching PRs</strong>Try another PR title or number.</div>';
    $('pr-info').textContent=`${prs.length} / ${author.prs.length} PRs · most recently updated`;
    const search=`is:pr is:open author:${author.login}`;
    $('author-github').href=`https://github.com/vllm-project/vllm-omni/pulls?q=${encodeURIComponent(search)}`;
    $('author-github').hidden=false;
  }

  async function load() {
    $('refresh').disabled=true;
    $('sync-label').textContent='Loading latest snapshot';
    notify();
    try {
      const response=await fetch(`data.json?t=${Date.now()}`,{cache:'no-store',signal:AbortSignal.timeout(25000)});
      if(!response.ok)throw Error(`HTTP ${response.status}`);
      const data=await response.json();validate(data);state.data=data;
      renderStats();rankAuthors();
    } catch(error) {
      if(state.data)renderStats();else $('sync-label').textContent='Snapshot unavailable';
      notify(`Unable to load data: ${error.message}. ${state.data?'Showing the last complete snapshot.':'Try Refresh data again shortly.'}`);
    } finally { $('refresh').disabled=false; }
  }

  $('author-list').addEventListener('click',event=>{
    const row=event.target.closest('[data-author]');if(!row)return;
    state.selected=row.dataset.author;history.replaceState(null,'',`#author=${encodeURIComponent(state.selected)}`);
    $('pr-search').value='';
    renderAuthors();renderDetail();$('pr-list').scrollTop=0;
    if(matchMedia('(max-width:760px)').matches)$('detail').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth',block:'start'});
  });
  $('author-search').addEventListener('input',()=>{state.query=$('author-search').value;state.limit=60;rankAuthors();});
  document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{
    state.filter=button.dataset.filter;state.limit=60;
    document.querySelectorAll('[data-filter]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
    rankAuthors();
  }));
  $('sort').addEventListener('change',()=>{state.sort=$('sort').value;state.limit=60;rankAuthors();});
  $('bots').addEventListener('change',()=>{state.bots=$('bots').checked;state.limit=60;rankAuthors();});
  $('exclude-committers').addEventListener('change',()=>{state.excludeCommitters=$('exclude-committers').checked;state.limit=60;rankAuthors();});
  $('load-more').addEventListener('click',()=>{state.limit+=60;renderAuthors();});
  $('pr-search').addEventListener('input',renderDetail);
  $('refresh').addEventListener('click',load);
  $('export').addEventListener('click',()=>{
    if(!state.data)return;
    const cell=value=>'"'+String(value).replace(/^[=+@-]/,"'$&").replaceAll('"','""')+'"';
    const rows=[['rank','author','open_prs','submitted_14d_all_states','non_draft','draft','last_updated','snapshot_at','recent_since','recent_until'],...state.visible.map(a=>[a.rank,a.login,a.prs.length,a.recent??'',a.ready,a.draft,a.updated,state.data.updatedAt,state.data.recentSubmissions?.since??'',state.data.recentSubmissions?.until??''])];
    const blob=new Blob(['\ufeff'+rows.map(row=>row.map(cell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});
    const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='vllm-omni-open-pr-ranking.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  document.addEventListener('keydown',event=>{if(event.key==='/' && !['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)){event.preventDefault();$('author-search').focus();}});
  load();
})();

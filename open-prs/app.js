'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const fmt = n => new Intl.NumberFormat('en-US').format(n);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const safeUrl = url => /^https:\/\//.test(url || '') ? esc(url) : '#';
  const state = {data:null, filter:'all', sort:'open', bots:true, query:'', selected:new URLSearchParams(location.hash.slice(1)).get('author'), limit:60, ranked:[], visible:[]};
  const age = date => {
    const minutes = Math.max(0, Math.floor((Date.now() - Date.parse(date)) / 60000));
    if (minutes < 1) return '刚刚';
    if (minutes < 60) return `${minutes} 分钟前`;
    if (minutes < 1440) return `${Math.floor(minutes / 60)} 小时前`;
    return `${Math.floor(minutes / 1440)} 天前`;
  };
  const dateLabel = date => new Date(date).toLocaleString(undefined,{year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
  const avatar = (author, cls='') => `<img class="avatar ${cls}" src="${safeUrl(author.avatar)}" alt="" loading="lazy" referrerpolicy="no-referrer" width="32" height="32">`;
  const badge = (label, cls) => `<span class="badge ${cls}">${esc(label)}</span>`;
  const notify = (message='') => { $('notice').hidden = !message; $('notice').textContent = message; };

  function validate(data) {
    if (data.schema !== 1 || data.complete !== true || !Array.isArray(data.pullRequests) || data.total !== data.pullRequests.length || !Number.isFinite(Date.parse(data.updatedAt))) throw Error('快照不完整');
    if (new Set(data.pullRequests.map(p=>p.number)).size !== data.total) throw Error('快照存在重复 PR');
    if (new Set(data.pullRequests.map(p=>p.author)).size !== data.authors) throw Error('作者统计不一致');
  }

  function renderStats() {
    const {pullRequests:prs, updatedAt, total, authors, pages} = state.data;
    const draft = prs.filter(p=>p.draft).length;
    const bots = new Set(prs.filter(p=>p.bot).map(p=>p.author)).size;
    $('stat-total').textContent=fmt(total);
    $('stat-authors').textContent=fmt(authors);
    $('stat-ready').textContent=fmt(total-draft);
    $('stat-draft').textContent=fmt(draft);
    $('stat-active').textContent=fmt(prs.filter(p=>Date.parse(updatedAt)-Date.parse(p.updated)<7*86400000).length);
    $('stat-humans').textContent=`${fmt(authors-bots)} 位贡献者 · ${bots} 个 Bot 账号`;
    $('ready-bar').style.width=`${total ? (total-draft)/total*100 : 0}%`;
    $('snapshot-time').textContent=dateLabel(updatedAt);
    $('snapshot-time').dateTime=updatedAt;
    $('sync-label').textContent=`快照更新于 ${age(updatedAt)}`;
    $('coverage').textContent=`${pages} 页完整抓取 · ${fmt(total)} 个 PR`;
    const stale = Date.now()-Date.parse(updatedAt)>3*3600000;
    if (stale) notify(`当前显示 ${age(updatedAt)} 的完整快照。自动更新可能延迟，可通过页底「更新记录」检查任务状态。`);
  }

  function rankAuthors() {
    if (!state.data) return;
    const groups = new Map();
    for (const pr of state.data.pullRequests) {
      if ((!state.bots && pr.bot) || (state.filter==='ready' && pr.draft) || (state.filter==='draft' && !pr.draft)) continue;
      if (!groups.has(pr.author)) groups.set(pr.author,{login:pr.author, avatar:pr.avatar, url:pr.authorUrl, bot:pr.bot, prs:[], ready:0, draft:0, updated:''});
      const a=groups.get(pr.author);
      a.prs.push(pr);
      if (pr.draft) a.draft++; else a.ready++;
      if (pr.updated>a.updated) a.updated=pr.updated;
    }
    const metric = a => state.sort==='updated' ? Date.parse(a.updated) : a.prs.length;
    state.ranked = [...groups.values()].sort((a,b)=>metric(b)-metric(a) || a.login.localeCompare(b.login,'en',{sensitivity:'base'}));
    let previous, rank=0;
    state.ranked.forEach((a,i)=>{const value=metric(a);if(value!==previous) rank=i+1; a.rank=rank;previous=value;});
    state.visible=state.ranked.filter(a=>a.login.toLowerCase().includes(state.query.toLowerCase().trim()));
    if (!state.visible.some(a=>a.login===state.selected)) state.selected=state.visible[0]?.login || null;
    $('author-count').textContent=fmt(state.ranked.length);
    $('scope').textContent=`当前范围 ${fmt(state.ranked.reduce((s,a)=>s+a.prs.length,0))} 个 PR · ${fmt(state.ranked.length)} 位作者`;
    renderAuthors();renderDetail();
  }

  function renderAuthors() {
    const authors=state.visible.slice(0,state.limit);
    const max=Math.max(1,...state.ranked.map(a=>a.prs.length));
    $('author-list').innerHTML=authors.map(a=>`<button class="author-row ${a.login===state.selected?'selected':''}" data-author="${esc(a.login)}" aria-pressed="${a.login===state.selected}" aria-label="第 ${a.rank} 名 ${esc(a.login)}，${a.prs.length} 个 PR"><span class="rank ${a.rank<=3?'top-rank':''}">${String(a.rank).padStart(2,'0')}</span><span class="author-identity">${avatar(a)}<span><span class="login">${esc(a.login)}${a.bot?'<span class="bot-tag">BOT</span>':''}</span><span class="author-sub">${state.sort==='updated'?'更新于 '+age(a.updated):`${a.ready} 非 Draft · ${a.draft} Draft`}</span></span></span><span class="author-value">${fmt(a.prs.length)}<span class="mini-bar"><span style="width:${a.prs.length/max*100}%"></span></span></span><span class="draft-count ${a.draft?'':'zero'}">${a.draft||'—'}</span></button>`).join('') || '<div class="empty-state"><strong>没有匹配的贡献者</strong>试试其他用户名，或调整状态筛选。</div>';
    $('list-info').textContent=`显示 ${fmt(authors.length)} / ${fmt(state.visible.length)} 位作者${state.query?' · 保留原排名':''}`;
    $('load-more').hidden=state.limit>=state.visible.length;
  }

  function renderDetail() {
    const author=state.ranked.find(a=>a.login===state.selected);
    if (!author) {
      $('detail-heading').innerHTML='<div class="overline">PULL REQUESTS</div><h2>没有选中的贡献者</h2>';
      $('pr-list').innerHTML='<div class="empty-state">清除搜索或调整筛选后再选择一位作者。</div>';
      $('pr-info').textContent='0 个 PR';$('author-github').hidden=true;return;
    }
    $('detail-heading').innerHTML=`<div class="overline">PULL REQUESTS</div><div class="profile-line">${avatar(author)}<div><h2><a href="${safeUrl(author.url)}" target="_blank" rel="noopener noreferrer">${esc(author.login)} ↗</a></h2><div class="profile-hint">最近更新于 ${age(author.updated)}</div></div><span class="profile-rank">RANK ${String(author.rank).padStart(2,'0')}</span></div><div class="profile-stats"><span><strong>${author.prs.length}</strong> Open PR</span><span><strong>${author.ready}</strong> 非 Draft</span><span><strong>${author.draft}</strong> Draft</span></div>`;
    const query=$('pr-search').value.toLowerCase().trim();
    const prs=author.prs.filter(p=>{
      if(query && !`${p.number} ${p.title}`.toLowerCase().includes(query.replace(/^#/,''))) return false;
      return true;
    }).sort((a,b)=>b.updated.localeCompare(a.updated)||b.number-a.number);
    $('pr-list').innerHTML=prs.map(p=>`<article class="pr-card"><div class="pr-meta"><a href="${safeUrl(p.url)}" class="pr-number" target="_blank" rel="noopener noreferrer">#${p.number}</a>${badge(p.draft?'Draft':'Open',p.draft?'draft':'open')}<time datetime="${esc(p.updated)}" title="${esc(dateLabel(p.updated))}">${age(p.updated)}更新</time></div><a class="pr-title" href="${safeUrl(p.url)}" target="_blank" rel="noopener noreferrer">${esc(p.title)} <span aria-hidden="true">↗</span></a><div class="pr-created">创建于 ${esc(new Date(p.created).toLocaleDateString())}</div></article>`).join('') || '<div class="empty-state"><strong>没有符合条件的 PR</strong>试试其他 PR 标题或编号。</div>';
    $('pr-info').textContent=`${prs.length} / ${author.prs.length} 个 PR · 最近更新优先`;
    const search=`is:pr is:open author:${author.login}`;
    $('author-github').href=`https://github.com/vllm-project/vllm-omni/pulls?q=${encodeURIComponent(search)}`;
    $('author-github').hidden=false;
  }

  async function load() {
    $('refresh').disabled=true;
    $('sync-label').textContent='正在读取最新快照';
    notify();
    try {
      const response=await fetch(`data.json?t=${Date.now()}`,{cache:'no-store',signal:AbortSignal.timeout(25000)});
      if(!response.ok)throw Error(`HTTP ${response.status}`);
      const data=await response.json();validate(data);state.data=data;
      renderStats();rankAuthors();
    } catch(error) {
      if(state.data)renderStats();else $('sync-label').textContent='暂时无法读取快照';
      notify(`数据加载失败：${error.message}。${state.data?'仍显示上一次完整快照。':'请稍后点击「刷新数据」重试。'}`);
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
  $('load-more').addEventListener('click',()=>{state.limit+=60;renderAuthors();});
  $('pr-search').addEventListener('input',renderDetail);
  $('refresh').addEventListener('click',load);
  $('export').addEventListener('click',()=>{
    if(!state.data)return;
    const cell=value=>'"'+String(value).replace(/^[=+@-]/,"'$&").replaceAll('"','""')+'"';
    const rows=[['rank','author','open_prs','non_draft','draft','last_updated','snapshot_at'],...state.visible.map(a=>[a.rank,a.login,a.prs.length,a.ready,a.draft,a.updated,state.data.updatedAt])];
    const blob=new Blob(['\ufeff'+rows.map(row=>row.map(cell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});
    const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='vllm-omni-open-pr-ranking.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  document.addEventListener('keydown',event=>{if(event.key==='/' && !['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)){event.preventDefault();$('author-search').focus();}});
  load();
})();

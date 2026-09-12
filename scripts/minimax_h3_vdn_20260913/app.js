"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const grid = $("method-grid");
  const timeline = $("timeline");
  const videos = new Map();
  let data = null;
  let visibleIds = null;
  let playing = false;
  let buffering = false;
  let seekResume = false;
  let raf = 0;
  let toastTimer = 0;
  let activePlay = 0;
  let commonTime = 0;
  const goodNumber = (x) => typeof x === "number" && Number.isFinite(x) && x >= 0;
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  // Media/data files must stay below this Pages subdirectory. No local filesystem
  // paths, remote trackers, script URLs or protocol-relative URLs are accepted.
  function localPath(value) {
    return typeof value === "string" && /^[a-zA-Z0-9][a-zA-Z0-9_./-]*$/.test(value)
      && !value.split("/").includes("..") && !value.includes(":") ? value : null;
  }
  function showToast(message) {
    $("toast").textContent = message;
    $("toast").hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
  }
  function formatTime(seconds) {
    const ms = Math.max(0, Math.round((goodNumber(seconds) ? seconds : 0) * 1000));
    return `${String(Math.floor(ms / 60000)).padStart(2, "0")}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
  }
  function available() {
    return [...videos.entries()].filter(([id, video]) => (!visibleIds || visibleIds.has(id)) && !video.error && video.readyState >= 1);
  }
  function endpoint() {
    const durations = available().map(([, v]) => v.duration).filter((x) => goodNumber(x) && x > 0);
    return durations.length ? Math.min(...durations) : (data?.fixture?.planned_duration_s || 362 / 24);
  }
  function masterVideo() {
    const items = available();
    return items.find(([id]) => id === $("audio-source").value)?.[1] || items[0]?.[1] || null;
  }
  function setPlayLabel() {
    $("play-icon").textContent = playing ? "Ⅱ" : "▶";
    $("play-label").textContent = playing ? "同步暂停" : "同步播放";
    $("play").setAttribute("aria-label", playing ? "同步暂停" : "同步播放");
  }
  function applyAudio() {
    const selected = $("audio-source").value;
    for (const [id, video] of videos) video.muted = id !== selected || (visibleIds && !visibleIds.has(id));
  }
  function updateControls() {
    const expected = [...videos.entries()].filter(([id, v]) => (!visibleIds || visibleIds.has(id)) && !v.error).length;
    const enabled = available().length > 0 && available().length === expected;
    for (const id of ["play", "restart", "previous-frame", "next-frame", "timeline"]) $(id).disabled = !enabled;
    timeline.max = String(endpoint());
    $("time-duration").textContent = formatTime(endpoint());
    const chosenAudio = $("audio-source").value;
    $("audio-source").replaceChildren(new Option("全部静音", "muted"));
    for (const [id] of available()) {
      const method = data.methods.find((m) => m.id === id);
      $("audio-source").add(new Option(method.name, id));
    }
    if ([...$("audio-source").options].some((o) => o.value === chosenAudio)) $("audio-source").value = chosenAudio;
    applyAudio();
    if (!enabled) $("playback-status").textContent = expected ? "等待所有可用视频载入元数据…" : "视频生成完成后可同步播放。";
    else if (!playing) $("playback-status").textContent = `${available().length} 路视频可播放；音频仅播放所选一路。`;
  }
  function pauseAll() {
    activePlay += 1;
    playing = false;
    buffering = false;
    cancelAnimationFrame(raf);
    for (const video of videos.values()) video.pause();
    setPlayLabel();
  }
  function seekTo(seconds) {
    commonTime = Math.min(Math.max(0, seconds), endpoint());
    for (const [, video] of available()) {
      if (Math.abs(video.currentTime - commonTime) > .0001) video.currentTime = commonTime;
    }
    timeline.value = String(commonTime);
    $("time-current").textContent = formatTime(commonTime);
  }
  function tick() {
    if (!playing) return;
    const master = masterVideo();
    if (!master) { pauseAll(); return; }
    commonTime = master.currentTime;
    if (commonTime >= endpoint() - .015 || master.ended) {
      pauseAll(); seekTo(endpoint()); $("playback-status").textContent = "播放完成；可重新播放或逐帧检查。"; return;
    }
    if (!buffering) {
      const items = available();
      for (const [, video] of items) {
        if (video !== master && !video.seeking && Math.abs(video.currentTime - commonTime) > .08) video.currentTime = commonTime;
      }
      timeline.value = String(commonTime);
      $("time-current").textContent = formatTime(commonTime);
    }
    raf = requestAnimationFrame(tick);
  }
  async function playAll() {
    if (!available().length) return;
    cancelAnimationFrame(raf);
    const token = ++activePlay;
    if (commonTime >= endpoint() - .03) seekTo(0);
    const items = available();
    for (const [, video] of items) {
      if (Math.abs(video.currentTime - commonTime) > .04) video.currentTime = commonTime;
      video.playbackRate = Number($("playback-rate").value);
    }
    playing = true;
    buffering = false;
    applyAudio();
    setPlayLabel();
    try {
      await Promise.all(items.map(([, video]) => video.play()));
      if (token !== activePlay) return;
      $("playback-status").textContent = `同步播放 ${items.length} 路 · 浏览器按共同时间轴校正`;
      raf = requestAnimationFrame(tick);
    } catch (error) {
      if (token !== activePlay) return;
      pauseAll();
      $("playback-status").textContent = "播放被浏览器阻止或视频尚未就绪，请再次点击播放。";
    }
  }
  function onWaiting() {
    if (!playing || buffering) return;
    buffering = true;
    commonTime = masterVideo()?.currentTime || commonTime;
    for (const [, video] of available()) video.pause();
    $("playback-status").textContent = "正在缓冲；所有画面共同等待。";
  }
  function onCanPlay() {
    if (playing && buffering && available().every(([, video]) => video.readyState >= 3 && !video.seeking)) playAll();
  }
  function addStat(list, label, value) {
    list.append(el("dt", "", label), el("dd", "", value == null ? "待登记" : String(value)));
  }
  function renderMethod(method, index) {
    const card = el("article", "method-card");
    card.dataset.id = method.id;
    card.setAttribute("aria-labelledby", `title-${method.id}`);
    const heading = el("div", "card-heading");
    const name = el("div", "method-name");
    name.append(el("span", "method-index", String(index + 1).padStart(2, "0")));
    const nameBox = el("div");
    const title = el("h3", "", method.name); title.id = `title-${method.id}`;
    nameBox.append(title, el("div", "method-subtitle", method.subtitle || "")); name.append(nameBox);
    const labels = {pending:"待完成",running:"生成中",completed:"已完成",failed:"未完成",blocked:"资格待通过"};
    const status = el("span", `status-pill ${method.status === "completed" ? "complete" : method.status === "failed" ? "failed" : ""}`, labels[method.status] || "待完成");
    heading.append(name,status);card.append(heading);
    const stage = el("div", "video-stage");
    const videoPath = method.status === "completed" ? localPath(method.video) : null;
    if (videoPath) {
      const video = document.createElement("video");
      video.src = videoPath; video.preload = "metadata"; video.playsInline = true; video.muted = true;
      video.setAttribute("aria-label", `${method.name} 生成的视频`);
      videos.set(method.id, video); stage.append(video);
      video.addEventListener("loadedmetadata", () => {
        updateControls();
        if (goodNumber(method.artifact?.duration_s) && Math.abs(video.duration - method.artifact.duration_s) > .12) {
          $("playback-status").textContent = "有视频的容器时长与记录不同；同步播放以可用视频最短时长为界。";
        }
      });
      video.addEventListener("waiting", onWaiting);
      video.addEventListener("canplay", onCanPlay);
      video.addEventListener("seeked", onCanPlay);
      video.addEventListener("ended", () => { if (playing) { pauseAll(); seekTo(endpoint()); } });
      video.addEventListener("error", () => {
        pauseAll();updateControls();
        const note = el("div", "notice error", "视频暂时无法载入。请检查文件是否已发布，或使用下方下载链接。");stage.append(note);
        $("playback-status").textContent = "有视频载入失败；其余已载入视频仍可播放。";
      });
      const full = el("button", "fullscreen", "⛶");full.type="button";full.title="全屏查看此视频";full.setAttribute("aria-label",`${method.name} 全屏`);
      full.addEventListener("click", () => {
        if (stage.requestFullscreen) stage.requestFullscreen().catch(() => showToast("此浏览器暂不支持嵌入式全屏。"));
        else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
        else showToast("此浏览器暂不支持嵌入式全屏。");
      });stage.append(full);
    } else {
      const empty = el("div", "pending-state");
      empty.append(el("span", "pending-glyph", method.status === "running" ? "◷" : "—"),el("p", "", method.status_note || "生成结果尚未完成"),el("small", "", "NO GENERATED VIDEO PUBLISHED"));stage.append(empty);
    }
    card.append(stage);
    const content=el("div","card-content"),timing=method.timing || {},primary=el("div","primary-time"),valueBox=el("div");
    valueBox.append(el("div","time-label",timing.e2e_label || "完整生成耗时 · E2E"));
    const value=el("div",`time-value${goodNumber(timing.e2e_s)?"":" pending"}`,goodNumber(timing.e2e_s)?timing.e2e_s.toFixed(3):(method.status === "completed" ? "未记录" : "待测"));
    if(goodNumber(timing.e2e_s))value.append(el("small","","s"));valueBox.append(value);primary.append(valueBox);
    primary.append(el("div","time-samples",timing.samples ? `${timing.samples} 次实测\n${timing.statistic || "统计口径待登记"}` : "实测完成后公布"));content.append(primary);
    const details=el("dl","timing-details");
    for(const [label,key] of [["DiT / 去噪","dit_s"],goodNumber(timing.post_denoise_s)?["去噪后 / 解码、封装与同步","post_denoise_s"]:["VAE / 解码","vae_s"]]){
      const item=el("div");item.append(el("dt","",label),el("dd",goodNumber(timing[key])?"":"pending",goodNumber(timing[key])?`${timing[key].toFixed(3)} s`:(method.status === "completed" ? "未记录" : "待测")));details.append(item);
    }content.append(details);
    const params=el("dl","method-params");addStat(params,"模型",method.model);addStat(params,"执行次数",method.actual_nfe == null ? `${method.planned_nfe ?? "—"} NFE（计划）` : `${method.actual_nfe} NFE（实际）`);addStat(params,"精度",method.precision);addStat(params,"编译 / 暖机",timing.warmup);content.append(params);
    content.append(el("p","scope-note",timing.scope || "计时尚未完成。E2E 的起止边界将随实测记录公布。"));
    if(method.quality_notes?.length)content.append(el("p","quality-note",method.quality_notes.join(" · ")));
    if(videoPath){
      const links=el("div","artifact-links"),download=el("a","","下载完整视频 ↓");download.href=videoPath;download.download="";links.append(download);
      const receipt=localPath(method.receipt);if(receipt){const link=el("a","","实测记录 ↗");link.href=receipt;link.target="_blank";link.rel="noopener";links.append(link);}content.append(links);
      const a=method.artifact || {},bits=[];if(a.frames)bits.push(`${a.frames} frames`);if(goodNumber(a.duration_s))bits.push(`${a.duration_s.toFixed(3)} s`);if(a.width&&a.height)bits.push(`${a.width} × ${a.height}`);if(a.fps)bits.push(`${a.fps} fps`);
      if(bits.length)content.append(el("p","artifact-summary",bits.join(" / ")));
      if(a.sha256)content.append(el("p","artifact-summary",`SHA256 ${a.sha256}`));
    }
    card.append(content);return card;
  }
  function setView(value) {
    pauseAll();visibleIds=value==="all"?null:new Set(value.split(","));
    let count=0;for(const card of grid.children){card.hidden=visibleIds?!visibleIds.has(card.dataset.id):false;if(!card.hidden)count++;}
    grid.classList.toggle("two-up",count===2);grid.classList.toggle("four-up",count>=3);
    for(const button of document.querySelectorAll("[data-view]"))button.setAttribute("aria-pressed",String(button.dataset.view===value));
    updateControls();seekTo(commonTime);
  }
  $("play").addEventListener("click",()=>{if(playing)pauseAll();else playAll();});
  $("restart").addEventListener("click",()=>{pauseAll();seekTo(0);});
  $("previous-frame").addEventListener("click",()=>{pauseAll();seekTo(commonTime-1/(data?.fixture?.fps||24));});
  $("next-frame").addEventListener("click",()=>{pauseAll();seekTo(commonTime+1/(data?.fixture?.fps||24));});
  timeline.addEventListener("pointerdown",()=>{seekResume=playing;pauseAll();});
  timeline.addEventListener("input",()=>{if(playing){seekResume=true;pauseAll();}seekTo(Number(timeline.value));});
  timeline.addEventListener("change",()=>{if(seekResume){seekResume=false;playAll();}});
  $("audio-source").addEventListener("change",applyAudio);
  $("playback-rate").addEventListener("change",()=>{for(const v of videos.values())v.playbackRate=Number($("playback-rate").value);});
  document.querySelectorAll("[data-view]").forEach((b)=>b.addEventListener("click",()=>setView(b.dataset.view)));
  document.querySelectorAll("[data-time]").forEach((b)=>b.addEventListener("click",()=>{if(!available().length){showToast("观察点来自提示词，视频完成后可跳转查看。");return;}pauseAll();seekTo(Number(b.dataset.time));}));
  document.addEventListener("keydown",(e)=>{if(e.code!=="Space"||e.repeat||e.altKey||e.ctrlKey||e.metaKey||/INPUT|TEXTAREA|SELECT|BUTTON|A|SUMMARY/.test(e.target.tagName)||e.target.isContentEditable)return;if(!available().length)return;e.preventDefault();if(playing)pauseAll();else playAll();});
  document.addEventListener("visibilitychange",()=>{if(document.hidden)pauseAll();});
  async function init(){
    try{
      const response=await fetch("results.json",{cache:"no-store"});if(!response.ok)throw new Error("results.json 无法读取");data=await response.json();
      if(data.schema_version!==1||!Array.isArray(data.methods)||!data.methods.length)throw new Error("结果清单格式不正确");
      const ids=new Set();for(const m of data.methods){if(!/^[a-z0-9-]+$/.test(m.id)||ids.has(m.id))throw new Error("方法标识重复或无效");ids.add(m.id);}
      grid.replaceChildren(...data.methods.map(renderMethod));
      const done=data.methods.filter((m)=>m.status==="completed").length;
      $("completed-count").textContent=`${done} / ${data.methods.length} 已完成`;
      $("page-status").textContent=done===data.methods.length?"实测结果已更新":done?`${done} 项结果已完成`:"等待实验结果";
      if(data.updated_at)$("updated-at").textContent=`更新：${data.updated_at}`;
      const context=[data.experiment?.hardware_summary,data.experiment?.software_summary,data.experiment?.run_conditions,data.experiment?.timing_note].filter(Boolean);
      if(context.length)$("run-context").querySelector("p").textContent=context.join("\n");
      setView("all");
      const promptPath=localPath(data.fixture?.prompt)||"assets/prompt-15s.txt";
      try{const p=await fetch(promptPath);if(!p.ok)throw new Error();$("prompt-text").textContent=await p.text();}catch{$("prompt-text").textContent="提示词暂未载入，可点击上方链接下载。";}
    }catch(error){$("load-error").hidden=false;$("load-error").textContent=`无法加载实验清单。请通过网页服务器打开此目录，并确认 results.json 已发布。${error.message?"（"+error.message+"）":""}`;}
  }
  init();
})();

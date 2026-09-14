"use strict";
(() => {
  const videos = [document.getElementById("baseline"), document.getElementById("candidate")];
  const [master, follower] = videos;
  const playButton = document.getElementById("play-both");
  const playLabel = document.getElementById("play-label");
  const playIcon = document.getElementById("play-icon");
  const resetButton = document.getElementById("reset-both");
  const syncButton = document.getElementById("sync-both");
  const timeline = document.getElementById("timeline");
  const currentTime = document.getElementById("current-time");
  const durationLabel = document.getElementById("duration");
  const status = document.getElementById("playback-status");
  const audioSource = document.getElementById("audio-source");
  const speedControl = document.getElementById("playback-speed");
  const expectedPauses = new WeakSet();
  let desiredPlaying = false;
  let buffering = false;
  let seeking = false;
  let seekRevision = 0;
  let playRevision = 0;
  let internalRateChange = false;
  let mediaError = false;
  let lastStatus = "";
  let lastHardSync = 0;
  let sliderActive = false;

  function announce(message, error = false) {
    if (message !== lastStatus) {
      status.textContent = message;
      lastStatus = message;
    }
    status.classList.toggle("error", error);
  }
  function ready() {
    return !mediaError && videos.every(video => Number.isFinite(video.duration) && video.duration > 0);
  }
  function duration() {
    return ready() ? Math.min(...videos.map(video => video.duration)) : 362 / 24;
  }
  function formatTime(seconds) {
    const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
    const minutes = Math.floor(safe / 60);
    return `${String(minutes).padStart(2, "0")}:${(safe % 60).toFixed(2).padStart(5, "0")}`;
  }
  function updateControls() {
    playButton.disabled = resetButton.disabled = syncButton.disabled = timeline.disabled = !ready();
    playLabel.textContent = desiredPlaying ? "同时暂停" : "同时播放";
    playIcon.textContent = desiredPlaying ? "Ⅱ" : "▶";
    playButton.setAttribute("aria-pressed", String(desiredPlaying));
  }
  function pauseMedia(video) {
    if (!video.paused) {
      expectedPauses.add(video);
      video.pause();
    }
  }
  function pauseBoth(message = "已暂停 · 可拖动共同时间轴检查细节") {
    desiredPlaying = false;
    buffering = false;
    playRevision++;
    videos.forEach(pauseMedia);
    setRates(Number(speedControl.value));
    updateControls();
    announce(message);
  }
  function setRates(rate) {
    internalRateChange = true;
    videos.forEach(video => { if (video.playbackRate !== rate) video.playbackRate = rate; });
    internalRateChange = false;
  }
  function applyAudio() {
    videos.forEach(video => { video.muted = audioSource.value !== video.id; });
  }
  function waitForSeek(video, target) {
    if (Math.abs(video.currentTime - target) < 0.015 && !video.seeking) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const complete = () => { cleanup(); resolve(); };
      const failed = () => { cleanup(); reject(new Error("无法定位视频")); };
      const timeout = setTimeout(() => { cleanup(); reject(new Error("视频定位超时，请稍后重试")); }, 15000);
      function cleanup() {
        clearTimeout(timeout);
        video.removeEventListener("seeked", complete);
        video.removeEventListener("error", failed);
      }
      video.addEventListener("seeked", complete, { once: true });
      video.addEventListener("error", failed, { once: true });
      try { video.currentTime = target; } catch (error) { cleanup(); reject(error); }
    });
  }
  async function alignBoth(target, resume = desiredPlaying) {
    if (!ready()) return;
    const revision = ++seekRevision;
    playRevision++;
    seeking = true;
    buffering = false;
    desiredPlaying = resume;
    videos.forEach(pauseMedia);
    updateControls();
    const bounded = Math.max(0, Math.min(Number(target) || 0, duration()));
    announce("正在对齐两个视频…");
    const results = await Promise.allSettled(videos.map(video => waitForSeek(video, bounded)));
    if (revision !== seekRevision) return;
    seeking = false;
    if (results.some(result => result.status === "rejected")) {
      pauseBoth();
      announce("视频定位未完成，请等待加载后点击「重新对齐」，或下载原始 MP4。", true);
      return;
    }
    if (desiredPlaying) await playBoth();
    else announce("已对齐并暂停 · 可检查相同时间点的画面");
  }
  async function playBoth() {
    if (!ready() || seeking) return;
    if (videos.some(video => video.ended || video.currentTime >= duration() - 0.02)) {
      await alignBoth(0, true);
      return;
    }
    const revision = ++playRevision;
    desiredPlaying = true;
    buffering = false;
    applyAudio();
    setRates(Number(speedControl.value));
    updateControls();
    announce("正在开始同步播放…");
    // Call both play() methods before awaiting, preserving the browser's user gesture.
    const attempts = videos.map(video => {
      try { return Promise.resolve(video.play()); } catch (error) { return Promise.reject(error); }
    });
    const results = await Promise.allSettled(attempts);
    if (revision !== playRevision) return;
    if (results.some(result => result.status === "rejected")) {
      pauseBoth();
      announce("浏览器未允许同时播放，请再点击「同时播放」；也可先选择静音。", true);
    } else if (desiredPlaying && !buffering) {
      announce("同步播放中 · 拖动时间轴可检查同一时刻");
    }
  }
  function onBuffering(video) {
    if (!desiredPlaying || seeking || buffering || video.readyState >= 3) return;
    buffering = true;
    playRevision++;
    videos.forEach(pauseMedia);
    announce("正在缓冲，两个视频已一起暂停；加载后自动继续…");
  }
  function resumeAfterBuffering() {
    if (desiredPlaying && buffering && !seeking && videos.every(video => video.readyState >= 3)) {
      buffering = false;
      if (Math.abs(master.currentTime - follower.currentTime) > 0.08) void alignBoth(master.currentTime, true);
      else void playBoth();
    }
  }
  function onMetadata() {
    updateControls();
    if (ready()) {
      timeline.max = String(duration());
      durationLabel.textContent = formatTime(duration());
      if (!desiredPlaying && !seeking) announce("已就绪 · 点击「同时播放」，默认听优化后音频");
    }
  }

  playButton.addEventListener("click", () => {
    if (desiredPlaying) pauseBoth();
    else if (Math.abs(master.currentTime - follower.currentTime) > 0.08) void alignBoth(master.currentTime, true);
    else void playBoth();
  });
  resetButton.addEventListener("click", () => { void alignBoth(0, true); });
  syncButton.addEventListener("click", () => { void alignBoth(master.currentTime); });
  audioSource.addEventListener("change", applyAudio);
  speedControl.addEventListener("change", () => { setRates(Number(speedControl.value)); });
  timeline.addEventListener("pointerdown", () => { sliderActive = true; });
  timeline.addEventListener("input", () => { currentTime.textContent = formatTime(Number(timeline.value)); });
  timeline.addEventListener("change", () => { sliderActive = false; void alignBoth(Number(timeline.value)); });
  timeline.addEventListener("pointercancel", () => { sliderActive = false; });
  timeline.addEventListener("blur", () => { sliderActive = false; });

  videos.forEach(video => {
    video.addEventListener("loadedmetadata", onMetadata);
    video.addEventListener("durationchange", onMetadata);
    video.addEventListener("canplay", resumeAfterBuffering);
    video.addEventListener("waiting", () => { onBuffering(video); });
    video.addEventListener("stalled", () => { onBuffering(video); });
    video.addEventListener("play", () => {
      if (!desiredPlaying && !seeking) {
        desiredPlaying = true;
        if (Math.abs(master.currentTime - follower.currentTime) > 0.08) void alignBoth(video.currentTime, true);
        else void playBoth();
      }
    });
    video.addEventListener("pause", () => {
      if (expectedPauses.has(video)) { expectedPauses.delete(video); return; }
      if (desiredPlaying && !video.ended) pauseBoth();
    });
    video.addEventListener("seeking", () => {
      if (!seeking && performance.now() - lastHardSync > 250) void alignBoth(video.currentTime);
    });
    video.addEventListener("ended", () => { pauseBoth("播放完成 · 可从头播放或拖动时间轴复查"); });
    const onMediaError = () => {
      mediaError = true;
      pauseBoth();
      updateControls();
      announce("有视频加载失败，请刷新重试或使用对应的原始 MP4 下载链接。", true);
    };
    video.addEventListener("error", onMediaError);
    // A failed <source> can leave video.error unset and does not bubble.
    video.querySelectorAll("source").forEach(source => source.addEventListener("error", onMediaError));
    video.addEventListener("volumechange", () => {
      // Native controls may unmute a track. Keep one audible source at a time.
      if (!video.muted && audioSource.value !== video.id) {
        audioSource.value = video.id;
        applyAudio();
      }
    });
    video.addEventListener("ratechange", () => {
      const nominal = Number(speedControl.value);
      if (!internalRateChange && [0.25, 0.5, 1].includes(video.playbackRate) && Math.abs(video.playbackRate - nominal) > 0.001) {
        speedControl.value = String(video.playbackRate);
        setRates(video.playbackRate);
      }
    });
  });

  setInterval(() => {
    resumeAfterBuffering();
    if (ready() && !sliderActive && !seeking) {
      timeline.value = String(master.currentTime);
      currentTime.textContent = formatTime(master.currentTime);
    }
    if (!desiredPlaying || buffering || seeking || videos.some(video => video.paused || video.seeking)) return;
    const drift = follower.currentTime - master.currentTime;
    const nominal = Number(speedControl.value);
    if (Math.abs(drift) > 0.18 && performance.now() - lastHardSync > 1500) {
      lastHardSync = performance.now();
      // Seek both through the shared path so a slow fetch never leaves one side running.
      void alignBoth(master.currentTime, true);
    } else {
      const correction = Math.abs(drift) > 0.035 ? (drift > 0 ? 0.96 : 1.04) : 1;
      if (follower.playbackRate !== nominal * correction) follower.playbackRate = nominal * correction;
    }
  }, 100);
  applyAudio();
  onMetadata();
})();

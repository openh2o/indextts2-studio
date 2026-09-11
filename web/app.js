/* IndexTTS2 Studio frontend (v2) */
"use strict";

const $ = (id) => document.getElementById(id);

/* ---------------- theme ---------------- */
const themeBtn = $("themeBtn"), themeIcon = $("themeIcon");
const SUN = '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4m11.4-11.4 1.4-1.4"/>';
const MOON = '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>';

function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  themeIcon.innerHTML = t === "dark" ? SUN : MOON;
  drawHeaderWave();
  if (spkBuffer) drawSpkWave();
  if (trim.buffer) drawTrimWave();
}
themeBtn.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("theme", next);
  applyTheme(next);
});

/* ---------------- header waveform (signature) ---------------- */
const hWave = $("waveHeader"), hCtx = hWave.getContext("2d");
let genAnim = { on: false, level: 0, progress: 0 };
let waveT = 0;

function sizeCanvas(cv) {
  const r = cv.getBoundingClientRect();
  const d = window.devicePixelRatio || 1;
  const bw = Math.round(r.width * d), bh = Math.round(r.height * d);
  if (cv.width !== bw || cv.height !== bh) { cv.width = bw; cv.height = bh; }
  // scale the ctx so CSS-pixel drawing coords fill the HiDPI backing store;
  // canvas resize resets transforms, so reapply every time
  cv.getContext("2d").setTransform(bw / r.width || 1, 0, 0, bh / r.height || 1, 0, 0);
  return { w: r.width, h: r.height };
}

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function drawHeaderWave(t) {
  t = t || waveT || 0;
  const { w, h } = sizeCanvas(hWave);
  hCtx.clearRect(0, 0, w, h);
  const accent = cssVar("--accent"), primary = cssVar("--primary");
  const n = Math.floor(w / 4);
  const mid = h / 2;
  for (let i = 0; i < n; i++) {
    let a;
    if (genAnim.on) {
      const env = Math.sin(i / n * Math.PI);
      a = env * (0.25 + 0.75 * Math.abs(Math.sin(i * 0.7 + t / 130))) * genAnim.level;
    } else {
      a = 0.06 + 0.05 * Math.sin(i * 0.5 + t / 900);
    }
    const bh = Math.max(2, a * h);
    hCtx.fillStyle = genAnim.on && i / n < genAnim.progress ? primary : accent;
    hCtx.globalAlpha = genAnim.on ? 0.95 : 0.5;
    hCtx.fillRect(i * 4, mid - bh / 2, 2.4, bh);
  }
  hCtx.globalAlpha = 1;
}
setInterval(() => drawHeaderWave(waveT + 16), 250); // gentle idle refresh
let waveRAF = 0;
function waveLoop(t) {
  waveT = t;
  if (genAnim.on) { drawHeaderWave(t); waveRAF = requestAnimationFrame(waveLoop); }
  else { waveRAF = 0; drawHeaderWave(0); }
}

/* ---------------- system gauges (device / cpu / mem / gpu / vram) ---------------- */
function setMeter(id, pct, label) {
  const el = $(id);
  if (!el) return;
  const b = el.querySelector("b");
  if (label !== undefined && b) b.textContent = label;
  const bar = el.querySelector(".bar i");
  if (bar) bar.style.width = `${Math.min(pct ?? 0, 100)}%`;
  el.classList.toggle("hot", (pct ?? 0) > 85);
}
async function pollMetrics() {
  try {
    const m = await (await fetch("/metrics")).json();
    $("devText").textContent = m.gpu_name
      ? m.gpu_name.replace(/^NVIDIA /, "")
      : "CPU";
    if (m.cpu_percent !== undefined) setMeter("mCpu", m.cpu_percent, `${Math.round(m.cpu_percent)}%`);
    if (m.mem_total_gb) setMeter("mMem", (m.mem_used_gb / m.mem_total_gb) * 100, `${m.mem_used_gb.toFixed(1)}/${m.mem_total_gb.toFixed(1)}G`);
    const hasGpu = m.gpu_percent !== undefined;
    $("mGpu").hidden = !hasGpu;
    $("mVram").hidden = !hasGpu;
    if (hasGpu) {
      setMeter("mGpu", m.gpu_percent, `${m.gpu_percent}%`);
      setMeter("mVram", (m.vram_used_mb / (m.vram_total_mb || 8192)) * 100, `${(m.vram_used_mb / 1024).toFixed(1)}/${((m.vram_total_mb || 0) / 1024).toFixed(1)}G`);
    }
  } catch (e) { /* gauges are best-effort */ }
}

/* ---------------- model status polling ---------------- */
let modelReady = false;
async function pollStatus() {
  try {
    const s = await (await fetch("/model")).json();
    modelReady = !!s.loaded;
    $("modelPill").dataset.s = s.loaded ? "ready" : "unloaded";
    $("modelPillText").textContent = s.loaded
      ? "模型已加载"
      : (jobActive() ? "模型加载中…" : "模型未加载");
  } catch (e) {
    $("modelPill").dataset.s = "unloaded";
    $("modelPillText").textContent = "服务连接中断…";
  }
}
setInterval(pollStatus, 2000);
setInterval(pollMetrics, 2000);

/* ---------------- speaker reference audio ---------------- */
const spkDrop = $("spkDrop"), spkFile = $("spkFile");
const spkWave = $("spkWave"), spkCtx = spkWave.getContext("2d");
const spkAudio = new Audio();
const spkPlay = $("spkPlay"), spkPlayIcon = $("spkPlayIcon");
let spkBuffer = null, spkPlaying = false, spkSeekFrac = 0;
let spkUrl = null;

spkDrop.addEventListener("dragover", (e) => { e.preventDefault(); spkDrop.classList.add("over"); });
spkDrop.addEventListener("dragleave", () => spkDrop.classList.remove("over"));
spkDrop.addEventListener("drop", (e) => { e.preventDefault(); spkDrop.classList.remove("over"); if (e.dataTransfer.files[0]) setSpkFile(e.dataTransfer.files[0]); });
spkFile.addEventListener("change", () => spkFile.files[0] && setSpkFile(spkFile.files[0]));

/* emo reference audio shares the same pattern */
const emoDrop = $("emoDrop"), emoFile = $("emoFile");
emoDrop.addEventListener("dragover", (e) => { e.preventDefault(); emoDrop.classList.add("over"); });
emoDrop.addEventListener("dragleave", () => emoDrop.classList.remove("over"));
emoDrop.addEventListener("drop", (e) => { e.preventDefault(); emoDrop.classList.remove("over"); if (e.dataTransfer.files[0]) setEmoFile(e.dataTransfer.files[0]); });
emoFile.addEventListener("change", () => emoFile.files[0] && setEmoFile(emoFile.files[0]));
function setEmoFile(f) { $("emoName").textContent = f.name; }
function clearEmoFile() { emoFile.value = ""; $("emoName").textContent = "点击选择情感参考音频"; }

function drawSpkWave() {
  const { w, h } = sizeCanvas(spkWave);
  const data = spkBuffer.getChannelData(0);
  const step = Math.max(1, Math.floor(data.length / w));
  const mid = h / 2;
  const ink = cssVar("--accent"), played = cssVar("--primary"), border = cssVar("--border");
  spkCtx.clearRect(0, 0, w, h);
  spkCtx.fillStyle = border;
  spkCtx.fillRect(0, mid - 0.5, w, 1);
  const pos = spkPlaying || spkAudio.currentTime > 0 ? spkSeekFrac * w : 0;
  for (let x = 0; x < w; x++) {
    let min = 1, max = -1;
    for (let i = 0; i < step; i++) {
      const v = data[x * step + i] || 0;
      if (v < min) min = v; if (v > max) max = v;
    }
    const y0 = mid - max * mid * 0.9, y1 = mid - min * mid * 0.9;
    spkCtx.fillStyle = x <= pos ? played : ink;
    spkCtx.globalAlpha = x <= pos ? 0.95 : 0.55;
    spkCtx.fillRect(x, y0, 1, Math.max(1, y1 - y0));
  }
  spkCtx.globalAlpha = 1;
}
function spkFmt(t) { return `${t.toFixed(2)}s`; }
function spkTick() {
  if (!spkBuffer) return;
  const d = spkBuffer.duration;
  if (spkPlaying) spkSeekFrac = Math.min(spkAudio.currentTime / d, 1);
  $("spkTime").textContent = `${spkFmt(spkPlaying ? spkAudio.currentTime : spkSeekFrac * d)} / ${spkFmt(d)}`;
  drawSpkWave();
  if (spkPlaying) requestAnimationFrame(spkTick);
}
spkAudio.addEventListener("ended", spkStop);
function spkStop() {
  spkAudio.pause();
  spkPlaying = false;
  spkAudio.currentTime = 0;
  spkSeekFrac = 0;
  spkPlayIcon.innerHTML = PLAY_SVG;
  spkTick();
}
const PLAY_SVG = '<path d="M8 5v14l11-7z"/>';
const PAUSE_SVG = '<path d="M6 5h4v14H6zM14 5h4v14h-4z"/>';
spkPlay.addEventListener("click", async () => {
  if (!spkBuffer) return;
  if (spkPlaying) {
    spkAudio.pause();
    spkPlaying = false;
    spkPlayIcon.innerHTML = PLAY_SVG;
    spkTick();
  } else {
    spkAudio.currentTime = spkSeekFrac * spkBuffer.duration;
    await spkAudio.play().catch(() => {});
    spkPlaying = true;
    spkPlayIcon.innerHTML = PAUSE_SVG;
    requestAnimationFrame(spkTick);
  }
});
spkWave.addEventListener("pointerdown", (e) => {
  if (!spkBuffer) return;
  const r = spkWave.getBoundingClientRect();
  spkSeekFrac = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
  spkAudio.currentTime = spkSeekFrac * spkBuffer.duration;
  spkTick();
});

const AC = window.AudioContext || window.webkitAudioContext;
let sharedAC = null; // browsers cap AudioContext instances — reuse one
function getAC() {
  if (!sharedAC || sharedAC.state === "closed") sharedAC = new AC();
  return sharedAC;
}

function setSpkFile(f) {
  $("spkName").textContent = f.name;
  try {
    if (spkUrl) URL.revokeObjectURL(spkUrl);
    spkUrl = URL.createObjectURL(f);
    spkAudio.src = spkUrl;
    f.arrayBuffer().then((ab) => getAC().decodeAudioData(ab)).then((buf) => {
      spkBuffer = buf;
      spkPlaying = false; spkSeekFrac = 0;
      $("spkPlayerBox").hidden = false;
      $("spkSub").textContent = spkBuffer.duration >= 3 && spkBuffer.duration <= 10
        ? `时长 ${spkBuffer.duration.toFixed(1)}s`
        : `时长 ${spkBuffer.duration.toFixed(1)}s · 建议 3~10 秒`;
      spkTick();
    }).catch(() => {
      spkBuffer = null;
      $("spkPlayerBox").hidden = true;
      $("spkName").textContent = "点击选择或拖入音频文件";
      $("spkSub").textContent = "不支持该音频格式，请换一个文件（wav/mp3/flac/ogg）";
    });
  } catch (e) { /* ignore */ }
}

/* ---------------- emotion mode visibility ---------------- */
const EMO_NAMES = ["喜", "怒", "哀", "惧", "厌恶", "低落", "惊喜", "平静"];
const vecRow = $("vecs");
EMO_NAMES.forEach((n, i) => {
  const row = document.createElement("div");
  row.className = "vec-row";
  const name = document.createElement("span");
  name.className = "vec-name"; name.textContent = n;
  const s = document.createElement("input");
  s.type = "range"; s.min = 0; s.max = 1; s.step = 0.05; s.value = 0;
  s.dataset.i = i; s.title = n;
  const val = document.createElement("span");
  val.className = "vec-val"; val.textContent = "0.00";
  s.addEventListener("input", () => {
    val.textContent = Number(s.value).toFixed(2);
    s.style.setProperty("--fill", (s.value * 100) + "%");
  });
  row.append(name, s, val);
  vecRow.appendChild(row);
});
const vecInputs = () => document.querySelectorAll("#vecs input");
function setVecInputs(arr) {
  vecInputs().forEach((s) => {
    const v = arr[parseInt(s.dataset.i)] ?? 0;
    s.value = v;
    s.style.setProperty("--fill", (v * 100) + "%");
    const val = s.parentElement.querySelector(".vec-val");
    if (val) val.textContent = Number(v).toFixed(2);
  });
}
function emoVisible() {
  const m = $("emoMode").value;
  $("emoAudioBox").hidden = m != "1";
  $("emoVecBox").hidden = m != "2";
  $("emoTextBox").hidden = m != "3";
}
$("emoMode").addEventListener("change", emoVisible);
// 清空旧情感音频选择，避免从模式1切走再切回时把过期文件发上去
$("emoMode").addEventListener("change", () => { if ($("emoMode").value != "1") clearEmoFile(); });

/* ---------------- sliders ---------------- */
const SLIDERS = [
  { id: "emoWeight", out: "emoWeightOut", dec: 2 },
  { id: "temperature", out: "temperatureOut", dec: 2 },
  { id: "topP", out: "topPOut", dec: 2 },
  { id: "topK", out: "topKOut", dec: 0 },
  { id: "numBeams", out: "numBeamsOut", dec: 0 },
  { id: "maxMel", out: "maxMelOut", dec: 0 },
  { id: "segTokens", out: "segTokensOut", dec: 0 },
];
const DEFAULTS = { emoWeight: 0.65, temperature: 0.8, topP: 0.8, topK: 30, numBeams: 3, maxMel: 1500, segTokens: 120, repPen: 10, lenPen: 0, doSample: true };

function sliderFill(el) {
  const min = +el.min, max = +el.max, v = +el.value;
  el.style.setProperty("--fill", `${((v - min) / (max - min)) * 100}%`);
}
function bindSliders() {
  for (const s of SLIDERS) {
    const el = $(s.id), out = $(s.out);
    const upd = () => { out.textContent = Number(el.value).toFixed(s.dec); sliderFill(el); };
    el.addEventListener("input", upd);
    upd();
  }
}
$("resetParams").addEventListener("click", () => {
  for (const [k, v] of Object.entries(DEFAULTS)) {
    const el = $(k);
    if (el.type === "checkbox") el.checked = v;
    else el.value = v;
  }
  bindSliders();
  scheduleSegments();
});
$("segTokens").addEventListener("input", scheduleSegments);

/* ---------------- segments preview (collapsible, capped scroll) ---------------- */
const segList = $("segList");
let segTimer = null;
async function updateSegments() {
  const text = $("text").value;
  if (!text.trim()) {
    segList.replaceChildren();
    $("segInfo").textContent = "分句预览待输入";
    $("estDurEcho").textContent = "输入文本后这里会显示分句预览";
    return;
  }
  const fd = new FormData();
  fd.append("text", text);
  fd.append("max_text_tokens_per_segment", $("segTokens").value);
  try {
    const r = await fetch("/segments", { method: "POST", body: fd });
    const j = await r.json();
    const segs = j.segments || [];
    if (!segs.length) { segList.replaceChildren(); return; }
    const rows = segs.map((s) => {
      const row = document.createElement("div");
      row.className = "seg-item";
      const idx = document.createElement("span");
      idx.className = "seg-idx";
      idx.textContent = s.index;
      const text = document.createElement("span");
      text.className = "seg-text";
      // 后端返回 BPE 分词文本（▁ = 空格），展示时还原为正常空格便于阅读
      text.textContent = s.text.replace(/▁/g, " ");
      const tok = document.createElement("span");
      tok.className = "seg-tok";
      tok.textContent = `${s.tokens} tok`;
      row.append(idx, text, tok);
      return row;
    });
    segList.replaceChildren(...rows);
    const totalTok = segs.reduce((a, s) => a + s.tokens, 0);
    $("segInfo").textContent = `分句 ${segs.length} 段 · ${totalTok} tok`;
    $("estDurEcho").textContent = `共 ${segs.length} 段 · ${$("text").value.trim().length} 字`;
  } catch (err) { /* keep the last good preview */ }
}
function scheduleSegments() {
  clearTimeout(segTimer);
  segTimer = setTimeout(updateSegments, 300);
}
$("text").addEventListener("input", () => {
  $("charCount").textContent = `${$("text").value.trim().length} 字`;
  scheduleSegments();
});

// 文本框高度由 CSS 固定（#text 固定高 + 内部滚动），不随内容自动撑高

/* gen summary line: emotion mode + preset name (visible param confirmation above CTA) */
function updateGenSummary() {
  const modeNames = ["情感同参考音频", "情感参考音频", "情感向量", "情感描述文本"];
  const m = modeNames[$("emoMode").value] ?? $("emoMode").value;
  const parts = [
    `情感: ${m} (权重 ${Number($("emoWeight").value).toFixed(2)})`,
    curPresetName ? `预设: ${curPresetName}` : "预设: 无",
    `采样: temp ${Number($("temperature").value).toFixed(2)} / top_p ${Number($("topP").value).toFixed(2)} / top_k ${$("topK").value}`,
  ];
  $("genSummary").textContent = parts.join("  ·  ");
}
["emoMode", "emoWeight", "temperature", "topP", "topK"].forEach((id) => {
  $(id).addEventListener("input", updateGenSummary);
  $(id).addEventListener("change", updateGenSummary);
});

/* ---------------- generation ---------------- */
const genBtn = $("genBtn"), errBox = $("errBox");
const emptyHint = $("emptyHint"), loadingBox = $("loadingBox"), resultBody = $("resultBody");
let elapsedTimer = null, genStart = 0;
let hasResult = false;
function jobActive() { return genBtn.disabled; }

function setResultVisible(visible) {
  resultBody.style.display = visible ? "block" : "none";
  emptyHint.style.display = visible ? "none" : "";
}

function showLoading(stage) {
  setResultVisible(false);
  loadingBox.hidden = false;
  $("pStage").textContent = stage;
  errBox.classList.remove("on");
}

const PROG_DESC = {
  "starting inference...": "启动推理",
  "text processing...": "文本分词与分段",
  "saving audio...": "解码合成音频",
};
function transDesc(d) {
  if (!d) return "";
  // 扩散步数标记：desc 形如 "speech synthesis 1/2...|s2mel|12/25"
  const mDiff = /\|s2mel\|(\d+)\/(\d+)/.exec(d);
  if (mDiff) {
    const mSeg = /speech synthesis (\d+)\/(\d+)/.exec(d);
    const segText = mSeg ? `语音合成 ${mSeg[1]}/${mSeg[2]} 段` : "";
    return `${segText ? segText + " · " : ""}扩散 ${mDiff[1]}/${mDiff[2]}`;
  }
  if (d.includes("|")) d = d.split("|")[0];
  if (PROG_DESC[d]) return PROG_DESC[d];
  const m = /speech synthesis (\d+)\/(\d+)\.\.\./.exec(d);
  if (m) return `语音合成 ${m[1]}/${m[2]} 段`;
  return d;
}

const stepEls = Array.from(document.querySelectorAll(".pstep"));
const stepState = [0, 0, 0, 0, 0];
function markStep(i, s) {
  const rank = { waiting: 0, running: 1, done: 2 };
  if (s === "running" && stepState[i] === 2) stepState[i] = 1;
  else if (rank[s] <= stepState[i]) return;
  else stepState[i] = rank[s];
  const el = stepEls[i];
  if (el) { el.classList.remove("waiting", "running", "done"); el.classList.add(s); }
}
function resetSteps() {
  for (let i = 0; i < 5; i++) { stepState[i] = 0; stepEls[i].classList.remove("running", "done"); }
}

function startElapsed() {
  genAnim.on = true; genAnim.level = 0.5; genAnim.progress = 0;
  cancelAnimationFrame(waveRAF);
  waveRAF = requestAnimationFrame(waveLoop);
  clearInterval(elapsedTimer);
  elapsedTimer = setInterval(() => {
    $("pElapsed").textContent = `${((performance.now() - genStart) / 1000).toFixed(1)}s`;
  }, 100);
}
function stopUi() {
  genBtn.disabled = false;
  clearInterval(elapsedTimer);
  cancelAnimationFrame(waveRAF);
  waveRAF = 0;
  genAnim.on = false;
  drawHeaderWave(0);
}
function showError(msg) {
  errBox.textContent = msg;
  errBox.classList.add("on");
  loadingBox.hidden = true;
  setResultVisible(hasResult);
  stopUi();
}

function showActionError(msg) {
  const hint = $("loadHint");
  hint.textContent = msg;
  hint.classList.add("error");
}

function showActionHint(msg) {
  const hint = $("loadHint");
  hint.textContent = msg;
  hint.classList.remove("error");
}

function clearActionError() {
  const hint = $("loadHint");
  hint.textContent = "";
  hint.classList.remove("error");
}

genBtn.addEventListener("click", async () => {
  const text = $("text").value.trim();
  if (!text) { showError("请先输入要合成的文本"); return; }
  const spkSelected = $("spkFile").files[0];
  if (!spkSelected) { showError("请先选择音色参考音频"); return; }
  const mode = $("emoMode").value;
  if (mode == "1" && !$("emoFile").files[0]) { showError("情感参考音频模式下请先选择情感音频"); return; }
  if (mode == "3" && !$("emoText").value.trim()) { showError("请输入情感描述文本"); return; }

  genBtn.disabled = true;
  genStart = performance.now();
  player.pause();
  bigPlayIcon.innerHTML = PLAY2;
  clearActionError();
  setResultVisible(false);
  errBox.classList.remove("on");
  showLoading(modelReady ? "准备生成…" : "首次生成，正在加载模型…（约 30~60 秒）");
  resetSteps();
  $("pbarFill").style.width = "0%";
  $("pStageMeta").textContent = "";
  startElapsed();

  const fd = new FormData();
  fd.append("spk_audio", spkSelected);
  fd.append("text", $("text").value);
  fd.append("emo_mode", mode);
  fd.append("emo_weight", $("emoWeight").value);
  fd.append("emo_text", $("emoText").value);
  fd.append("use_random", $("emoRandom").checked);
  const ea = $("emoFile").files[0];
  if (mode == "1" && ea) fd.append("emo_audio", ea);
  if (mode == "2") vecInputs().forEach((s) => fd.append("vec" + (parseInt(s.dataset.i) + 1), s.value));
  fd.append("max_text_tokens_per_segment", $("segTokens").value);
  fd.append("do_sample", $("doSample").checked);
  fd.append("top_p", $("topP").value);
  fd.append("top_k", $("topK").value);
  fd.append("temperature", $("temperature").value);
  fd.append("length_penalty", $("lenPen").value);
  fd.append("num_beams", $("numBeams").value);
  fd.append("repetition_penalty", $("repPen").value);
  fd.append("max_mel_tokens", $("maxMel").value);
  fd.append("file_naming", "title_time");

  try {
    const r = await fetch("/tts", { method: "POST", body: fd });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.detail || `请求失败（${r.status}）`);
    }
    // consume the SSE stream manually — EventSource can't POST form data
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let loadingModel = false;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const dl = block.split("\n").find((l) => l.startsWith("data:"));
        if (!dl) continue;
        let ev;
        try { ev = JSON.parse(dl.slice(5).trim()); } catch (e) { continue; }
        if (ev.type === "load") {
          loadingModel = true;
          markStep(0, "running");
          showLoading("正在加载模型…首次调用需 30~60 秒");
          $("pbarFill").style.width = "8%";
        } else if (ev.type === "progress") {
          if (loadingModel) { loadingModel = false; $("pbarFill").style.width = "0%"; }
          markStep(0, "done");
          const pStage = String(ev.desc || "").split("|")[1] || "";
          if (pStage === "ref" || (ev.value || 0) < 0.1) markStep(1, "running");
          if (pStage !== "ref" && (ev.value || 0) >= 0.1) markStep(1, "done");
          if (pStage === "gpt") markStep(2, "running");
          if (pStage === "s2mel") { markStep(2, "done"); markStep(3, "running"); }
          if (pStage === "bigvgan") { markStep(3, "done"); markStep(4, "running"); }
          showLoading(transDesc(ev.desc) || "正在生成语音…");
          genAnim.level = Math.min(1, 0.4 + (ev.value || 0));
          genAnim.progress = ev.value || 0;
          $("pbarFill").style.width = `${Math.round((ev.value || 0) * 100)}%`;
          // s2mel 阶段优先显示扩散步数，其余阶段显示全局百分比
          const mDiff = /\|s2mel\|(\d+)\/(\d+)/.exec(String(ev.desc || ""));
          $("pStageMeta").textContent = mDiff
            ? `扩散 ${mDiff[1]}/${mDiff[2]}`
            : `${Math.round((ev.value || 0) * 100)}%`;
        } else if (ev.type === "done") {
          for (let i = 0; i < 5; i++) markStep(i, "done");
          $("pbarFill").style.width = "100%";
          loadingBox.hidden = true;
          stopUi();
          histCurId = ev.history_id || "";
          await loadResult(ev.wav, ev.elapsed);
          await refreshHistory();
          pollStatus();
        } else if (ev.type === "error") {
          throw new Error(ev.detail || "生成失败");
        }
      }
    }
    if (genBtn.disabled) showError("生成连接中断，请重试");
  } catch (e) {
    showError(e.message);
    pollStatus();
  }
});

/* ---------------- result, trim & download ---------------- */
const trim = { buffer: null, sel: null, drag: null };
const tWave = $("waveTrim"), tCtx = tWave.getContext("2d");
const player = $("player");
const bigPlay = $("bigPlay"), bigPlayIcon = $("bigPlayIcon");
const PLAY2 = '<path d="M8 5v14l11-7z"/>';
const PAUSE2 = '<path d="M6 5h4v14H6zM14 5h4v14h-4z"/>';
let curUrl = null;

async function loadResult(url, elapsed) {
  curUrl = url;
  loadingBox.hidden = true;
  setResultVisible(true);
  clearActionError();
  player.src = url;
  try {
    const buf = await (await fetch(url)).arrayBuffer();
    trim.buffer = await getAC().decodeAudioData(buf);
  } catch (e) {
    trim.buffer = null;
    hasResult = false;
    player.pause();
    player.removeAttribute("src");
    player.load();
    setResultVisible(false);
    showError("生成结果音频加载失败，请重试");
    return;
  }
  trim.sel = null;
  hasResult = true;
  $("trimRange").textContent = `全长 ${trim.buffer.duration.toFixed(2)}s`;
  $("genMeta").textContent = `耗时 ${Number(elapsed || 0).toFixed(1)}s`;
  updateDlName();
  drawTrimWave();
}

bigPlay.addEventListener("click", () => {
  if (player.paused) { player.play(); bigPlayIcon.innerHTML = PAUSE2; }
  else { player.pause(); bigPlayIcon.innerHTML = PLAY2; }
});
player.addEventListener("ended", () => { bigPlayIcon.innerHTML = PLAY2; });

function drawTrimWave() {
  const { w, h } = sizeCanvas(tWave);
  const data = trim.buffer.getChannelData(0);
  const step = Math.floor(data.length / w) || 1;
  const mid = h / 2;
  const ink = cssVar("--accent"), mut = cssVar("--border"), pri = cssVar("--primary");
  const pos = player.duration ? (player.currentTime / player.duration) * w : 0;
  const s = trim.sel;
  tCtx.clearRect(0, 0, w, h);
  for (let x = 0; x < w; x++) {
    let min = 1, max = -1;
    for (let i = 0; i < step; i++) {
      const v = data[x * step + i] || 0;
      if (v < min) min = v; if (v > max) max = v;
    }
    const y0 = mid - max * mid * 0.92, y1 = mid - min * mid * 0.92;
    const inSel = !s || (x / w >= s[0] && x / w <= s[1]);
    tCtx.fillStyle = x <= pos ? pri : (inSel ? ink : mut);
    tCtx.globalAlpha = x <= pos ? 0.95 : (inSel ? 0.9 : 0.55);
    tCtx.fillRect(x, y0, 1, Math.max(1, y1 - y0));
  }
  tCtx.globalAlpha = 1;
  if (s) {
    tCtx.fillStyle = pri;
    tCtx.fillRect(s[0] * w - 1, 0, 2, h);
    tCtx.fillRect(s[1] * w - 1, 0, 2, h);
    tCtx.globalAlpha = 0.08;
    tCtx.fillRect(s[0] * w, 0, (s[1] - s[0]) * w, h);
    tCtx.globalAlpha = 1;
  }
}
player.addEventListener("timeupdate", () => { if (trim.buffer) drawTrimWave(); });

function xToFrac(e) {
  const r = tWave.getBoundingClientRect();
  return Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
}
tWave.addEventListener("pointerdown", (e) => {
  if (!trim.buffer) return;
  trim.drag = [xToFrac(e), xToFrac(e)];
  tWave.setPointerCapture(e.pointerId);
});
tWave.addEventListener("pointermove", (e) => {
  if (!trim.drag) return;
  trim.drag[1] = xToFrac(e);
  trim.sel = [Math.min(...trim.drag), Math.max(...trim.drag)];
  drawTrimWave();
  const d = trim.buffer.duration;
  $("trimRange").textContent = `选中 ${(trim.sel[0] * d).toFixed(2)}s – ${(trim.sel[1] * d).toFixed(2)}s`;
});
tWave.addEventListener("pointerup", () => { trim.drag = null; });

let selStopTick = null;
$("playSel").addEventListener("click", () => {
  if (!trim.buffer) return;
  if (!player.paused) player.pause();
  const d = trim.buffer.duration;
  if (selStopTick) { player.removeEventListener("timeupdate", selStopTick); selStopTick = null; }
  player.currentTime = (trim.sel ? trim.sel[0] : 0) * d;
  player.play();
  bigPlayIcon.innerHTML = PAUSE2;
  if (trim.sel) {
    const stopAt = trim.sel[1] * d;
    selStopTick = () => { if (player.currentTime >= stopAt) { player.pause(); player.removeEventListener("timeupdate", selStopTick); selStopTick = null; bigPlayIcon.innerHTML = PLAY2; } };
    player.addEventListener("timeupdate", selStopTick);
  }
});
$("resetSel").addEventListener("click", () => {
  trim.sel = null;
  $("trimRange").textContent = `全长 ${trim.buffer ? trim.buffer.duration.toFixed(2) : "—"}s`;
  drawTrimWave();
});

player.addEventListener("play", () => { bigPlayIcon.innerHTML = PAUSE2; });
player.addEventListener("pause", () => { bigPlayIcon.innerHTML = PLAY2; });

/* ---- WAV export & download: mirrors the server-side naming rules ---- */
function encodeWav(buffer, s0, s1) {
  const sr = buffer.sampleRate;
  const from = Math.floor((s0 || 0) * buffer.length);
  const to = Math.floor((s1 ?? 1) * buffer.length);
  const n = Math.max(to - from, 1);
  const ch = buffer.numberOfChannels;
  const out = new DataView(new ArrayBuffer(44 + n * ch * 2));
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) out.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); out.setUint32(4, 36 + n * ch * 2, true); ws(8, "WAVE"); ws(12, "fmt ");
  out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, ch, true);
  out.setUint32(24, sr, true); out.setUint32(28, sr * ch * 2, true);
  out.setUint16(32, ch * 2, true); out.setUint16(34, 16, true);
  ws(36, "data"); out.setUint32(40, n * ch * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < ch; c++) {
      const v = Math.max(-1, Math.min(1, buffer.getChannelData(c)[from + i] || 0));
      out.setInt16(o, v < 0 ? v * 0x8000 : v * 0x7fff, true); o += 2;
    }
  }
  return new Blob([out], { type: "audio/wav" });
}

function safeTitle(text, maxlen = 15) {
  let t = String(text || "").replace(/[\r\n\t]+/g, " ").trim();
  t = t.replace(/[\\/:*?"<>|]+/g, " ").replace(/\s+/g, " ").trim().replace(/^[\s.]+|[\s.]+$/g, "");
  return t ? t.slice(0, maxlen) : "";
}
function updateDlName() {
  const title = safeTitle($("text").value);
  const t = new Date();
  const pad = (x) => String(x).padStart(2, "0");
  const stamp = `${t.getFullYear()}${pad(t.getMonth() + 1)}${pad(t.getDate())}_${pad(t.getHours())}${pad(t.getMinutes())}${pad(t.getSeconds())}`;
  $("dlName").textContent = title ? `${title}_${stamp}.wav` : `spk_${Math.floor(t.getTime() / 1000)}.wav`;
}
$("text").addEventListener("input", () => { if (trim.buffer) updateDlName(); });

$("dlBtn").addEventListener("click", () => {
  if (!trim.buffer) return;
  const blob = encodeWav(trim.buffer, trim.sel ? trim.sel[0] : 0, trim.sel ? trim.sel[1] : undefined);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = $("dlName").textContent;
  a.click();
  URL.revokeObjectURL(a.href);
});

/* ---------------- preset library (right column, tabs with examples) ---------------- */
let presetNamesCache = [];
let curPresetName = "";       // applied preset name (for summary line + card highlight)
let exCache = [];             // example items cache

async function fetchPresetAudio(url, inputEl, applyFn) {
  try {
    const r = await fetch(url);
    if (!r.ok) return;
    const blob = await r.blob();
    const fileName = (url.split("/").pop() || "audio") + ".wav";
    const file = new File([blob], fileName, { type: "audio/wav" });
    const dt = new DataTransfer();
    dt.items.add(file);
    inputEl.files = dt.files;
    applyFn(file);
  } catch (e) { /* audio optional */ }
}
function applyPresetToForm(j) {
  $("emoMode").value = String(j.emo_control_method ?? 0);
  $("emoWeight").value = j.emo_weight ?? 0.65;
  const adv = j.advanced_params || {};
  if (adv.do_sample != null) $("doSample").checked = !!adv.do_sample;
  if (adv.top_p != null) $("topP").value = adv.top_p;
  if (adv.top_k != null) $("topK").value = adv.top_k;
  if (adv.temperature != null) $("temperature").value = adv.temperature;
  if (adv.length_penalty != null) $("lenPen").value = adv.length_penalty;
  if (adv.num_beams != null) $("numBeams").value = adv.num_beams;
  if (adv.repetition_penalty != null) $("repPen").value = adv.repetition_penalty;
  if (adv.max_mel_tokens != null) $("maxMel").value = adv.max_mel_tokens;
  if (adv.max_text_tokens_per_segment != null) $("segTokens").value = adv.max_text_tokens_per_segment;
  if (j.emo_text != null) $("emoText").value = j.emo_text || "";
  setVecInputs(j.emo_vector || [0, 0, 0, 0, 0, 0, 0, 0]);
  bindSliders();
  emoVisible();
  if (j.prompt_audio_url) fetchPresetAudio(j.prompt_audio_url, $("spkFile"), setSpkFile);
  if (j.emo_audio_url) fetchPresetAudio(j.emo_audio_url, $("emoFile"), setEmoFile);
  scheduleSegments();
}
function collectPresetState(fd) {
  fd.append("emo_control_method", $("emoMode").value);
  fd.append("emo_weight", $("emoWeight").value);
  fd.append("emo_text", $("emoText").value);
  fd.append("use_random", $("emoRandom").checked);
  fd.append("do_sample", $("doSample").checked);
  fd.append("top_p", $("topP").value);
  fd.append("top_k", $("topK").value);
  fd.append("temperature", $("temperature").value);
  fd.append("length_penalty", $("lenPen").value);
  fd.append("num_beams", $("numBeams").value);
  fd.append("repetition_penalty", $("repPen").value);
  fd.append("max_mel_tokens", $("maxMel").value);
  fd.append("max_text_tokens_per_segment", $("segTokens").value);
  vecInputs().forEach((s) => fd.append("vec" + (parseInt(s.dataset.i) + 1), s.value));
}

/* library tab switching (presets / examples) */
const libTabs = document.querySelectorAll("#libTabs .libtab");
let curLib = "presets";

/* shared: apply a preset by name; returns success */
async function applyPresetByName(name) {
  try {
    const r = await fetch("/presets/" + encodeURIComponent(name));
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "加载失败");
    curPresetName = name;
    applyPresetToForm(j);
    showActionHint("已加载预设: " + name);
    updateGenSummary();
    return true;
  } catch (e) { showActionError("加载预设失败: " + e.message); return false; }
}

function switchLib(which) {
  curLib = which;
  libTabs.forEach((b) => b.classList.toggle("cur", b.dataset.lib === which));
  $("libSearch").value = "";
  renderLib();
  $("libTools").hidden = which !== "presets"; // 搜索/刷新/保存仅预设 tab 需要
}
libTabs.forEach((b) => b.addEventListener("click", () => switchLib(b.dataset.lib)));

async function refreshPresets() {
  try {
    const r = await (await fetch("/presets")).json();
    presetNamesCache = r.presets || [];
  } catch (err) { /* keep old list */ }
}

function renderLib() {
  const box = $("libList");
  const q = ($("libSearch").value || "").trim().toLowerCase();
  box.replaceChildren();
  if (curLib === "presets") {
    const names = presetNamesCache.filter((n) => !q || n.toLowerCase().includes(q));
    if (!names.length) {
      const empty = document.createElement("div");
      empty.className = "hist-item";
      empty.style.cursor = "default";
      empty.style.color = "var(--faint)";
      empty.textContent = q ? "没有匹配的预设" : "暂无预设，点右上角「保存当前为预设」创建";
      box.appendChild(empty);
      return;
    }
    for (const name of names) {
      const row = document.createElement("div");
      row.className = "hist-item" + (name === curPresetName ? " cur" : "");
      row.innerHTML = `<span class="htext"></span><span class="hmeta">加载中…</span><button class="pm-del" type="button" title="删除此预设">✕</button>`;
      row.querySelector(".htext").textContent = name;
      row.title = "点击应用此预设";
      row.querySelector(".pm-del").setAttribute("aria-label", "删除预设 " + name);
      const meta = row.querySelector(".hmeta");
      fetch("/presets/" + encodeURIComponent(name)).then((r) => r.ok ? r.json() : null).then((d) => {
        if (!d) { meta.textContent = "-"; return; }
        const adv = d.advanced_params || {};
        meta.textContent = `${["同参考", "情感音频", "向量", "文本"][d.emo_control_method] ?? "-"} · ${d.prompt_audio ? "含音频" : "无音频"} · temp ${adv.temperature ?? "-"}`;
      });
      row.addEventListener("click", async (e) => {
        if (e.target.closest(".pm-del")) return;
        if (await applyPresetByName(name)) renderLib();
      });
      row.querySelector(".pm-del").addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm("确定删除预设 " + name + " ？此操作不可恢复")) return;
        try {
          const r = await fetch("/presets/" + encodeURIComponent(name), { method: "DELETE" });
          if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "删除失败");
          if (curPresetName === name) { curPresetName = ""; updateGenSummary(); }
          showActionHint("已删除预设: " + name);
          await refreshPresets();
          renderLib();
          refreshPmList();
        } catch (err) { showActionError("删除预设失败: " + err.message); }
      });
      box.appendChild(row);
    }
  } else {
    // examples tab
    const items = exCache.filter((ex) => !q || (ex.text || "").toLowerCase().includes(q) || (ex.prompt_audio || "").toLowerCase().includes(q));
    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "hist-item";
      empty.style.cursor = "default";
      empty.style.color = "var(--faint)";
      empty.textContent = "暂无示例";
      box.appendChild(empty);
      return;
    }
    for (const ex of items) {
      const row = document.createElement("div");
      row.className = "hist-item";
      row.innerHTML = `<span class="htext"></span><span class="hmeta"></span>`;
      row.querySelector(".htext").textContent = ex.text || "";
      row.querySelector(".hmeta").textContent = (ex.prompt_audio || "").split("/").pop();
      row.title = "点击加载示例音频与参数";
      row.addEventListener("click", async () => {
        try {
          const audio = await fetch(ex.prompt_audio);
          if (!audio.ok) throw new Error("示例音频不存在");
          const blob = await audio.blob();
          const fileName = (ex.prompt_audio || "sample.wav").split("/").pop();
          const file = new File([blob], fileName, { type: "audio/wav" });
          const dt = new DataTransfer();
          dt.items.add(file);
          $("spkFile").files = dt.files;
          setSpkFile(file);
          $("text").value = ex.text || "";
          $("charCount").textContent = `${(ex.text || "").trim().length} 字`;
          $("emoMode").value = String(ex.emo_mode ?? 0);
          if (ex.emo_weight != null) { $("emoWeight").value = ex.emo_weight; }
          $("emoText").value = ex.emo_text || "";
          setVecInputs([ex.emo_vec_1, ex.emo_vec_2, ex.emo_vec_3, ex.emo_vec_4,
            ex.emo_vec_5, ex.emo_vec_6, ex.emo_vec_7, ex.emo_vec_8]);
          bindSliders();
          emoVisible();
          curPresetName = "";
          scheduleSegments();
          showActionHint("示例已加载: " + ((ex.text || "").slice(0, 30) || fileName));
          updateGenSummary();
        } catch (err) { showActionError("加载示例失败: " + err.message); }
      });
      box.appendChild(row);
    }
  }
}
$("libSearch").addEventListener("input", renderLib);
$("btnLibRefresh").addEventListener("click", async () => {
  if (curLib !== "presets") return;
  clearActionError();
  await refreshPresets();
  renderLib();
  if (!presetNamesCache.length) showActionHint("暂无预设");
});
$("btnLibSave").addEventListener("click", async () => {
  const name = prompt("请输入预设名称：");
  if (name === null) return;
  const trimmed = name.trim();
  if (!trimmed) { showActionError("请输入预设名称"); return; }
  const fd = new FormData();
  fd.append("name", trimmed);
  collectPresetState(fd);
  const spkF = $("spkFile").files[0];
  if (spkF) fd.append("prompt_audio", spkF);
  const emoF = $("emoFile").files[0];
  if (emoF) fd.append("emo_audio", emoF);
  try {
    const r = await fetch("/presets", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "保存失败");
    curPresetName = trimmed;
    showActionHint("预设已保存: " + trimmed);
    await refreshPresets();
    switchLib("presets");
    updateGenSummary();
    refreshPmList();
  } catch (e) { showActionError("保存预设失败: " + e.message); }
});

async function loadExamples() {
  try {
    const j = await (await fetch("/examples")).json();
    exCache = j.examples || [];
  } catch (err) { exCache = []; }
}

/* ---------------- generation history (right sidebar) ----------------
   Scope is this server run: the list lives in the backend's memory, so a page
   refresh keeps it but a server restart clears it. No persistence by design. */
const histBar = $("histBar");
const genPage = document.querySelector('.page[data-page="gen"]');
const HIST_KEY = "histCollapsed";
const EMO_LABEL = ["同参考", "情感音频", "向量", "文本"];
let histItems = [];
let histCurId = "";

function histSetCollapsed(collapsed, persist = true) {
  histBar.classList.toggle("collapsed", collapsed);
  genPage.classList.toggle("hist-collapsed", collapsed);
  if (persist) localStorage.setItem(HIST_KEY, collapsed ? "1" : "0");
  $("histRailCount").textContent = histItems.length;
}

function histFmtTime(ts) {
  const d = new Date((ts || 0) * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function histFmtSize(bytes) {
  const b = Number(bytes) || 0;
  return b >= 1048576 ? (b / 1048576).toFixed(1) + "M" : Math.max(1, Math.round(b / 1024)) + "K";
}

function renderHistory() {
  const box = $("histList");
  $("histCount").textContent = histItems.length;
  $("histRailCount").textContent = histItems.length;
  box.replaceChildren();
  if (!histItems.length) {
    const empty = document.createElement("div");
    empty.className = "hb-empty";
    empty.textContent = "暂无记录。生成一次语音后，这里会留下本次运行的历史。";
    box.appendChild(empty);
    return;
  }
  for (const it of histItems) {
    const row = document.createElement("div");
    row.className = "hb-item" + (it.id === histCurId ? " cur" : "");
    row.innerHTML = `<span class="hb-text"></span><span class="hb-meta"></span><button class="pm-del" type="button" title="删除这条记录">✕</button>`;
    row.querySelector(".hb-text").textContent = it.text || "(无文本)";
    row.querySelector(".hb-meta").textContent =
      `${histFmtTime(it.created_at)} · ${Number(it.elapsed).toFixed(1)}s · ${histFmtSize(it.size)} · ${EMO_LABEL[it.emo_mode] ?? "-"}`;
    row.title = "点击载入到右侧播放器";
    const del = row.querySelector(".pm-del");
    del.setAttribute("aria-label", "删除这条历史记录");
    row.addEventListener("click", async (e) => {
      if (e.target.closest(".pm-del")) return;
      histCurId = it.id;
      renderHistory();
      await loadResult(it.url, it.elapsed);
    });
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      try {
        const r = await fetch("/history/" + encodeURIComponent(it.id), { method: "DELETE" });
        if (!r.ok) throw new Error("删除失败");
        if (histCurId === it.id) histCurId = "";
        await refreshHistory();
      } catch (err) { showActionError("删除历史失败: " + err.message); }
    });
    box.appendChild(row);
  }
}

async function refreshHistory() {
  try {
    const r = await (await fetch("/history")).json();
    histItems = r.items || [];
    $("histAuto").checked = !!r.auto_clean;
    $("histNote").textContent = `本次运行生成 · 自动清理保留最近 ${r.limit} 条`;
    if (histBar.classList.contains("collapsed")) {
      $("histRailCount").textContent = histItems.length;
    } else {
      renderHistory();
    }
  } catch (e) { /* history is a side feature: never block the main flow */ }
}

$("histToggle").addEventListener("click", () => histSetCollapsed(false));
$("histCollapse").addEventListener("click", () => histSetCollapsed(true));

$("histAuto").addEventListener("change", async (e) => {
  const fd = new FormData();
  fd.append("auto_clean", e.target.checked);
  try {
    const r = await (await fetch("/history/config", { method: "POST", body: fd })).json();
    $("histAuto").checked = !!r.auto_clean;
  } catch (err) { showActionError("切换自动清理失败"); }
});

$("histClear").addEventListener("click", async () => {
  if (!histItems.length) return;
  if (!confirm("清空生成历史？自动清理开启时，对应的音频文件会一并删除，此操作不可恢复。")) return;
  try {
    await fetch("/history", { method: "DELETE" });
    histCurId = "";
    await refreshHistory();
  } catch (err) { showActionError("清空历史失败"); }
});

/* ---------------- glossary ---------------- */
const glError = $("glError");
function showGlossaryError(msg) {
  glError.textContent = msg;
  glError.hidden = false;
}
function clearGlossaryError() {
  glError.textContent = "";
  glError.hidden = true;
}

async function renderGlossary() {
  try {
    const g = (await (await fetch("/glossary")).json()).glossary || {};
    const keys = Object.keys(g);
    const box = $("glTable");
    box.replaceChildren();
    if (!keys.length) {
      const empty = document.createElement("div");
      empty.className = "gl-row";
      const term = document.createElement("span");
      term.className = "gl-term";
      term.textContent = "暂无自定义术语";
      empty.append(term, document.createElement("span"), document.createElement("span"));
      box.appendChild(empty);
      return;
    }
    for (const key of keys) {
      const value = g[key];
      const row = document.createElement("div");
      row.className = "gl-row";
      const term = document.createElement("span");
      term.className = "gl-term";
      const zh = document.createElement("span");
      const en = document.createElement("span");
      row.append(term, zh, en);
      term.textContent = key;
      zh.textContent = typeof value === "object" ? (value.zh || "") : "";
      en.textContent = typeof value === "object" ? (value.en || "") : (value || "");
      box.appendChild(row);
    }
  } catch (err) { /* keep old list */ }
}
$("btnAddTerm").addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("term", $("glTerm").value);
  fd.append("reading_zh", $("glZh").value);
  fd.append("reading_en", $("glEn").value);
  try {
    const r = await fetch("/glossary", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) { showGlossaryError("添加术语失败: " + (j.detail || "")); return; }
    $("glTerm").value = ""; $("glZh").value = ""; $("glEn").value = "";
    clearGlossaryError();
    await renderGlossary();
    showActionHint("术语已添加");
  } catch (err) { showGlossaryError("添加术语失败: " + err); }
});
["glTerm", "glZh", "glEn"].forEach((id) => {
  $(id).addEventListener("input", clearGlossaryError);
});

/* ---------------- page tabs (gen / presets / model) ---------------- */
const tabBtns = document.querySelectorAll("#tabs .tab");
function switchPage(name) {
  tabBtns.forEach((b) => b.classList.toggle("cur", b.dataset.page === name));
  document.querySelectorAll(".page").forEach((p) => {
    const on = p.dataset.page === name;
    p.classList.toggle("cur", on);
    p.hidden = !on;
  });
  if (name === "model") refreshModelPage();
  if (name === "presets") refreshPmList();
}
tabBtns.forEach((b) => b.addEventListener("click", () => switchPage(b.dataset.page)));

/* ---------------- preset manage page (card flow + search) ---------------- */
const pmCards = $("pmCards"), pmHint = $("pmHint"), pmSearch = $("pmSearch");
let pmListCache = [];       // all preset names
let pmDetailCache = {};     // name -> detail object (lazy)
let pmCurName = "";
function pmFlash(msg, isErr) {
  pmHint.textContent = msg;
  pmHint.style.color = isErr ? cssVar("--primary") : "";
}
const EMO_MODE_NAMES = ["与音色参考相同", "情感参考音频", "情感向量", "情感描述文本"];

async function refreshPmList() {
  try {
    const r = await (await fetch("/presets")).json();
    pmListCache = r.presets || [];
    renderPmCards();
  } catch (err) {
    pmFlash("刷新预设失败: " + err, true);
  }
}

async function pmLoadDetail(name) {
  if (pmDetailCache[name]) return pmDetailCache[name];
  try {
    const r = await fetch("/presets/" + encodeURIComponent(name));
    if (!r.ok) return null;
    const d = await r.json();
    pmDetailCache[name] = d;
    return d;
  } catch (e) { return null; }
}

async function renderPmCards() {
  const q = (pmSearch.value || "").trim().toLowerCase();
  const names = pmListCache.filter((n) => !q || n.toLowerCase().includes(q));
  pmCards.replaceChildren();
  if (!names.length) {
    const empty = document.createElement("div");
    empty.className = "pm-empty";
    empty.textContent = q ? `没有匹配「${q}」的预设` : "暂无预设，可在右侧创建";
    pmCards.appendChild(empty);
    return;
  }
  for (const name of names) {
    const card = document.createElement("div");
    card.className = "pm-card" + (name === pmCurName ? " cur" : "");
    card.innerHTML = `
      <span class="pm-name"></span>
      <span class="pm-meta"><span class="pm-m1">加载中…</span></span>
      <button class="pm-del" type="button" title="删除此预设">✕</button>`;
    card.querySelector(".pm-name").textContent = name;
    card.querySelector(".pm-del").setAttribute("aria-label", "删除预设 " + name);
    const m1 = card.querySelector(".pm-m1");
    // lazy detail: first line of info (emo mode + audio)
    pmLoadDetail(name).then((d) => {
      if (!d) { m1.textContent = "详情加载失败"; return; }
      const adv = d.advanced_params || {};
      m1.textContent = `${EMO_MODE_NAMES[d.emo_control_method] ?? d.emo_control_method} · ${d.prompt_audio ? "含音频" : "无音频"} · temp ${adv.temperature ?? "-"}`;
    });
    // click card → apply preset (jump to gen page)
    card.addEventListener("click", async (e) => {
      if (e.target.closest(".pm-del")) return; // delete handled below
      pmCurName = name;
      if (await applyPresetByName(name)) switchPage("gen");
    });
    // click ✕ → delete with confirm
    card.querySelector(".pm-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm("确定删除预设 " + name + " ？此操作不可恢复")) return;
      try {
        const r = await fetch("/presets/" + encodeURIComponent(name), { method: "DELETE" });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "删除失败");
        delete pmDetailCache[name];
        if (pmCurName === name) pmCurName = "";
        pmFlash("已删除预设: " + name);
        await refreshPmList();
        await refreshPresets();
      } catch (err) { pmFlash("删除失败: " + err.message, true); }
    });
    pmCards.appendChild(card);
  }
}
pmSearch.addEventListener("input", renderPmCards);
$("btnPmRefresh").addEventListener("click", () => { pmDetailCache = {}; refreshPmList(); pmFlash("已刷新"); });
$("btnPmCreate").addEventListener("click", async () => {
  const name = $("pmName").value.trim();
  if (!name) { pmFlash("请输入预设名称", true); return; }
  const fd = new FormData();
  fd.append("name", name);
  collectPresetState(fd);
  const spkF = $("spkFile").files[0];
  if (spkF) fd.append("prompt_audio", spkF);
  const emoF = $("emoFile").files[0];
  if (emoF) fd.append("emo_audio", emoF);
  try {
    const r = await fetch("/presets", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "创建失败");
    $("pmName").value = "";
    pmCurName = name;
    pmFlash("预设已创建: " + name);
    await refreshPmList();
    await refreshPresets();
    pmDetailCache[name] = undefined; // force reload of its detail
    delete pmDetailCache[name];
  } catch (e) { pmFlash("创建失败: " + e.message, true); }
});

/* ---------------- model manage page ---------------- */
const mmState = $("mmState"), mmRuntime = $("mmRuntime"), mmErr = $("mmErr");
const CFG_KEYS = ["fp16", "s2mel_fp16", "w2v_fp16", "qwen_fp16", "cudnn_benchmark"];
const CFG_BOX = {
  fp16: "cfgFp16", s2mel_fp16: "cfgS2melFp16", w2v_fp16: "cfgW2vFp16", qwen_fp16: "cfgQwenFp16", cudnn_benchmark: "cfgCudnnBench",
};
const mmCfgSliders = [
  { id: "cfgSteps", out: "cfgStepsOut", dec: 0 },
  { id: "cfgRate", out: "cfgRateOut", dec: 2 },
];
for (const s of mmCfgSliders) {
  const el = $(s.id), out = $(s.out);
  const upd = () => { out.textContent = Number(el.value).toFixed(s.dec); sliderFill(el); };
  el.addEventListener("input", upd);
  upd();
}
function mmFlashErr(msg) { mmErr.textContent = msg; mmErr.classList.toggle("on", !!msg); }
/* 模型加载/重启期间的实时耗时提示，写在状态位上；返回 interval id，调用方负责 clearInterval */
function mmElapsedTimer(label) {
  const t0 = Date.now();
  mmState.textContent = `${label}… 0s`;
  mmState.className = "mono";
  return setInterval(() => {
    mmState.textContent = `${label}… ${Math.round((Date.now() - t0) / 1000)}s`;
  }, 500);
}
async function refreshModelPage() {
  try {
    const s = await (await fetch("/model")).json();
    mmState.textContent = s.loaded ? "已加载" : "未加载";
    mmState.className = "mono " + (s.loaded ? "ok" : "off");
    for (const k of CFG_KEYS) { const el = $(CFG_BOX[k]); if (el) el.checked = !!s[k]; }
    $("cfgSteps").value = s.diffusion_steps ?? 25;
    $("cfgRate").value = s.inference_cfg_rate ?? 0.7;
    for (const s2 of mmCfgSliders) {
      const el = $(s2.id), out = $(s2.out);
      out.textContent = Number(el.value).toFixed(s2.dec);
      sliderFill(el);
    }
    mmRuntime.textContent = [
      `FP16=${s.fp16}  S2MEL_FP16=${s.s2mel_fp16}`,
      `W2V_FP16=${s.w2v_fp16}  QWEN_FP16=${s.qwen_fp16}`,
      `CUDNN_BENCHMARK=${s.cudnn_benchmark}`,
      `DIFFUSION_STEPS=${s.diffusion_steps}  CFG_RATE=${s.inference_cfg_rate}`,
    ].join("\n");
    mmFlashErr("");
  } catch (e) {
    mmFlashErr("无法读取模型状态: " + e);
  }
}
$("btnModelLoad").addEventListener("click", async () => {
  const btn = $("btnModelLoad");
  if (btn.disabled) return;               // 防止加载期间重复点击
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "加载中…";
  mmFlashErr("");
  const iv = mmElapsedTimer("加载中");     // 首次加载 30~120s，状态区实时计时
  try {
    const r = await (await fetch("/model/load", { method: "POST" })).json();
    if (!r.ok) throw new Error(r.error || "加载失败");
    pollStatus();
  } catch (e) { mmFlashErr(e.message); }
  clearInterval(iv);
  btn.disabled = false;
  btn.textContent = old;
  await refreshModelPage();               // 无论成败都刷新状态位，清掉计时文案
});
$("btnModelUnload").addEventListener("click", async () => {
  if (!confirm("确定卸载模型？卸载后首次生成需重新加载（约 30~60 秒）")) return;
  try {
    const r = await (await fetch("/model/unload", { method: "POST" })).json();
    if (!r.ok) throw new Error(r.error || "卸载失败");
    await refreshModelPage();
    pollStatus();
  } catch (e) { mmFlashErr(e.message); }
});
$("btnModelRestart").addEventListener("click", async () => {
  if (!confirm("确定重启模型？将应用当前页面上的加速配置，期间无法生成")) return;
  const btn = $("btnModelRestart");
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "重启中…";
  mmFlashErr("");
  const iv = mmElapsedTimer("重启中");
  try {
    const r = await (await fetch("/model/restart", { method: "POST" })).json();
    if (!r.ok) throw new Error(r.error || "重启失败");
    pollStatus();
  } catch (e) { mmFlashErr(e.message); }
  clearInterval(iv);
  btn.disabled = false;
  btn.textContent = old;
  await refreshModelPage();
});
/* 推荐配置：两个半精度开关开，cuDNN 自动调优关（实测负优化：BigVGAN 3.35s → 97s）；
   步数 25、CFG 0.7 */
const RECOMMEND_CFG = {
  s2mel_fp16: true, fp16: true, w2v_fp16: true, qwen_fp16: true, cudnn_benchmark: false,
  diffusion_steps: 25, inference_cfg_rate: 0.7,
};
$("btnPresetCfg").addEventListener("click", () => {
  for (const k of CFG_KEYS) {
    const el = $(CFG_BOX[k]);
    if (el) el.checked = !!RECOMMEND_CFG[k];
  }
  $("cfgSteps").value = RECOMMEND_CFG.diffusion_steps;
  $("cfgSteps").dispatchEvent(new Event("input", { bubbles: true }));
  $("cfgRate").value = RECOMMEND_CFG.inference_cfg_rate;
  $("cfgRate").dispatchEvent(new Event("input", { bubbles: true }));
  mmFlashErr("");
});

$("btnCfgSave").addEventListener("click", async () => {
  const body = {};
  for (const k of CFG_KEYS) body[k] = $(CFG_BOX[k]).checked;
  body.diffusion_steps = parseInt($("cfgSteps").value, 10);
  body.inference_cfg_rate = parseFloat($("cfgRate").value);
  try {
    const r = await fetch("/model/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!j.ok) throw new Error(j.detail || "保存失败");
    await refreshModelPage();
    pollStatus();
    const save = $("btnCfgSave");
    save.textContent = "已保存 ✓";
    setTimeout(() => { save.textContent = "保存配置"; }, 1600);
  } catch (e) { mmFlashErr("保存配置失败: " + e.message); }
});

/* ---------------- init (single entry point) ---------------- */
applyTheme(localStorage.getItem("theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
/* 侧边栏初始状态：没有保存过偏好就按窗口宽度决定（宽屏展开，窄屏收起） */
const _savedHist = localStorage.getItem(HIST_KEY);
histSetCollapsed(_savedHist === null ? window.innerWidth < 1360 : _savedHist === "1", false);
refreshHistory();
pollStatus();
pollMetrics();
bindSliders();
emoVisible();
updateSegments();
updateGenSummary();
switchLib("presets");
(async () => {
  await Promise.all([refreshPresets(), loadExamples()]);
  renderLib();
})();
refreshPmList();
renderGlossary();
switchPage("gen");

const _ro = new ResizeObserver(() => {
  drawHeaderWave();
  if (spkBuffer) drawSpkWave();
  if (trim.buffer) drawTrimWave();
});
_ro.observe(hWave);
_ro.observe(spkWave);
_ro.observe(tWave);

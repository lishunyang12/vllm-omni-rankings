const pptxgen = require('pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Shunyang Li';
pptx.company = 'vLLM / VeRL Community';
pptx.subject = 'vLLM-Omni and VeRL-Omni technical discussion for NVIDIA engineers';
pptx.title = 'vLLM-Omni × VeRL-Omni: An Open Systems Stack for Omni-Modal Inference and RL';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Arial',
  bodyFontFace: 'Arial',
  lang: 'en-US',
};
pptx.defineSlideMaster({
  title: 'DARK',
  background: { color: 'FFFFFF' },
  objects: [],
  slideNumber: { x: 12.78, y: 7.13, w: 0.25, h: 0.14, color: '8B9196', fontFace: 'Arial', fontSize: 7, align: 'right', margin: 0 },
});

const C = {
  bg: 'FFFFFF',
  panel: 'F5F6F7',
  panel2: 'ECEFF1',
  panel3: 'DDE1E4',
  white: '0B0C0D',
  text: '303438',
  muted: '656B70',
  faint: '8B9196',
  line: 'CCD1D5',
  green: '76B900',
  green2: '568800',
  blue: '0067B1',
  cyan: '007C83',
  orange: 'C45B00',
  purple: '6336A3',
  red: 'C62828',
  yellow: '9A7600',
};

const S = pptx.ShapeType;
const OUTPUT = 'scripts/vLLM-Omni_x_VeRL-Omni_for_NVIDIA_Technical_Discussion.pptx';

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: opts.fontFace || 'Arial',
    fontSize: opts.fontSize || 14,
    color: opts.color || C.text,
    bold: opts.bold || false,
    align: opts.align || 'left',
    valign: opts.valign || 'mid',
    margin: opts.margin === undefined ? 0 : opts.margin,
    breakLine: opts.breakLine,
    fit: opts.fit || 'shrink',
    paraSpaceAfterPt: opts.paraSpaceAfterPt,
    bullet: opts.bullet,
    isTextBox: true,
    ...opts,
  });
}

function addLine(slide, x1, y1, x2, y2, color = C.line, width = 1, endArrowType) {
  slide.addShape(S.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width, endArrowType },
  });
}

function addRect(slide, x, y, w, h, fill = C.panel, line = C.line, radius = 0) {
  slide.addShape(radius ? S.roundRect : S.rect, {
    x, y, w, h,
    rectRadius: radius,
    fill: { color: fill },
    line: { color: line, width: 1 },
  });
}

function addTag(slide, text, x, y, w, color = C.green) {
  slide.addShape(S.roundRect, {
    x, y, w, h: 0.26,
    rectRadius: 0.05,
    fill: { color, transparency: 82 },
    line: { color, transparency: 35, width: 0.7 },
  });
  addText(slide, text.toUpperCase(), x + 0.08, y + 0.01, w - 0.16, 0.22, {
    fontFace: 'Consolas', fontSize: 7.5, color, bold: true, charSpacing: 0.8,
  });
}

function addTitle(slide, section, title, subtitle) {
  addText(slide, section.toUpperCase(), 4.68, 0.21, 4.0, 0.18, {
    fontFace: 'Consolas', fontSize: 6.8, color: C.green2, bold: true, align: 'center', charSpacing: 1.2,
  });
  addText(slide, title, 0.48, 0.68, 12.2, 0.56, { fontSize: 26, bold: true, color: C.white, fit: 'shrink' });
  if (subtitle) {
    addText(slide, subtitle, 0.5, 1.22, 12.0, 0.32, { fontSize: 11.5, color: C.muted });
  }
}

function addFooter(slide, source, label = 'TECHNICAL DISCUSSION') {
  addLine(slide, 0.48, 7.05, 12.85, 7.05, C.line, 0.6);
  addText(slide, source, 0.5, 7.09, 10.2, 0.16, { fontSize: 6.8, color: C.faint });
  addText(slide, label, 10.72, 7.09, 1.88, 0.16, { fontFace: 'Consolas', fontSize: 6.8, color: C.faint, align: 'right' });
}

function addCard(slide, x, y, w, h, title, body, accent = C.green, opts = {}) {
  addRect(slide, x, y, w, h, opts.fill || C.panel, opts.line || C.line);
  slide.addShape(S.rect, { x, y, w: 0.055, h, fill: { color: accent }, line: { color: accent } });
  if (opts.kicker) {
    addText(slide, opts.kicker.toUpperCase(), x + 0.2, y + 0.15, w - 0.35, 0.18, {
      fontFace: 'Consolas', fontSize: 7.3, color: accent, bold: true, charSpacing: 0.6,
    });
  }
  addText(slide, title, x + 0.2, y + (opts.kicker ? 0.38 : 0.2), w - 0.34, 0.34, {
    fontSize: opts.titleSize || 14, color: C.white, bold: true, valign: 'top',
  });
  if (body) {
    addText(slide, body, x + 0.2, y + (opts.kicker ? 0.82 : 0.62), w - 0.34, h - (opts.kicker ? 0.96 : 0.76), {
      fontSize: opts.bodySize || 10, color: opts.bodyColor || C.muted, valign: 'top', breakLine: false,
    });
  }
}

function addNode(slide, x, y, w, h, title, subtitle, color, opts = {}) {
  slide.addShape(S.roundRect, {
    x, y, w, h,
    rectRadius: 0.06,
    fill: { color: opts.fill || C.panel2 },
    line: { color, width: opts.width || 1.5 },
  });
  addText(slide, title, x + 0.08, y + 0.12, w - 0.16, subtitle ? 0.25 : h - 0.12, {
    fontSize: opts.titleSize || 12.5, color: C.white, bold: true, align: 'center', valign: subtitle ? 'mid' : 'mid',
  });
  if (subtitle) {
    addText(slide, subtitle, x + 0.08, y + 0.43, w - 0.16, h - 0.52, {
      fontSize: opts.subSize || 8.5, color: opts.subColor || C.muted, align: 'center', valign: 'top',
    });
  }
}

function addArrow(slide, x1, y1, x2, y2, color = C.green, width = 1.5) {
  addLine(slide, x1, y1, x2, y2, color, width, 'triangle');
}

function addMetric(slide, x, y, w, value, label, detail, color = C.green) {
  addRect(slide, x, y, w, 1.42, C.panel, C.line);
  addText(slide, value, x + 0.16, y + 0.12, w - 0.32, 0.62, { fontSize: 27, bold: true, color });
  addText(slide, label, x + 0.17, y + 0.72, w - 0.34, 0.22, { fontSize: 10.5, bold: true, color: C.white });
  addText(slide, detail, x + 0.17, y + 0.99, w - 0.34, 0.25, { fontSize: 8.2, color: C.muted });
}

function addTechnicalMesh(slide, opts = {}) {
  const x0 = opts.x === undefined ? 7.45 : opts.x;
  const y0 = opts.y === undefined ? 0 : Math.max(0, opts.y);
  const color = opts.color || C.line;
  for (let i = 0; i < 9; i++) {
    addLine(slide, x0 - 1.45 + i * 0.62, y0, x0 + i * 0.62, y0 + 3.6, color, 0.45);
  }
  for (let i = 0; i < 6; i++) {
    addLine(slide, x0 - 1.6, y0 + i * 0.63, 13.3, y0 + i * 0.63, color, 0.35);
  }
  slide.addShape(S.rect, {
    x: 9.6, y: 0.35, w: 3.1, h: 1.18, rotate: -14,
    fill: { color: C.green, transparency: 7 }, line: { color: C.green, transparency: 100 },
  });
  slide.addShape(S.rect, {
    x: 10.25, y: 1.28, w: 2.4, h: 0.62, rotate: -14,
    fill: { color: C.green, transparency: 28 }, line: { color: C.green, transparency: 100 },
  });
}

function sectionSlide(kicker, title, statement, number, color = C.green) {
  const slide = pptx.addSlide('DARK');
  slide.background = { color: C.bg };
  addTechnicalMesh(slide, { x: 8.0, y: -0.2 });
  slide.addShape(S.rect, { x: 9.25, y: 2.1, w: 3.95, h: 2.7, rotate: -17, fill: { color, transparency: 4 }, line: { color, transparency: 100 } });
  slide.addShape(S.rect, { x: 0, y: 0, w: 0.16, h: 7.5, fill: { color }, line: { color } });
  addTag(slide, kicker, 0.68, 0.72, Math.max(1.3, kicker.length * 0.09 + 0.45), color);
  addText(slide, number, 10.72, 0.64, 1.75, 0.9, { fontFace: 'Consolas', fontSize: 50, bold: true, color: C.green2, align: 'right' });
  addText(slide, title, 0.68, 2.12, 11.6, 0.86, { fontSize: 37, bold: true, color: C.white });
  addText(slide, statement, 0.7, 3.18, 10.4, 0.58, { fontSize: 18, color: C.muted });
  addLine(slide, 0.7, 4.06, 4.0, 4.06, color, 3);
  addText(slide, 'vLLM-Omni × VeRL-Omni', 0.7, 6.72, 3.1, 0.25, { fontFace: 'Consolas', fontSize: 9, color: C.faint });
  addFooter(slide, 'Community-maintained technical material · September 2026', 'FOR NVIDIA TECHNICAL DISCUSSION');
  return slide;
}

function addTable(slide, rows, x, y, w, h, colWidths, opts = {}) {
  const rowH = h / rows.length;
  rows.forEach((row, ri) => {
    let cx = x;
    row.forEach((cell, ci) => {
      const cw = colWidths[ci] * w;
      const header = ri === 0;
      slide.addShape(S.rect, {
        x: cx, y: y + ri * rowH, w: cw, h: rowH,
        fill: { color: header ? (opts.headerFill || 'EAF3DA') : (ri % 2 ? C.panel : C.panel2) },
        line: { color: C.line, width: 0.6 },
      });
      addText(slide, cell, cx + 0.09, y + ri * rowH + 0.04, cw - 0.18, rowH - 0.08, {
        fontSize: header ? (opts.headerSize || 9) : (opts.bodySize || 8.5),
        color: header ? (opts.headerColor || C.green2) : (ci === 0 ? C.white : C.text),
        bold: header || ci === 0,
        valign: 'mid',
      });
      cx += cw;
    });
  });
}

// 1 — Cover
{
  const slide = pptx.addSlide('DARK');
  slide.background = { color: C.bg };
  addTechnicalMesh(slide, { x: 8.2, y: -0.15 });
  slide.addShape(S.rect, { x: 0, y: 0, w: 0.18, h: 7.5, fill: { color: C.green }, line: { color: C.green } });
  addTag(slide, 'OPEN OMNI SYSTEMS', 0.72, 0.62, 1.9);
  addText(slide, 'vLLM-Omni × VeRL-Omni', 0.72, 1.25, 7.4, 0.55, { fontSize: 31, bold: true, color: C.white });
  addText(slide, 'An Open Systems Stack for\nOmni-Modal Inference and RL', 0.72, 1.88, 7.2, 1.28, { fontSize: 30, bold: true, color: C.white, breakLine: false });
  addText(slide, 'Technical discussion for NVIDIA systems and performance engineers', 0.75, 3.38, 6.9, 0.42, { fontSize: 14, color: C.muted });
  addText(slide, 'Shunyang Li  ·  September 2026', 0.75, 5.84, 4.4, 0.3, { fontFace: 'Consolas', fontSize: 10, color: C.text });
  addText(slide, 'Community-maintained material · no NVIDIA endorsement implied', 0.75, 6.2, 5.4, 0.24, { fontSize: 8, color: C.faint });

  addRect(slide, 8.35, 1.28, 4.2, 4.72, C.panel, C.line);
  addText(slide, 'ONE STACK · TWO LOOPS', 8.72, 1.64, 3.46, 0.28, { fontFace: 'Consolas', fontSize: 9, color: C.green2, bold: true, align: 'center', charSpacing: 1.1 });
  addNode(slide, 9.02, 2.18, 2.85, 0.82, 'vLLM-Omni', 'serving · rollout · streaming', C.green);
  addArrow(slide, 10.45, 3.02, 10.45, 3.42, C.green);
  addNode(slide, 9.02, 3.44, 2.85, 0.82, 'VeRL-Omni', 'reward · update · orchestration', C.cyan);
  addArrow(slide, 11.88, 3.82, 12.22, 3.82, C.cyan);
  addArrow(slide, 12.22, 3.82, 12.22, 2.58, C.cyan);
  addArrow(slide, 12.22, 2.58, 11.88, 2.58, C.cyan);
  addText(slide, 'weights', 11.96, 3.06, 0.48, 0.2, { fontFace: 'Consolas', fontSize: 7, color: C.cyan, rotate: 270, align: 'center' });
  addText(slide, 'GPU execution data plane', 8.94, 4.72, 3.0, 0.28, { fontSize: 10, color: C.muted, align: 'center' });
  addLine(slide, 9.15, 5.2, 11.72, 5.2, C.green, 3);
  addFooter(slide, 'Sources integrated from four uploaded PPTX decks, two supplied PDFs, and current official project documentation.', 'FOR NVIDIA TECHNICAL DISCUSSION');
}

// 2 — Thesis
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'SYSTEM THESIS', 'Omni turns one model into a system graph', 'The familiar “one engine · one model · one decoding loop” assumption no longer holds.');
  addCard(slide, 0.55, 1.76, 3.15, 1.15, 'Input is no longer tokens', 'Waveforms, video frames, image latents, and outputs from earlier stages all enter the request lifecycle.', C.blue, { kicker: 'BROKEN 01', bodySize: 9.2 });
  addCard(slide, 0.55, 3.06, 3.15, 1.15, 'Output has no single EOS', 'Text tokens, codec codes, audio packets, images, video chunks, and actions have different completion contracts.', C.orange, { kicker: 'BROKEN 02', bodySize: 9.2 });
  addCard(slide, 0.55, 4.36, 3.15, 1.15, 'Execution is heterogeneous', 'AR + DiT + VAE + codec stages need different batching, parallelism, precision, and scheduling policies.', C.purple, { kicker: 'BROKEN 03', bodySize: 9.2 });

  addText(slide, 'A representative request', 4.23, 1.78, 3.4, 0.28, { fontFace: 'Consolas', fontSize: 9, color: C.muted, bold: true });
  const nodes = [
    ['MEDIA', 'encode', C.blue], ['THINKER', 'AR', C.orange], ['TALKER', 'AR', C.cyan], ['GENERATOR', 'DiT / codec', C.purple], ['OUTPUT', 'stream', C.green],
  ];
  let x = 4.23;
  nodes.forEach((n, i) => {
    addNode(slide, x, 2.32, 1.44, 0.88, n[0], n[1], n[2], { titleSize: 10.5, subSize: 7.5 });
    if (i < nodes.length - 1) addArrow(slide, x + 1.45, 2.76, x + 1.74, 2.76, C.faint, 1.1);
    x += 1.75;
  });
  addRect(slide, 4.23, 3.65, 8.2, 1.85, C.panel, C.line);
  addText(slide, 'SYSTEM CONSEQUENCE', 4.48, 3.9, 2.2, 0.25, { fontFace: 'Consolas', fontSize: 8, color: C.green2, bold: true });
  addText(slide, 'Optimize each stage locally — then optimize the handoffs globally.', 4.48, 4.25, 7.3, 0.45, { fontSize: 19, bold: true, color: C.white });
  addText(slide, 'The serving runtime needs stage-aware execution. The RL runtime repeatedly invokes that serving path and adds reward, updates, and weight movement.', 4.48, 4.78, 7.38, 0.44, { fontSize: 10.5, color: C.muted });
  addFooter(slide, 'Source: vLLM-Omni architecture overview · “vLLM-Omni 架构演化” uploaded deck.');
}

// 3 — Workload landscape
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'WORKLOAD LANDSCAPE', 'One serving surface, four execution families', 'The system abstraction must survive model churn without flattening fundamentally different workloads.');
  addCard(slide, 0.55, 1.76, 3.0, 1.72, 'Omni & speech', 'Qwen3-Omni\nMiniCPM-o\nQwen3-TTS · Fish Speech', C.blue, { kicker: 'MULTI-STAGE AR + CODEC', bodySize: 11 });
  addCard(slide, 3.73, 1.76, 3.0, 1.72, 'Image · video · audio', 'Qwen-Image · FLUX\nWan · MiniMax H3\nLTX · SANA-Video', C.purple, { kicker: 'DIFFUSION / FLOW', bodySize: 11 });
  addCard(slide, 6.91, 1.76, 3.0, 1.72, 'Unified models', 'BAGEL\nHunyuanImage\nGLM-Image', C.orange, { kicker: 'AR + SPECIALIZED GEN', bodySize: 11 });
  addCard(slide, 10.09, 1.76, 2.7, 1.72, 'World & action', 'Cosmos · DreamZero\nπ0 · GR00T\nInternVLA', C.green, { kicker: 'STATEFUL GENERATION', bodySize: 11 });

  addText(slide, 'Shared requirements', 0.58, 3.9, 2.2, 0.28, { fontFace: 'Consolas', fontSize: 9, color: C.green2, bold: true });
  const reqs = [
    ['Heterogeneous I/O', 'text · image · audio · video · action'],
    ['Mixed execution loops', 'token decode · denoise · codec · VAE'],
    ['Stage-local scaling', 'TP · SP/USP · EP · CFG · replicas'],
    ['Streaming & state', 'partial outputs · KV · cancellation · ordering'],
  ];
  reqs.forEach((r, i) => {
    const rx = 0.58 + i * 3.06;
    addRect(slide, rx, 4.35, 2.82, 1.22, C.panel2, C.line);
    addText(slide, String(i + 1).padStart(2, '0'), rx + 0.16, 4.53, 0.42, 0.3, { fontFace: 'Consolas', fontSize: 11, color: C.green, bold: true });
    addText(slide, r[0], rx + 0.58, 4.49, 2.02, 0.3, { fontSize: 11, color: C.white, bold: true });
    addText(slide, r[1], rx + 0.58, 4.88, 2.02, 0.38, { fontSize: 8.3, color: C.muted });
  });
  addFooter(slide, 'Source: vLLM-Omni v0.26.0 README and supported-model documentation. Model list is illustrative, not exhaustive.');
}

// 4 — One stack, two loops
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'SYSTEM MAP', 'One GPU execution data plane, two feedback loops', 'Serving optimizes requests. RL post-training repeatedly invokes serving and optimizes the policy.');
  addRect(slide, 0.62, 1.82, 12.05, 4.54, C.panel, C.line);
  addText(slide, 'ONLINE / OFFLINE SERVING LOOP', 0.94, 2.08, 3.1, 0.25, { fontFace: 'Consolas', fontSize: 8.5, color: C.green2, bold: true });
  addNode(slide, 1.0, 2.55, 1.55, 0.85, 'REQUESTS', 'text · media', C.blue);
  addArrow(slide, 2.57, 2.98, 3.02, 2.98, C.green);
  addNode(slide, 3.04, 2.55, 2.05, 0.85, 'vLLM-Omni', 'stage execution', C.green);
  addArrow(slide, 5.11, 2.98, 5.56, 2.98, C.green);
  addNode(slide, 5.58, 2.55, 2.05, 0.85, 'OUTPUTS', 'text · media · action', C.orange);
  addArrow(slide, 7.65, 2.98, 8.1, 2.98, C.faint);
  addNode(slide, 8.12, 2.55, 1.75, 0.85, 'SLA', 'TTFP · JCT · RTF', C.purple);

  addText(slide, 'RL POST-TRAINING LOOP', 0.94, 4.0, 3.1, 0.25, { fontFace: 'Consolas', fontSize: 8.5, color: C.cyan, bold: true });
  addNode(slide, 1.0, 4.45, 1.55, 0.85, 'PROMPTS', 'grouped samples', C.blue);
  addArrow(slide, 2.57, 4.88, 3.02, 4.88, C.cyan);
  addNode(slide, 3.04, 4.45, 2.05, 0.85, 'ROLLOUT', 'vLLM-Omni replicas', C.green);
  addArrow(slide, 5.11, 4.88, 5.56, 4.88, C.cyan);
  addNode(slide, 5.58, 4.45, 1.72, 0.85, 'REWARD', 'async scorers', C.orange);
  addArrow(slide, 7.32, 4.88, 7.77, 4.88, C.cyan);
  addNode(slide, 7.79, 4.45, 1.72, 0.85, 'UPDATE', 'FSDP2 / VeOmni', C.purple);
  addArrow(slide, 9.53, 4.88, 10.0, 4.88, C.cyan);
  addNode(slide, 10.02, 4.45, 1.72, 0.85, 'WEIGHTS', 'broadcast / P2P', C.cyan);
  addArrow(slide, 10.88, 4.43, 10.88, 3.55, C.cyan);
  addArrow(slide, 10.88, 3.55, 5.08, 3.55, C.cyan);
  addText(slide, 'Serving is the inner loop of training.', 9.76, 2.02, 2.55, 0.34, { fontSize: 12, color: C.white, bold: true, align: 'right' });
  addFooter(slide, 'Sources: vLLM-Omni architecture overview · VeRL-Omni README and June 2026 system slides.');
}

// 5 — Section
sectionSlide('PART I · SERVING', 'vLLM-Omni', 'Stage-aware execution for heterogeneous omni-modal pipelines.', '01', C.green);

// 6 — Qwen3 stage graph
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'MODEL → STAGES', 'Qwen3-Omni exposes three serving workloads', 'Thinker, Talker, and Code2Wav have different state, cadence, batching, and latency sensitivity.');
  addNode(slide, 0.72, 2.4, 1.5, 1.0, 'INPUTS', 'text · image\naudio · video', C.blue);
  addArrow(slide, 2.24, 2.9, 2.72, 2.9, C.faint);
  addNode(slide, 2.74, 2.22, 2.18, 1.36, 'THINKER', 'AR · reasoning + text\nKV cache · MoE', C.orange, { titleSize: 15 });
  addArrow(slide, 4.94, 2.9, 5.42, 2.9, C.orange);
  addNode(slide, 5.44, 2.22, 2.18, 1.36, 'TALKER', 'AR · codec tokens\nstreaming cadence', C.cyan, { titleSize: 15 });
  addArrow(slide, 7.64, 2.9, 8.12, 2.9, C.cyan);
  addNode(slide, 8.14, 2.22, 2.18, 1.36, 'CODE2WAV', 'non-AR · waveform\nfirst-packet critical', C.purple, { titleSize: 15 });
  addArrow(slide, 10.34, 2.9, 10.82, 2.9, C.green);
  addNode(slide, 10.84, 2.4, 1.7, 1.0, 'OUTPUTS', 'streamed text\n+ audio', C.green);

  const details = [
    ['STATE', 'KV + hidden states', 'codec history', 'chunk / waveform state'],
    ['HOT PATH', 'prefill + decode', 'per-token predictor', 'vocoder / flow decode'],
    ['SCALE', 'TP / EP', 'batch / replicas', 'graphs / replicas'],
    ['PRIMARY KPI', 'TTFT · TPOT', 'token cadence', 'TTFP · RTF'],
  ];
  addTable(slide, [['', 'THINKER', 'TALKER', 'CODE2WAV'], ...details], 1.15, 4.2, 11.0, 1.55, [0.17, 0.277, 0.277, 0.276], { headerSize: 8.2, bodySize: 8.2 });
  addFooter(slide, 'Source: vLLM-Omni Qwen3-Omni performance deep dive · website PR #258 · merged 2026-07-03.');
}

// 7 — Runtime architecture
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'RUNTIME ARCHITECTURE', 'Topology, placement, control, and transport stay separate', 'A logical stage is not necessarily a process; deployment policy is external to model topology.');
  const cols = [0.68, 3.18, 5.68, 8.18, 10.68];
  const labels = [
    ['ENTRY POINTS', 'API server\nOmni / AsyncOmni', C.blue],
    ['ASYNC ENGINE', 'composition root\nbackground loop', C.green],
    ['ORCHESTRATOR', 'route · state · cancel\norder outputs', C.orange],
    ['STAGE RUNTIME', 'placement · replicas\nreadiness · lifecycle', C.cyan],
    ['EXECUTION', 'AR / diffusion\nscheduler · worker', C.purple],
  ];
  labels.forEach((n, i) => {
    addNode(slide, cols[i], 2.12, 2.02, 1.16, n[0], n[1], n[2], { titleSize: 11, subSize: 8.2 });
    if (i < labels.length - 1) addArrow(slide, cols[i] + 2.04, 2.7, cols[i + 1] - 0.1, 2.7, C.faint, 1.1);
  });
  addRect(slide, 1.16, 3.9, 11.0, 1.62, C.panel, C.line);
  addText(slide, 'CONFIGURATION PLANE', 1.4, 4.15, 2.1, 0.22, { fontFace: 'Consolas', fontSize: 8, color: C.green2, bold: true });
  addNode(slide, 1.45, 4.52, 2.42, 0.65, 'PipelineConfig', 'topology + relationships', C.blue, { titleSize: 10.5, subSize: 7.5 });
  addNode(slide, 4.18, 4.52, 2.42, 0.65, 'DeployConfig', 'devices + resources', C.orange, { titleSize: 10.5, subSize: 7.5 });
  addNode(slide, 6.91, 4.52, 2.42, 0.65, 'OmniConnector', 'payload + KV transport', C.cyan, { titleSize: 10.5, subSize: 7.5 });
  addNode(slide, 9.64, 4.52, 2.18, 0.65, 'StagePool', 'replicas + routing', C.green, { titleSize: 10.5, subSize: 7.5 });
  addText(slide, 'Model code defines stages. Deployment config decides where and how they run.', 2.04, 5.86, 9.4, 0.35, { fontSize: 14, color: C.white, bold: true, align: 'center' });
  addFooter(slide, 'Source: vLLM-Omni latest architecture overview · v0.26.0 system walkthrough.');
}

// 8 — Execution policies
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'STAGE-LOCAL POLICY', 'AR and diffusion need different execution contracts', 'The common runtime boundary is stable; batching and acceleration remain stage-specific.');
  const rows = [
    ['RUNTIME CONCERN', 'AR STAGE', 'DIFFUSION STAGE', 'SYSTEM IMPLICATION'],
    ['Execution loop', 'next-token decode', 'iterative denoising', 'different scheduler state machines'],
    ['Batching', 'continuous token batching', 'request or step batching', 'compatibility by shape / CFG / LoRA'],
    ['Attention', 'causal + KV cache', 'full or mixed attention', 'different kernels and memory traffic'],
    ['Parallelism', 'TP / EP / replicas', 'TP / SP / CFG / EP / VAE', 'placement must follow bottleneck'],
    ['Quantization', 'checkpoint / AR scope', 'component / pipeline scope', 'precision support cannot be assumed'],
    ['Primary metrics', 'TTFT · TPOT · throughput', 'JCT · denoise time · RTF', 'benchmark contract changes by stage'],
  ];
  addTable(slide, rows, 0.58, 1.78, 12.2, 4.6, [0.18, 0.25, 0.25, 0.32], { headerSize: 8.5, bodySize: 8.8 });
  addText(slide, 'Principle', 0.72, 6.56, 0.72, 0.2, { fontFace: 'Consolas', fontSize: 8, color: C.green2, bold: true });
  addText(slide, 'Share orchestration primitives; keep execution policy local to the stage.', 1.52, 6.48, 8.8, 0.34, { fontSize: 13, color: C.white, bold: true });
  addFooter(slide, 'Source: vLLM-Omni v0.26.0 architecture overview and diffusion design documents.');
}

// 9 — Performance stack
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'PERFORMANCE STACK', 'Performance comes from removing a sequence of different bottlenecks', 'No single kernel or parallelism strategy explains end-to-end gains.');
  const cards = [
    ['Stage split + batching', 'Keep unlike workloads independent; build batches at each stage.', C.blue, 'UTILIZATION'],
    ['CUDA Graph capture', 'Capture outer AR, predictor, and bucketed vocoder work locally.', C.green, 'LAUNCH OVERHEAD'],
    ['Async chunk', 'Forward partial codec chunks before upstream completion.', C.orange, 'FIRST PACKET'],
    ['Async output', 'Move payload assembly away from the decode worker.', C.cyan, 'DECODE STALLS'],
    ['Selective replicas', 'Replicate only saturated Talker / Code2Wav stages.', C.purple, 'CAPACITY'],
    ['Hot-path cleanup', 'Keep state on GPU and remove per-step concatenation and copies.', C.red, 'PER-STEP WORK'],
  ];
  cards.forEach((c, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    addCard(slide, 0.58 + col * 4.1, 1.82 + row * 2.12, 3.78, 1.78, c[0], c[1], c[2], { kicker: c[3], titleSize: 13, bodySize: 9.3 });
  });
  addFooter(slide, 'Source: vLLM-Omni Qwen3-Omni performance deep dive · website PR #258.');
}

// 10 — Performance case
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'PUBLISHED CASE STUDY', 'Qwen3-Omni C64: optimize throughput and first audio together', 'Cumulative sweep: batching → graphs → async handoffs → async output → selective replicas.');
  addMetric(slide, 0.58, 1.82, 3.72, '5.4×', 'THROUGHPUT', '2.2 → 11.7 requests/s', C.green);
  addMetric(slide, 4.49, 1.82, 3.72, '−89%', 'TIME TO FIRST AUDIO', '5.884 s → 0.632 s', C.orange);
  addMetric(slide, 8.4, 1.82, 3.72, '−59%', 'MEAN REAL-TIME FACTOR', '1.15 → 0.47', C.cyan);

  addText(slide, 'Cumulative optimization path', 0.6, 3.62, 2.5, 0.24, { fontFace: 'Consolas', fontSize: 8.5, color: C.muted, bold: true });
  const steps = [
    ['BATCH', C.blue], ['GRAPHS', C.green], ['ASYNC CHUNK', C.orange], ['ASYNC OUTPUT', C.cyan], ['REPLICAS', C.purple],
  ];
  steps.forEach((s, i) => {
    const x = 0.6 + i * 2.43;
    addNode(slide, x, 4.12, 1.88, 0.62, s[0], '', s[1], { titleSize: 9.5 });
    if (i < steps.length - 1) addArrow(slide, x + 1.9, 4.43, x + 2.3, 4.43, C.faint, 1.1);
  });
  addRect(slide, 0.6, 5.16, 11.94, 1.02, C.panel, C.line);
  addText(slide, 'BENCHMARK CONTRACT', 0.82, 5.4, 1.56, 0.2, { fontFace: 'Consolas', fontSize: 7.5, color: C.green2, bold: true });
  addText(slide, 'Seed-TTS en · Qwen3-Omni-30B-A3B-Instruct · concurrency 1 / 16 / 32 / 64 · 5 warmups · 3 GPU IDs · server restart per configuration', 2.4, 5.32, 9.72, 0.36, { fontSize: 9.2, color: C.text });
  addText(slide, 'Treat the numbers as a published case study—not a universal hardware claim.', 2.4, 5.72, 8.7, 0.22, { fontSize: 8.2, color: C.muted, italic: true });
  addFooter(slide, 'Source: vLLM-Omni website PR #258 · merged 2026-07-03 · uploaded v0.26.0 release deck.');
}

// 11 — NVIDIA optimization map
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'GPU OPTIMIZATION MAP', 'Where NVIDIA expertise has the highest leverage', 'Current runtime levers and candidate collaboration areas, organized by stage.');
  const rows = [
    ['STAGE', 'DOMINANT BOTTLENECK', 'CURRENT RUNTIME LEVERS', 'HIGH-LEVERAGE NVIDIA DISCUSSION'],
    ['Encoders', 'preprocess · attention · memory', 'TP · caching · stage placement', 'FA / cuDNN paths · graph capture · split placement'],
    ['AR / MoE', 'KV cache · decode · experts', 'continuous batching · TP / EP · FP8', 'FA4 · FP8 / NVFP4 · expert routing · CUDA Graph'],
    ['DiT', 'attention · GEMM · denoise loop', 'step batching · TP / SP / CFG / EP', 'Blackwell kernels · TRT-LLM / FA integration · fusion'],
    ['VAE / codec', 'conv · upsample · first packet', 'tiling · patch parallel · graphs', 'cuDNN · fused decode · latency-oriented kernels'],
    ['Connector', 'copy · synchronization · weight move', 'async transfer · SHM · remote transport', 'NVLink / NVSwitch placement · NCCL · GPUDirect'],
  ];
  addTable(slide, rows, 0.55, 1.8, 12.25, 4.62, [0.14, 0.23, 0.29, 0.34], { headerSize: 8.1, bodySize: 8.5 });
  addText(slide, 'Discussion rule', 0.7, 6.58, 1.04, 0.2, { fontFace: 'Consolas', fontSize: 7.6, color: C.green2, bold: true });
  addText(slide, 'Every speedup claim pairs latency / throughput evidence with output-quality or convergence evidence.', 1.82, 6.5, 9.9, 0.35, { fontSize: 12, color: C.white, bold: true });
  addFooter(slide, 'Sources: vLLM-Omni v0.26.0 parallelism, quantization, H3, Qwen3-Omni, and diffusion documentation. Candidate areas are proposals.');
}

// 12 — Section
sectionSlide('PART II · POST-TRAINING', 'VeRL-Omni', 'Serving becomes the inner loop of multimodal RL.', '02', C.cyan);

// 13 — Why multimodal RL differs
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'WHY A DEDICATED RL STACK', 'Multimodal generative RL is not text RL with bigger tensors', 'Model structure, I/O, rollout cost, reward latency, and convergence failure modes all change.');
  const rows = [
    ['SYSTEM DIMENSION', 'TEXT LLM RL', 'DIFFUSION / OMNI RL'],
    ['Generation unit', 'autoregressive tokens', 'denoising trajectory · media chunks · mixed stages'],
    ['Rollout cost', 'prefill + decode', 'N denoise steps + VAE / codec + media materialization'],
    ['Reward', 'rule / RM after text', 'OCR · aesthetic · VLM · audio / video · multiple scorers'],
    ['Parallelism', 'TP / DP / EP', 'TP / USP / SP / CFG / DP + stage placement'],
    ['Stability', 'policy drift · variance', 'trajectory correction · logP cost · deterministic media path'],
  ];
  addTable(slide, rows, 0.6, 1.82, 8.25, 4.55, [0.25, 0.34, 0.41], { headerSize: 8.4, bodySize: 8.7 });
  addCard(slide, 9.1, 1.82, 3.62, 1.28, 'Rollout dominates', 'Each prompt fans out to multiple expensive media trajectories.', C.green, { kicker: 'COST', bodySize: 9.2 });
  addCard(slide, 9.1, 3.26, 3.62, 1.28, 'Reward can hide—or block', 'Asynchronous scoring overlaps only if capacity and sample flow are designed together.', C.orange, { kicker: 'LATENCY', bodySize: 9.2 });
  addCard(slide, 9.1, 4.7, 3.62, 1.28, 'Training must converge', 'Serving throughput matters only when policy correction and numerics remain valid.', C.purple, { kicker: 'CORRECTNESS', bodySize: 9.2 });
  addFooter(slide, 'Source: VeRL-Omni README · June 2026 system walkthrough · current trainer documentation.');
}

// 14 — VeRL architecture
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'TRAINING ARCHITECTURE', 'A single controller composes three specialized engines', 'Training, rollout, and reward evolve independently behind Ray RPC and checkpoint movement.');
  addRect(slide, 0.6, 1.76, 12.14, 0.66, C.panel2, C.cyan);
  addText(slide, 'SINGLE CONTROLLER RL TRAINER', 0.85, 1.9, 11.65, 0.27, { fontFace: 'Consolas', fontSize: 11, color: C.cyan, bold: true, align: 'center', charSpacing: 1.3 });
  addCard(slide, 0.78, 2.75, 3.5, 2.48, 'Training engine', 'FSDP2 / Diffusers\nVeOmni backend\nMegatron-Core paths\nActor + reference model', C.purple, { kicker: 'POLICY UPDATE', titleSize: 15, bodySize: 10.5 });
  addCard(slide, 4.5, 2.75, 4.1, 2.48, 'Rollout engine', 'Agent loop\nDiffusion / omni single-turn\nvLLM-Omni server replicas\nRouting · batching · cache', C.green, { kicker: 'HIGH-THROUGHPUT GENERATION', titleSize: 15, bodySize: 10.5 });
  addCard(slide, 8.82, 2.75, 3.5, 2.48, 'Reward engine', 'Rule-based rewards\nVLM / OCR / preference models\nMultiRewardLoop\nAsync reward workers', C.orange, { kicker: 'MULTI-REWARD SERVING', titleSize: 15, bodySize: 10.5 });
  addArrow(slide, 4.3, 3.98, 4.46, 3.98, C.faint, 1.2);
  addArrow(slide, 8.62, 3.98, 8.78, 3.98, C.faint, 1.2);
  addRect(slide, 1.0, 5.64, 11.32, 0.46, C.panel2, C.line);
  addText(slide, 'TransferQueue / Ray RPC  ·  CheckpointEngine (broadcast / P2P)', 1.18, 5.76, 10.96, 0.2, { fontFace: 'Consolas', fontSize: 9, color: C.text, align: 'center' });
  addFooter(slide, 'Source: VeRL-Omni June 2026 slides · current trainer and rollout documentation.');
}

// 15 — Efficient rollout + async reward
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'HIDE THE IDLE TIME', 'Batch the rollout; stream the reward', 'Two complementary overlaps improve GPU utilization without changing the policy-update gate.');
  addRect(slide, 0.58, 1.78, 6.05, 4.75, C.panel, C.line);
  addText(slide, '01 · STEP-WISE ROLLOUT', 0.86, 2.02, 2.7, 0.25, { fontFace: 'Consolas', fontSize: 8.5, color: C.green2, bold: true });
  addText(slide, 'Async arrivals → compatible batch → out-of-order completion', 0.86, 2.38, 5.3, 0.36, { fontSize: 14.5, color: C.white, bold: true });
  const times = [['A', 0.86, C.blue], ['C', 1.7, C.purple], ['B', 2.54, C.orange], ['D', 3.38, C.cyan]];
  times.forEach((t, i) => {
    addNode(slide, t[1], 3.08, 0.62, 0.5, t[0], '', t[2], { titleSize: 9 });
    if (i < times.length - 1) addArrow(slide, t[1] + 0.64, 3.33, t[1] + 0.78, 3.33, C.faint, 0.8);
  });
  addArrow(slide, 4.04, 3.33, 4.54, 3.33, C.green);
  addNode(slide, 4.56, 2.86, 1.54, 0.94, 'SCHEDULER', 'same shape · CFG', C.green, { titleSize: 9.5, subSize: 7 });
  addText(slide, 'Reported source-deck result', 0.86, 4.32, 2.4, 0.24, { fontFace: 'Consolas', fontSize: 7.5, color: C.muted });
  addText(slide, '20–25%', 0.86, 4.56, 2.1, 0.58, { fontSize: 27, bold: true, color: C.green });
  addText(slide, 'lower rollout generation time', 2.3, 4.72, 3.42, 0.3, { fontSize: 11.5, color: C.white, bold: true });
  addText(slide, 'Attributed to larger effective batches and step-wise execution; validate on the target workload.', 0.86, 5.32, 5.2, 0.52, { fontSize: 8.8, color: C.muted });

  addRect(slide, 6.82, 1.78, 5.94, 4.75, C.panel, C.line);
  addText(slide, '02 · ASYNC REWARD', 7.1, 2.02, 2.4, 0.25, { fontFace: 'Consolas', fontSize: 8.5, color: C.orange, bold: true });
  addText(slide, 'Completed samples flow to reward workers immediately', 7.1, 2.38, 5.1, 0.36, { fontSize: 14.5, color: C.white, bold: true });
  const y0 = 3.05;
  for (let i = 0; i < 4; i++) {
    addText(slide, `GPU ${i}`, 7.12, y0 + i * 0.42, 0.52, 0.18, { fontFace: 'Consolas', fontSize: 6.7, color: C.muted });
    slide.addShape(S.roundRect, { x: 7.72, y: y0 + i * 0.42, w: 2.4 - i * 0.12, h: 0.22, rectRadius: 0.03, fill: { color: C.blue, transparency: 25 }, line: { color: C.blue, width: 0.5 } });
    addArrow(slide, 9.72 - i * 0.05, y0 + 0.1 + i * 0.42, 10.3 + i * 0.12, 4.85, C.faint, 0.7);
  }
  addText(slide, 'REWARD', 7.08, 4.8, 0.62, 0.18, { fontFace: 'Consolas', fontSize: 6.7, color: C.muted });
  for (let i = 0; i < 4; i++) {
    slide.addShape(S.rect, { x: 10.12 + i * 0.46, y: 4.76, w: 0.42, h: 0.24, fill: { color: C.orange, transparency: 18 }, line: { color: C.orange, width: 0.5 } });
    addText(slide, `r${i}`, 10.12 + i * 0.46, 4.77, 0.42, 0.2, { fontFace: 'Consolas', fontSize: 6.7, color: C.bg, bold: true, align: 'center' });
  }
  addText(slide, 'Reward latency is hidden behind the remaining rollout wave; actor update still waits for the scored batch.', 7.12, 5.43, 5.1, 0.46, { fontSize: 8.8, color: C.muted });
  addFooter(slide, 'Source: VeRL-Omni June 2026 slides 16 and 20 · current async reward documentation.');
}

// 16 — Resource topology
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'SCALE THE TRAINING STEP', 'Resource topology is an algorithmic choice', 'Colocation saves capacity; separation reduces interference; checkpoint movement couples both decisions.');
  addText(slide, 'Candidate colocated topology', 0.62, 1.82, 3.2, 0.27, { fontFace: 'Consolas', fontSize: 8.5, color: C.muted, bold: true });
  const gpuX = [0.72, 2.2, 3.68, 5.16];
  gpuX.forEach((x, i) => {
    addRect(slide, x, 2.28, 1.28, 2.45, C.panel, C.line);
    addText(slide, `GPU ${i}`, x + 0.12, 2.44, 1.04, 0.22, { fontFace: 'Consolas', fontSize: 8, color: C.green2, bold: true, align: 'center' });
    addNode(slide, x + 0.12, 2.86, 1.04, 0.58, 'ACTOR', 'FSDP2 shard', C.purple, { titleSize: 8.5, subSize: 6.4 });
    addNode(slide, x + 0.12, 3.57, 1.04, 0.58, 'ROLLOUT', 'sleep / resume', C.green, { titleSize: 8.5, subSize: 6.4 });
  });
  addNode(slide, 2.13, 5.1, 2.52, 0.68, 'CHECKPOINT ENGINE', 'broadcast / P2P / NCCL', C.cyan, { titleSize: 9.5, subSize: 6.8 });
  gpuX.forEach((x) => addArrow(slide, 3.39, 5.08, x + 0.64, 4.78, C.faint, 0.65));

  addText(slide, 'System levers', 6.86, 1.82, 2.1, 0.27, { fontFace: 'Consolas', fontSize: 8.5, color: C.muted, bold: true });
  const levers = [
    ['Training', 'FSDP2 · VeOmni · TP / USP / DP', C.purple],
    ['Rollout', 'replicas · routing · cache · sleep / resume', C.green],
    ['Reward', 'dedicated pool · async multi-reward serving', C.orange],
    ['Weights', 'broadcast / P2P · update cadence · staleness', C.cyan],
    ['Profiling', 'actor phase + rollout servers + reward servers', C.blue],
  ];
  levers.forEach((l, i) => addCard(slide, 6.86, 2.24 + i * 0.78, 5.78, 0.63, l[0], l[1], l[2], { titleSize: 10.5, bodySize: 8.5 }));
  addFooter(slide, 'Source: VeRL-Omni README · trainer/config/profiler documentation · June 2026 distributed-training slides.');
}

// 17 — Qwen3 case
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'INTEGRATED RECIPE', 'Qwen3-Omni GSPO connects both projects end to end', 'One model provides the cleanest bridge between multimodal serving and post-training.');
  addNode(slide, 0.72, 2.42, 1.72, 0.92, 'DATA', 'text · image\naudio · video', C.blue);
  addArrow(slide, 2.46, 2.88, 2.94, 2.88, C.faint);
  addNode(slide, 2.96, 2.22, 2.08, 1.32, 'THINKER ACTOR', 'Qwen3-Omni 30B-A3B\nLoRA · FSDP2', C.purple, { titleSize: 13 });
  addArrow(slide, 5.06, 2.88, 5.54, 2.88, C.green);
  addNode(slide, 5.56, 2.22, 2.08, 1.32, 'ROLLOUT', 'vLLM-Omni TP=2\nasync replicas', C.green, { titleSize: 13 });
  addArrow(slide, 7.66, 2.88, 8.14, 2.88, C.orange);
  addNode(slide, 8.16, 2.22, 2.08, 1.32, 'REWARD', 'task / format /\nmultimodal scorer', C.orange, { titleSize: 13 });
  addArrow(slide, 10.26, 2.88, 10.74, 2.88, C.cyan);
  addNode(slide, 10.76, 2.22, 1.86, 1.32, 'GSPO', 'advantage\n+ update', C.cyan, { titleSize: 13 });
  addArrow(slide, 11.68, 3.58, 11.68, 4.2, C.cyan);
  addArrow(slide, 11.68, 4.2, 4.02, 4.2, C.cyan);
  addArrow(slide, 4.02, 4.2, 4.02, 3.58, C.cyan);
  addText(slide, 'updated weights', 7.45, 4.06, 1.56, 0.2, { fontFace: 'Consolas', fontSize: 7.2, color: C.cyan, align: 'center' });

  addMetric(slide, 0.72, 4.72, 3.72, '4×', 'REFERENCE GPU TOPOLOGY', 'H100 / H200 80 GB', C.green);
  addMetric(slide, 4.62, 4.72, 3.72, 'TP=2', 'ROLLOUT PLACEMENT', 'colocated with FSDP2 actor', C.cyan);
  addMetric(slide, 8.52, 4.72, 3.72, '3', 'INPUT RECIPES', 'text · image · text+image+audio', C.orange);
  addText(slide, 'The current recipe trains the Thinker while stripping inference-only components and typically freezing media encoders.', 0.8, 6.38, 11.8, 0.28, { fontSize: 10.2, color: C.muted, align: 'center' });
  addFooter(slide, 'Source: VeRL-Omni Qwen3-Omni GSPO documentation · supported-model guide · updated August 2026.');
}

// 18 — Case-study divider
sectionSlide('PART III · CASE STUDY', 'MiniMax H3', 'From Day-0 inference to audio-video RL on one open systems stack.', '03', C.green);

// 19 — H3 Day-0 inference
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'CASE STUDY · DAY-0', 'Turn a 134 GiB audio-video model into a service', 'The hard part was not loading weights; it was translating the model contract into a reliable serving contract.');
  addCard(slide, 0.58, 1.72, 3.82, 1.22, '≈134 GiB BF16', 'Per task partition; the release carries two task-specific DiT partitions.', C.green, { kicker: 'WEIGHT FOOTPRINT', bodySize: 9.2 });
  addCard(slide, 4.76, 1.72, 3.82, 1.22, '33B DiT + 32B encoder', 'A Qwen3-VL text encoder, joint video/audio denoiser, and two VAEs per replica.', C.blue, { kicker: 'HETEROGENEOUS PIPELINE', bodySize: 9.2 });
  addCard(slide, 8.94, 1.72, 3.82, 1.22, '50 denoising steps', 'Long packed multimodal sequences make attention, memory, and decode policy first-class.', C.orange, { kicker: 'ITERATIVE WORK', bodySize: 9.2 });

  addText(slide, 'Task-routed runtime', 0.62, 3.28, 2.4, 0.25, { fontFace: 'Consolas', fontSize: 8.2, color: C.green2, bold: true });
  addNode(slide, 0.7, 3.7, 1.72, 0.86, 'REQUEST', 'T2VA · FL2VA · Ref2VA', C.blue, { titleSize: 10.5, subSize: 7.4 });
  addArrow(slide, 2.44, 4.13, 2.84, 4.13, C.faint);
  addNode(slide, 2.86, 3.6, 2.14, 1.06, 'SHARED ENCODER', 'Qwen3-VL · TP', C.orange, { titleSize: 11.5 });
  addArrow(slide, 5.02, 4.13, 5.42, 4.13, C.faint);
  addNode(slide, 5.44, 3.52, 2.4, 1.22, 'TASK DiT', 'FL2VA / Ref2VA · SP\nvarlen TRTLLM attention', C.purple, { titleSize: 12.5, subSize: 7.6 });
  addArrow(slide, 7.86, 4.13, 8.26, 4.13, C.faint);
  addNode(slide, 8.28, 3.6, 2.14, 1.06, 'VIDEO + AUDIO VAE', 'tiling · patch parallel', C.cyan, { titleSize: 10.5 });
  addArrow(slide, 10.44, 4.13, 10.84, 4.13, C.green);
  addNode(slide, 10.86, 3.7, 1.72, 0.86, 'MP4 STREAM', 'H.264 + stereo audio', C.green, { titleSize: 10.5, subSize: 7.4 });

  slide.addShape(S.rect, { x: 0.6, y: 5.38, w: 12.15, h: 0.8, fill: { color: C.green }, line: { color: C.green } });
  addText(slide, 'DAY-0 GATE', 0.86, 5.56, 1.2, 0.22, { fontFace: 'Consolas', fontSize: 8, color: C.bg, bold: true });
  addText(slide, '1× H200 + CPU offload: T2VA, FL2VA, and Ref2VA pass end-to-end with byte-level stream checks.', 2.02, 5.48, 10.3, 0.36, { fontSize: 12.2, color: C.bg, bold: true });
  addFooter(slide, 'Source: supplied “MiniMax-H3 on vLLM-Omni” PDF, pp. 4–12 · vLLM-Omni PR #5691 validation.');
}

// 20 — H3 inference evidence
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'CASE STUDY · INFERENCE', 'One model, three deployment questions', 'These are separate validation tracks—not a cross-hardware benchmark. Each answers a different systems question.');
  addCard(slide, 0.58, 1.72, 3.82, 3.78, 'Can it run correctly?', '1× H200 · CPU offload\n\nT2VA / FL2VA / Ref2VA\nbyte-level stream checks\n\nResult: all three task paths pass end to end.', C.green, { kicker: 'DAY-0 VALIDATION', titleSize: 16, bodySize: 10.3 });
  addCard(slide, 4.76, 1.72, 3.82, 3.78, 'Can Blackwell run it faster?', '4× B300 · matched seed / recipe\n1248×768 · 209 frames · 50 steps\n\nDiffusion: 83.85 → 71.99 s  (−14.2%)\nEnd to end: 88.56 → 76.18 s  (−14.0%)\n\nOnly attention backend changes: FA4 → TRTLLM_ATTN.', C.blue, { kicker: 'BACKEND A/B', titleSize: 16, bodySize: 9.4 });
  addCard(slide, 8.94, 1.72, 3.82, 3.78, 'Can it escape datacenter HBM?', '2× RTX 5090 · TP2 + rank-local DLO\n1344×768 · 124 frames · 50 steps\n\n8 min 38 s · no OOM\n≈22.6 GiB peak / GPU\n\n20 DiT blocks resident; 30 streamed.', C.orange, { kicker: 'CONSUMER-GPU PATH', titleSize: 16, bodySize: 9.6 });
  addText(slide, 'Engineering pattern', 0.66, 5.92, 1.55, 0.2, { fontFace: 'Consolas', fontSize: 7.6, color: C.green2, bold: true });
  addText(slide, 'Correctness first → isolate the hot backend → redesign memory movement for the target topology.', 2.2, 5.82, 10.08, 0.34, { fontSize: 12.7, color: C.white, bold: true });
  addFooter(slide, 'Source: supplied H3 PDF, pp. 10, 12, 23–31 · each result retains its original benchmark scope.');
}

// 21 — H3 RL data plane
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'CASE STUDY · RL', 'The serving pipeline becomes the policy rollout engine', 'VeRL-Omni keeps model-specific trajectory semantics while reusing one trainer, reward, and weight-sync control plane.');
  addNode(slide, 0.62, 2.12, 1.54, 0.86, 'PROMPT', 'text + optional keyframe', C.blue, { titleSize: 10.5, subSize: 7.2 });
  addArrow(slide, 2.18, 2.55, 2.56, 2.55, C.faint);
  addNode(slide, 2.58, 1.98, 2.02, 1.14, 'vLLM-OMNI', 'H3 rollout\nvideo + audio state', C.green, { titleSize: 12.5, subSize: 7.8 });
  addArrow(slide, 4.62, 2.55, 5.0, 2.55, C.faint);
  addNode(slide, 5.02, 1.98, 2.02, 1.14, 'REWARD', 'CLAP + ImageBind\nasync multi-reward', C.orange, { titleSize: 12.5, subSize: 7.8 });
  addArrow(slide, 7.06, 2.55, 7.44, 2.55, C.faint);
  addNode(slide, 7.46, 1.98, 2.02, 1.14, 'OBJECTIVE', 'FlowGRPO or\nDiffusionNFT', C.cyan, { titleSize: 12.5, subSize: 7.8 });
  addArrow(slide, 9.5, 2.55, 9.88, 2.55, C.faint);
  addNode(slide, 9.9, 1.98, 2.02, 1.14, 'FSDP2 ACTOR', 'rank-64 LoRA\ndiffusers weights', C.purple, { titleSize: 12.5, subSize: 7.8 });
  addArrow(slide, 10.92, 3.14, 10.92, 3.48, C.cyan);
  addArrow(slide, 10.92, 3.48, 3.6, 3.48, C.cyan);
  addArrow(slide, 3.6, 3.48, 3.6, 3.14, C.cyan);
  addText(slide, 'LoRA tensor payload · rollout weight refresh', 6.05, 3.32, 2.72, 0.2, { fontFace: 'Consolas', fontSize: 6.8, color: C.cyan, align: 'center' });

  addCard(slide, 0.62, 4.12, 5.86, 1.66, 'FlowGRPO', 'Reverse-SDE / CPS trajectory · separate video and audio schedules · rollout/replay log-probability contract.\nMerged 2026-08-28 · NVIDIA GPU + Ascend NPU T2VA; NVIDIA GPU FL2VA recipe.', C.green, { kicker: 'ONLINE POLICY GRADIENT', titleSize: 15, bodySize: 9.2 });
  addCard(slide, 6.76, 4.12, 5.86, 1.66, 'DiffusionNFT', 'Score the clean audio-video sample, forward-noise the selected target, then update without reverse-process likelihood.\nMerged 2026-08-21 · T2VA and FL2VA rank-64 LoRA recipes.', C.purple, { kicker: 'FORWARD-PROCESS RL', titleSize: 15, bodySize: 9.2 });
  addText(slide, 'The invariant is exact replay: prompt IDs, packed rows, modality-specific schedules, and base-policy weight layouts must agree.', 1.0, 6.16, 11.3, 0.3, { fontSize: 11.4, color: C.white, bold: true, align: 'center' });
  addFooter(slide, 'Sources: VeRL-Omni main · H3 FlowGRPO PR #368 · H3 DiffusionNFT PR #383 · current H3 recipes (snapshot 2026-09-02).');
}

// 22 — H3 conclusion and discussion
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'CASE STUDY · EVIDENCE', 'H3 closes the loop—and exposes the next bottlenecks', 'Support snapshot on 2026-09-02; SUPPORTED marks implemented paths and TODO marks explicit gaps.');
  const rows = [
    ['PATH', 'T2VA', 'FL2VA', 'REF2VA', 'CORRECTNESS GATE'],
    ['vLLM-Omni inference', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'byte-level media stream + task routing'],
    ['VeRL-Omni DiffusionNFT', 'SUPPORTED', 'SUPPORTED', 'TODO', 'checkpoint parity + target-only noising'],
    ['VeRL-Omni FlowGRPO', 'GPU + NPU', 'NVIDIA GPU', 'TODO', 'dual schedules + rollout/replay consistency'],
  ];
  addTable(slide, rows, 0.58, 1.72, 12.18, 2.3, [0.22, 0.14, 0.14, 0.14, 0.36], { headerSize: 8, bodySize: 8.7 });

  addText(slide, 'Three NVIDIA-facing questions', 0.62, 4.38, 3.1, 0.26, { fontFace: 'Consolas', fontSize: 8.2, color: C.green2, bold: true });
  addCard(slide, 0.62, 4.78, 3.78, 1.18, 'Kernel consistency', 'Can Actor and rollout use the fastest Blackwell attention path without policy-ratio drift?', C.green, { bodySize: 8.6 });
  addCard(slide, 4.58, 4.78, 3.78, 1.18, 'Memory + weight movement', 'How should 33B Actor shards, rollout TP, LoRA refresh, and reward GPUs share a node?', C.blue, { bodySize: 8.6 });
  addCard(slide, 8.54, 4.78, 3.78, 1.18, 'One evidence contract', 'Measure serving JCT / memory and RL step time / reward curve under the same model revision.', C.orange, { bodySize: 8.6 });
  addText(slide, 'Case-study thesis', 0.7, 6.38, 1.4, 0.2, { fontFace: 'Consolas', fontSize: 7.5, color: C.green2, bold: true });
  addText(slide, 'The optimization target is useful GPU work per convergent training step.', 2.12, 6.28, 8.5, 0.34, { fontSize: 14, color: C.white, bold: true });
  addText(slide, 'Q & A', 10.78, 6.24, 1.52, 0.42, { fontSize: 22, color: C.white, bold: true, align: 'right' });
  addFooter(slide, 'Status derived from current official vLLM-Omni / VeRL-Omni code, recipes, and merged PRs; Ref2VA RL remains unsupported.');
}

// 23 — Appendix divider
sectionSlide('APPENDIX', 'Reference material', 'Release highlights, model patterns, parallelism, algorithms, and benchmark discipline.', 'A', C.faint);

// 24 — Release highlights
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'APPENDIX · RELEASE', 'vLLM-Omni 0.26.0: the production surface broadened', 'A concise selection from the uploaded release deck.');
  addCard(slide, 0.58, 1.82, 2.94, 1.72, 'Multimodal generation', 'MiniMax H3 serving\nadditional image / video / audio pipelines\nworld-model coverage', C.blue, { kicker: 'MODELS', bodySize: 9.5 });
  addCard(slide, 3.72, 1.82, 2.94, 1.72, 'Realtime interaction', 'Experimental MiniCPM-o duplex\nstreaming audio input / output\ncancellation + barge-in', C.green, { kicker: 'INTERACTION', bodySize: 9.5 });
  addCard(slide, 6.86, 1.82, 2.94, 1.72, 'Scale diffusion', 'DLO / CPU offload\nparallelism and batching\nquantization expansion', C.purple, { kicker: 'MEMORY + THROUGHPUT', bodySize: 9.5 });
  addCard(slide, 10.0, 1.82, 2.74, 1.72, 'Runtime alignment', 'Aligned release cadence\nserving and metrics fixes\nplatform delivery', C.orange, { kicker: 'CORE', bodySize: 9.5 });
  addText(slide, 'MiniMax H3 post-release optimization examples', 0.6, 3.98, 4.0, 0.28, { fontFace: 'Consolas', fontSize: 8.5, color: C.muted, bold: true });
  addMetric(slide, 0.6, 4.42, 3.7, '−22%', 'ONLINE FP8 DIT MEMORY', '68.52 → 53.51 GiB / GPU · 2×H100', C.green);
  addMetric(slide, 4.5, 4.42, 3.7, '1.865×', 'FUSED QK NORM + 3D ROPE', 'B300 BF16 · kernel-only · bitwise exact', C.blue);
  addMetric(slide, 8.4, 4.42, 3.7, '−11.26 GiB', 'ADALN SCHEDULE CACHE', 'per GPU steady state · 2×H20 TP2', C.orange);
  addFooter(slide, 'Source: uploaded “vLLM-Omni 0.26.0 Release” deck · figures retain their original benchmark scope.');
}

// 25 — Model patterns
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'APPENDIX · MODEL PATTERNS', 'Different model structures stress different runtime primitives', 'The stage abstraction is useful only if it preserves these differences.');
  addCard(slide, 0.58, 1.82, 3.86, 3.78, 'Qwen3-Omni', 'Thinker → Talker → Code2Wav\n\nNeeds: stage-local batching, streaming chunks, payload ordering, graph capture, selective replicas.\n\nPrimary KPI: first audio + RTF.', C.green, { kicker: 'MULTI-AR + CODEC', titleSize: 16, bodySize: 10 });
  addCard(slide, 4.73, 1.82, 3.86, 3.78, 'BAGEL / unified image', 'AR reasoning → visual generator\n\nNeeds: AR–DiT handoff, CFG-aware requests, KV or hidden-state transfer, diffusion scheduler.\n\nPrimary KPI: JCT + quality.', C.orange, { kicker: 'AR + DIT', titleSize: 16, bodySize: 10 });
  addCard(slide, 8.88, 1.82, 3.86, 3.78, 'Cosmos / DreamZero', 'Causal chunked world generation\n\nNeeds: persistent diffusion state, KV-aware chunking, streaming output, long sequence parallelism.\n\nPrimary KPI: chunk latency + memory.', C.purple, { kicker: 'STATEFUL WORLD MODEL', titleSize: 16, bodySize: 10 });
  addText(slide, 'Design lesson', 0.72, 6.18, 1.0, 0.2, { fontFace: 'Consolas', fontSize: 7.5, color: C.green2, bold: true });
  addText(slide, 'Build reusable primitives around stages and connectors; avoid one vertical engine per model.', 1.8, 6.09, 10.4, 0.35, { fontSize: 12.5, color: C.white, bold: true });
  addFooter(slide, 'Sources: vLLM-Omni architecture overview · uploaded architecture-evolution and release decks.');
}

// 26 — Parallelism guide
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'APPENDIX · PARALLELISM', 'Choose the axis by bottleneck—then verify', 'Parallelism methods are not interchangeable and may compose only under model-specific constraints.');
  const rows = [
    ['BOTTLENECK', 'METHOD', 'TYPICAL SCOPE', 'VALIDATION QUESTION'],
    ['DiT weights / compute', 'Tensor parallel', 'transformer blocks', 'does communication erase the GEMM gain?'],
    ['Full transformer memory', 'HSDP / FSDP2', 'parameter shards + replicas', 'is offload or gather on the critical path?'],
    ['Long visual sequence', 'Ulysses / Ring SP', 'attention sequence', 'is attention backend compatible?'],
    ['Positive / negative branches', 'CFG parallel', 'guidance branches', 'are shapes and LoRA identical?'],
    ['MoE expert weights', 'Expert parallel', 'experts', 'is routing balanced and group size > 1?'],
    ['VAE activation peak', 'VAE patch / tile', 'decode / encode', 'does seam-free quality remain unchanged?'],
    ['Stage saturation', 'StagePool replicas', 'whole stage instances', 'is the stage truly the service bottleneck?'],
  ];
  addTable(slide, rows, 0.58, 1.82, 12.2, 4.84, [0.23, 0.19, 0.25, 0.33], { headerSize: 8, bodySize: 8.3 });
  addFooter(slide, 'Source: vLLM-Omni v0.26.0 diffusion parallelism guides · Qwen3-Omni replica case study.');
}

// 27 — Algorithm and model map
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'APPENDIX · VERL-OMNI', 'Model families map to different RL objectives', 'Keep the main talk system-focused; use this matrix for algorithm questions.');
  const rows = [
    ['MODEL FAMILY', 'EXAMPLES', 'ALGORITHM FAMILY', 'ROLLOUT CHARACTER'],
    ['Image diffusion', 'Qwen-Image · SD3.5', 'FlowGRPO · DiffusionNFT · DPO', 'multi-step denoise + visual reward'],
    ['Video / audio diffusion', 'Wan2.2 · MiniMax H3 · LTX', 'FlowGRPO · DiffusionNFT', 'long trajectory · heavy media materialization'],
    ['Unified understand + gen', 'BAGEL · HunyuanImage', 'FlowGRPO · MixGRPO', 'AR + generator stage graph'],
    ['Omni reasoning', 'Qwen3-Omni Thinker', 'GSPO · DPO', 'multimodal input · text policy output'],
  ];
  addTable(slide, rows, 0.58, 1.82, 12.18, 3.16, [0.21, 0.25, 0.27, 0.27], { headerSize: 8.1, bodySize: 8.7 });
  addCard(slide, 0.62, 5.34, 3.78, 1.18, 'Online policy gradient', 'Rollout → reward → advantage → actor update.', C.green, { bodySize: 8.8 });
  addCard(slide, 4.58, 5.34, 3.78, 1.18, 'Direct preference', 'Paired / scored samples with single-step preference updates.', C.orange, { bodySize: 8.8 });
  addCard(slide, 8.54, 5.34, 3.78, 1.18, 'System invariant', 'Algorithm choice must not leak into model adapters or rollout plumbing.', C.cyan, { bodySize: 8.8 });
  addFooter(slide, 'Source: VeRL-Omni README · trainer API · supported-model guide · August 2026.');
}

// 28 — Benchmark checklist and sources
{
  const slide = pptx.addSlide('DARK');
  addTitle(slide, 'APPENDIX · EVIDENCE', 'A benchmark slide is a reproducibility contract', 'Keep serving readiness separate from unavailable hardware or pending quality validation.');
  const checks = [
    ['SOFTWARE', 'repo SHA · release · backend · dependency versions'],
    ['HARDWARE', 'GPU SKU · count · interconnect · clocks / power mode'],
    ['WORKLOAD', 'model · prompt / media shape · steps · batch · concurrency'],
    ['PRECISION', 'dtype · quantized components · accuracy / quality check'],
    ['PROCEDURE', 'warmups · repetitions · restart policy · profiler range'],
    ['METRICS', 'TTFT / TTFP / RTF / JCT · memory · samples / card / s'],
  ];
  checks.forEach((c, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    addCard(slide, 0.58 + col * 6.16, 1.78 + row * 1.12, 5.84, 0.88, c[0], c[1], i % 2 ? C.cyan : C.green, { titleSize: 10.5, bodySize: 8.3 });
  });
  addText(slide, 'Primary source set', 0.62, 5.42, 2.0, 0.24, { fontFace: 'Consolas', fontSize: 8.2, color: C.green2, bold: true });
  addText(slide, 'docs.vllm.ai/projects/vllm-omni  ·  github.com/vllm-project/vllm-omni  ·  verl-omni.readthedocs.io  ·  github.com/verl-project/verl-omni', 0.62, 5.82, 12.02, 0.34, { fontFace: 'Consolas', fontSize: 8.3, color: C.text });
  addText(slide, 'Uploaded source decks remain unchanged in scripts/. This integrated deck is a new, editable derivative.', 0.62, 6.36, 11.5, 0.28, { fontSize: 9.3, color: C.muted });
  addFooter(slide, 'Evidence checklist aligned with the LSY workspace pre-submit rules and source-deck benchmark footnotes.');
}

// Basic notes for the speaker.
pptx._slides.forEach((slide, i) => {
  if (typeof slide.addNotes === 'function') {
    slide.addNotes(`Slide ${i + 1}: keep the discussion focused on the system claim shown on the slide. Distinguish published measurements from proposed collaboration areas.`);
  }
});

pptx.writeFile({ fileName: OUTPUT });

const fs = require('fs');
const fsp = require('fs/promises');
const os = require('os');
const path = require('path');

const JSZip = require('jszip');
const { DOMParser, XMLSerializer } = require('@xmldom/xmldom');
const { Automizer } = require('pptx-automizer');

const SCRIPT_DIR = __dirname;
const OUTPUT = 'vLLM-Omni_VeRL-Omni_All_Source_Slides_Merged_Editable.pptx';
const TARGET_SIZE = { cx: 12192000, cy: 6858000 }; // 13.333 × 7.5 in, 16:9

const SOURCES = [
  { file: 'vLLM-Omni 0.26.0 Release.pptx', label: 'release' },
  { file: 'vLLM-Omni Slides (Public) 2026-04 latest (1).pptx', label: 'public' },
  { file: 'vLLM-Omni-架构演化.pptx', label: 'architecture' },
  { file: 'verl-omni_slides_20260627.pptx', label: 'verl' },
];

const LENGTH_ATTRS = new Map([
  ['off', ['x', 'y']],
  ['ext', ['cx', 'cy']],
  ['chOff', ['x', 'y']],
  ['chExt', ['cx', 'cy']],
  ['gridCol', ['w']],
  ['tr', ['h']],
  ['bodyPr', ['lIns', 'rIns', 'tIns', 'bIns']],
  ['pPr', ['marL', 'marR', 'indent', 'defTabSz']],
  ['lvl1pPr', ['marL', 'marR', 'indent', 'defTabSz']],
  ['lvl2pPr', ['marL', 'marR', 'indent', 'defTabSz']],
  ['lvl3pPr', ['marL', 'marR', 'indent', 'defTabSz']],
  ['lvl4pPr', ['marL', 'marR', 'indent', 'defTabSz']],
  ['lvl5pPr', ['marL', 'marR', 'indent', 'defTabSz']],
  ['lvl6pPr', ['marL', 'marR', 'indent', 'defTabSz']],
  ['lvl7pPr', ['marL', 'marR', 'indent', 'defTabSz']],
  ['lvl8pPr', ['marL', 'marR', 'indent', 'defTabSz']],
  ['lvl9pPr', ['marL', 'marR', 'indent', 'defTabSz']],
  ['defPPr', ['marL', 'marR', 'indent', 'defTabSz']],
  ['ln', ['w']],
  ['tab', ['pos']],
  ['spcPts', ['val']],
  ['buSzPts', ['val']],
  ['outerShdw', ['blurRad', 'dist']],
  ['innerShdw', ['blurRad', 'dist']],
  ['glow', ['rad']],
  ['softEdge', ['rad']],
  ['rPr', ['sz']],
  ['defRPr', ['sz']],
  ['endParaRPr', ['sz']],
]);

function localName(node) {
  return node.localName || node.nodeName.split(':').pop();
}

function scaleIntegerAttribute(node, name, factor) {
  if (!node.hasAttribute(name)) return;
  const value = Number(node.getAttribute(name));
  if (!Number.isFinite(value)) return;
  node.setAttribute(name, String(Math.round(value * factor)));
}

function scaleXml(xml, factor, isPresentationXml) {
  const document = new DOMParser().parseFromString(xml, 'application/xml');
  const nodes = document.getElementsByTagName('*');

  for (let index = 0; index < nodes.length; index += 1) {
    const node = nodes.item(index);
    const name = localName(node);
    const attrs = LENGTH_ATTRS.get(name);
    if (attrs) attrs.forEach((attr) => scaleIntegerAttribute(node, attr, factor));

    if (isPresentationXml && name === 'sldSz') {
      node.setAttribute('cx', String(TARGET_SIZE.cx));
      node.setAttribute('cy', String(TARGET_SIZE.cy));
    }
  }

  return new XMLSerializer().serializeToString(document);
}

async function getPresentationSize(zip) {
  const xml = await zip.file('ppt/presentation.xml').async('string');
  const document = new DOMParser().parseFromString(xml, 'application/xml');
  const nodes = document.getElementsByTagName('*');
  for (let index = 0; index < nodes.length; index += 1) {
    const node = nodes.item(index);
    if (localName(node) === 'sldSz') {
      return { cx: Number(node.getAttribute('cx')), cy: Number(node.getAttribute('cy')) };
    }
  }
  throw new Error('ppt/presentation.xml does not define p:sldSz');
}

async function normalizePresentationSize(sourcePath, outputPath) {
  const input = await fsp.readFile(sourcePath);
  const zip = await JSZip.loadAsync(input);
  const sourceSize = await getPresentationSize(zip);

  if (sourceSize.cx === TARGET_SIZE.cx && sourceSize.cy === TARGET_SIZE.cy) {
    await fsp.copyFile(sourcePath, outputPath);
    return;
  }

  const factorX = TARGET_SIZE.cx / sourceSize.cx;
  const factorY = TARGET_SIZE.cy / sourceSize.cy;
  if (Math.abs(factorX - factorY) > 1e-9) {
    throw new Error(`Cannot uniformly scale ${path.basename(sourcePath)}: ${factorX} × ${factorY}`);
  }

  const xmlFiles = Object.keys(zip.files).filter(
    (name) => name.startsWith('ppt/') && name.endsWith('.xml'),
  );
  for (const name of xmlFiles) {
    const xml = await zip.file(name).async('string');
    zip.file(name, scaleXml(xml, factorX, name === 'ppt/presentation.xml'));
  }

  const output = await zip.generateAsync({
    type: 'nodebuffer',
    compression: 'DEFLATE',
    compressionOptions: { level: 6 },
  });
  await fsp.writeFile(outputPath, output);
}

async function main() {
  const workDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'vllm-omni-ppt-merge-'));
  try {
    for (const source of SOURCES) {
      const sourcePath = path.join(SCRIPT_DIR, source.file);
      if (!fs.existsSync(sourcePath)) throw new Error(`Missing source deck: ${sourcePath}`);
      source.normalizedFile = `${source.label}.pptx`;
      await normalizePresentationSize(sourcePath, path.join(workDir, source.normalizedFile));
    }

    const automizer = new Automizer({
      templateDir: workDir,
      outputDir: SCRIPT_DIR,
      removeExistingSlides: true,
      autoImportSlideMasters: true,
      cleanup: false,
      compression: 6,
      verbosity: 1,
    });

    let presentation = automizer.loadRoot('architecture.pptx');
    for (const source of SOURCES) {
      presentation = presentation.load(source.normalizedFile, source.label);
    }

    let expectedSlides = 0;
    for (const source of SOURCES) {
      const slideNumbers = await presentation
        .getTemplate(source.label)
        .getAllSlideNumbers();
      expectedSlides += slideNumbers.length;
      for (const slideNumber of slideNumbers) {
        presentation.addSlide(source.label, slideNumber);
      }
    }

    const summary = await presentation.write(OUTPUT);
    if (summary.slides !== expectedSlides) {
      throw new Error(`Expected ${expectedSlides} slides, wrote ${summary.slides}`);
    }
    console.log(`Wrote ${summary.slides} slides to ${path.join(SCRIPT_DIR, OUTPUT)}`);
  } finally {
    await fsp.rm(workDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

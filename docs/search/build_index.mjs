// Embed the documentation chunks, sentences and pages that `search_chunks.py` wrote, and publish the browser runtime.
//
//   node build_index.mjs <html-dir>
//
// The browser loads the same model, revision and dtype from the Hugging Face Hub, so query and
// chunk vectors share one space. Vectors are truncated (Matryoshka), renormalized, then stored
// as int8 with one float32 scale per vector. The storage format is not a model format.
import fs from 'node:fs';
import path from 'node:path';
import { AutoConfig, env, pipeline } from '@huggingface/transformers';
import { embeddingTexts } from '../source/_static/search-core.js';

const MODEL = 'onnx-community/embeddinggemma-2-ONNX';
const REVISION = 'daa72c51243991dfcaf9f9137d2c573d8f7790c0';
const DTYPE = 'q8';
const DIM = 512;
const QUERY_PREFIX = 'task: search result | query: ';
const BATCH = 4;

const htmlDir = process.argv[2];
if (!htmlDir) throw new Error('usage: node build_index.mjs <html-dir>');
const staticDir = path.join(htmlDir, '_static');
if (process.env.HF_CACHE_DIR) env.cacheDir = process.env.HF_CACHE_DIR;

const chunks = JSON.parse(fs.readFileSync(path.join(staticDir, 'search-chunks.json'), 'utf8'));
const { chunkTexts, sentenceTexts, pageTexts } = embeddingTexts(chunks);
// Vector file order: chunks, sentences, pages. `search-core.js` reads the same order.
const texts = [...chunkTexts, ...sentenceTexts, ...pageTexts];

// The checkpoint is multimodal. Without vision and audio configs, only the text model loads.
const config = await AutoConfig.from_pretrained(MODEL, { revision: REVISION });
delete config.vision_config;
delete config.audio_config;
const extractor = await pipeline('feature-extraction', MODEL, {
  device: 'cpu', dtype: DTYPE, revision: REVISION, config,
});

// Sort by length so that a batch pads little, then restore the order.
const order = texts.map((_, i) => i).sort((a, b) => texts[a].length - texts[b].length);
const codes = new Int8Array(texts.length * DIM);
const scales = new Float32Array(texts.length);
for (let i = 0; i < order.length; i += BATCH) {
  const ids = order.slice(i, i + BATCH);
  const out = await extractor(ids.map((j) => texts[j]), { pooling: 'mean', normalize: true });
  const width = out.dims[1];
  ids.forEach((id, row) => {
    const v = Array.from(out.data.subarray(row * width, row * width + DIM));
    const norm = Math.hypot(...v);
    const step = Math.max(...v.map(Math.abs)) / norm / 127;
    scales[id] = step;
    v.forEach((x, j) => { codes[id * DIM + j] = Math.round(x / norm / step); });
  });
}
fs.writeFileSync(
  path.join(staticDir, 'search-vectors.bin'),
  Buffer.concat([Buffer.from(scales.buffer), Buffer.from(codes.buffer)]),
);

// Self-host the runtime. The version that embedded the chunks is the version that serves them.
const modules = path.join(import.meta.dirname, 'node_modules');
const pkgDir = path.join(modules, '@huggingface', 'transformers');
const ortDir = path.join(modules, 'onnxruntime-web');
const runtimeDir = path.join(staticDir, 'search');
fs.mkdirSync(runtimeDir, { recursive: true });
fs.copyFileSync(path.join(pkgDir, 'dist', 'transformers.min.js'), path.join(runtimeDir, 'transformers.min.js'));
// WebGPU runs on the asyncify build. The CPU (WASM) backend runs on the JSEP build.
for (const build of ['asyncify', 'jsep']) {
  for (const name of [`ort-wasm-simd-threaded.${build}.mjs`, `ort-wasm-simd-threaded.${build}.wasm`]) {
    fs.copyFileSync(path.join(ortDir, 'dist', name), path.join(runtimeDir, name));
  }
}

// Ship the license texts of the third-party runtime with it.
const licenseDir = path.join(runtimeDir, 'licenses');
fs.mkdirSync(licenseDir, { recursive: true });
for (const name of fs.readdirSync(path.join(import.meta.dirname, 'licenses'))) {
  fs.copyFileSync(path.join(import.meta.dirname, 'licenses', name), path.join(licenseDir, name));
}

fs.writeFileSync(path.join(staticDir, 'search-meta.json'), JSON.stringify({
  model: MODEL, revision: REVISION, dtype: DTYPE, dim: DIM,
  query_prefix: QUERY_PREFIX,
}));
console.log(`Embedded ${chunkTexts.length} chunks, ${sentenceTexts.length} sentences and ${pageTexts.length} pages (${DTYPE}, ${DIM}-d, int8 storage).`);

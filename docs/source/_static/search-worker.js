/*
 * Web Worker for the semantic search: it loads EmbeddingGemma 2 (text tower) and embeds queries,
 * so that typing never blocks the page. Messages in: {type: 'load', meta}, {type: 'embed', id, text}.
 * Messages out: {type: 'progress', loaded, total}, {type: 'ready', backend}, {type: 'error', message},
 * {type: 'vector', id, vector}.
 */
const runtimeBase = new URL('search/', self.location.href).href;
let extractor = null;
let latest = null;
let busy = false;

// The checkpoint lists a vision and an audio encoder that this page never loads.
// Count only the files that download for text.
const downloads = new Map();
function onProgress(info) {
  if (info.status !== 'progress' || /vision_encoder|audio_encoder/.test(info.file)) return;
  downloads.set(info.file, info);
  let loaded = 0;
  let total = 0;
  for (const d of downloads.values()) { loaded += d.loaded; total += d.total; }
  self.postMessage({ type: 'progress', loaded, total });
}

async function load(meta) {
  const tf = await import(runtimeBase + 'transformers.min.js');
  tf.env.allowLocalModels = false;
  // ONNX Runtime Web ships several WASM builds. WebGPU needs the asyncify build. The CPU
  // backend needs the JSEP build: the asyncify build lacks the GatherBlockQuantized kernel
  // that the quantized checkpoints use.
  const runtime = (build) => {
    tf.env.backends.onnx.wasm.wasmPaths = {
      mjs: `${runtimeBase}ort-wasm-simd-threaded.${build}.mjs`,
      wasm: `${runtimeBase}ort-wasm-simd-threaded.${build}.wasm`,
    };
  };
  // The checkpoint is multimodal. Without these configs, only the text model downloads.
  const config = await tf.AutoConfig.from_pretrained(meta.model, { revision: meta.revision });
  delete config.vision_config;
  delete config.audio_config;
  const options = (device) => ({
    device, dtype: meta.dtype, revision: meta.revision, config, progress_callback: onProgress,
  });
  let adapter = null;
  try { adapter = self.navigator.gpu ? await self.navigator.gpu.requestAdapter() : null; } catch (e) { /* no WebGPU */ }
  let backend = 'WASM';
  if (adapter) {
    try {
      runtime('asyncify');
      extractor = await tf.pipeline('feature-extraction', meta.model, options('webgpu'));
      backend = 'WebGPU';
    } catch (e) {
      console.warn('WebGPU failed, using WASM:', e);
    }
  }
  if (!extractor) {
    runtime('jsep');
    extractor = await tf.pipeline('feature-extraction', meta.model, options('wasm'));
  }
  self.postMessage({ type: 'ready', backend });
}

// Only the newest query matters: a query that waits while another one runs is dropped.
async function drain() {
  if (busy) return;
  busy = true;
  while (latest) {
    const { id, text } = latest;
    latest = null;
    try {
      const out = await extractor(text, { pooling: 'mean', normalize: true });
      self.postMessage({ type: 'vector', id, vector: Float32Array.from(out.data) });
    } catch (e) {
      self.postMessage({ type: 'error', id, message: String(e) });
    }
  }
  busy = false;
}

self.onmessage = async ({ data }) => {
  if (data.type === 'load') {
    try { await load(data.meta); drain(); } catch (e) { self.postMessage({ type: 'error', message: String(e) }); }
  } else if (data.type === 'embed') {
    latest = data;
    if (extractor) drain();
  }
};

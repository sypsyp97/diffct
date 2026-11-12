/*
 * Web Worker for the semantic search: it loads EmbeddingGemma 2 (text tower) and embeds queries,
 * so that typing never blocks the page. Messages in: {type: 'load', meta, forceWasm}, {type: 'embed', id, text}.
 * Messages out: {type: 'progress', loaded, total}, {type: 'ready', backend}, {type: 'fallback'}, {type: 'error', message},
 * {type: 'vector', id, vector}. An error is fatal: the worker closes after it.
 */
const runtimeBase = new URL('search/', self.location.href).href;
let extractor = null;
let latest = null;
let busy = false;
let backend = '';
let failed = false;

// A fatal error ends this worker. The page keeps keyword search and offers a retry.
function fail(error) {
  if (failed) return;
  failed = true;
  latest = null;
  extractor = null;
  try {
    self.postMessage({ type: 'error', message: String(error) });
  } finally {
    self.close();
  }
}

// transformers.js 4.3.1 can reject a model download (the external weight file) outside the
// promise that `pipeline()` returns. Without this handler the page would wait forever.
self.addEventListener('unhandledrejection', (event) => {
  event.preventDefault();
  fail(event.reason);
});

// The checkpoint lists a vision and an audio encoder that this page never loads.
// Count only the files that download for text.
const downloads = new Map();
function onProgress(info) {
  if (failed || info.status !== 'progress' || /vision_encoder|audio_encoder/.test(info.file)) return;
  if (!Number.isFinite(info.loaded) || !Number.isFinite(info.total)) return;
  downloads.set(info.file, info);
  let loaded = 0;
  let total = 0;
  for (const d of downloads.values()) { loaded += d.loaded; total += d.total; }
  self.postMessage({ type: 'progress', loaded, total });
}

async function load(meta, forceWasm) {
  const tf = await import(runtimeBase + 'transformers.min.js');
  tf.env.allowLocalModels = false;
  // transformers.js 4.3.1 probes file sizes at `main` without the revision, and loads config.json
  // twice. Point every Hub request at the pinned revision and fetch config.json once.
  const hubFetch = tf.env.fetch;
  const configs = new Map();
  tf.env.fetch = (url, init) => {
    if (typeof url !== 'string') return hubFetch(url, init);
    const pinned = url.replace(`/${meta.model}/resolve/main/`, `/${meta.model}/resolve/${meta.revision}/`);
    if (/\/(?:vision|audio)[^/]*\.onnx/.test(pinned)) throw new Error('Only the text model is permitted.');
    if (!pinned.endsWith('/config.json') || (init && init.headers && init.headers.Range)) return hubFetch(pinned, init);
    if (!configs.has(pinned)) configs.set(pinned, hubFetch(pinned, init));
    return configs.get(pinned).then((response) => response.clone());
  };
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
  try { adapter = forceWasm ? null : self.navigator.gpu ? await self.navigator.gpu.requestAdapter() : null; } catch (e) { /* no WebGPU */ }
  if (adapter) {
    try {
      runtime('asyncify');
      extractor = await tf.pipeline('feature-extraction', meta.model, options('webgpu'));
      backend = 'WebGPU';
    } catch (e) {
      if (failed) return;
      // ONNX Runtime keeps its WebGPU state in this worker, so the WASM retry needs a fresh one.
      console.warn('WebGPU failed, asking for a fresh WASM worker:', e);
      self.postMessage({ type: 'fallback' });
      return;
    }
  } else {
    runtime('jsep');
    extractor = await tf.pipeline('feature-extraction', meta.model, options('wasm'));
    backend = 'WASM';
  }
  if (!failed) self.postMessage({ type: 'ready', backend });
}

// Only the newest query matters: a query that waits while another one runs is dropped.
async function drain() {
  if (busy || !extractor) return;
  busy = true;
  try {
    while (latest) {
      const { id, text } = latest;
      latest = null;
      try {
        const out = await extractor(text, { pooling: 'mean', normalize: true });
        if (failed) return;
        if (out.dims.length !== 2 || out.dims[0] !== 1 || out.data.length !== out.dims[1]) {
          throw new Error('The query model returned an invalid embedding shape.');
        }
        const vector = Float32Array.from(out.data);
        self.postMessage({ type: 'vector', id, vector }, [vector.buffer]);
      } catch (e) {
        if (failed) return;
        if (backend === 'WebGPU') {
          // Same reason as a WebGPU load failure: retry in a fresh WASM worker.
          console.warn('WebGPU embedding failed, asking for a fresh WASM worker:', e);
          self.postMessage({ type: 'fallback' });
          return;
        }
        fail(e);
        return;
      }
    }
  } finally {
    busy = false;
  }
}

self.onmessage = async ({ data }) => {
  if (failed) return;
  if (data.type === 'load') {
    try { await load(data.meta, data.forceWasm); drain(); } catch (e) { fail(e); }
  } else if (data.type === 'embed') {
    latest = data;
    drain();
  }
};

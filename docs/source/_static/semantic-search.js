/*
 * Hybrid search for the search page: BM25 keyword ranking and EmbeddingGemma 2 semantic
 * ranking over section chunks, merged with reciprocal rank fusion (RRF).
 *
 * Keyword results show at once. The model loads on the first query. WebGPU runs the model
 * when the browser has it; WASM runs it otherwise. `docs/search/build_index.mjs` writes the
 * chunk vectors with the same model and dtype, so query and chunk vectors share one space.
 */
(function () {
  'use strict';

  const results = document.getElementById('search-results');
  const input = document.querySelector('.bd-search-container form.bd-search input[name="q"]');
  if (!results || !input) return;

  const staticBase = new URL('.', document.currentScript.src).href;
  const siteRoot = new URL('../', staticBase).href;
  const RRF_K = 60;
  const POOL = 50;
  const SHOWN = 10;
  const PER_PAGE = 2;
  const STOP = new Set(('a an and are as at be by can do does for from how i in is it of on or ' +
    'the this to use what when where which with').split(' '));

  // ---- panel ----------------------------------------------------------------------------
  const panel = document.createElement('section');
  panel.id = 'semantic-results';
  panel.innerHTML =
    '<div class="sem-head"><h2>Best matches</h2>' +
    '<label class="sem-toggle"><input type="checkbox"> Semantic search</label></div>' +
    '<p class="sem-status" role="status"></p>' +
    '<progress class="sem-progress" max="100" value="0" hidden></progress>' +
    '<ul class="sem-list" role="list"></ul>';
  results.parentNode.insertBefore(panel, results);
  const toggle = panel.querySelector('.sem-toggle input');
  const statusEl = panel.querySelector('.sem-status');
  const bar = panel.querySelector('.sem-progress');
  const list = panel.querySelector('.sem-list');

  function stored(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }
  function store(key, value) {
    try { localStorage.setItem(key, value); } catch (e) { /* storage is optional */ }
  }
  const saveData = !!(navigator.connection && navigator.connection.saveData);
  toggle.checked = (stored('diffct-semantic-search') || (saveData ? 'off' : 'on')) === 'on';

  // ---- data -----------------------------------------------------------------------------
  const dataReady = (async () => {
    const [meta, chunks, vectors] = await Promise.all([
      fetch(staticBase + 'search-meta.json').then((r) => r.json()),
      fetch(staticBase + 'search-chunks.json').then((r) => r.json()),
      fetch(staticBase + 'search-vectors.bin').then((r) => r.arrayBuffer()),
    ]);
    const n = chunks.length;
    if (vectors.byteLength !== n * 4 + n * meta.dim) throw new Error('search index is out of date');
    const scales = new Float32Array(vectors, 0, n);
    const codes = new Int8Array(vectors, n * 4, n * meta.dim);
    return { meta, chunks, scales, codes, bm25: buildBm25(chunks) };
  })();

  // ---- keyword ranking ------------------------------------------------------------------
  function tokens(text) {
    const out = [];
    const lower = text.toLowerCase();
    for (const m of lower.matchAll(/[a-z0-9]+|[぀-ヿ㐀-鿿가-힯]+/g)) {
      const word = m[0];
      if (/^[a-z0-9]/.test(word)) {
        const stem = word.length > 3 && word.endsWith('s') ? word.slice(0, -1) : word;
        if (!STOP.has(word)) out.push(stem);
      } else {
        for (let i = 0; i < word.length; i++) {
          out.push(word[i]);
          if (i + 1 < word.length) out.push(word.slice(i, i + 2));
        }
      }
    }
    return out;
  }

  function buildBm25(chunks) {
    const docs = chunks.map((c) => {
      const counts = new Map();
      const title = tokens(c.p + ' ' + c.s);
      const body = tokens(c.t);
      for (const t of title) counts.set(t, (counts.get(t) || 0) + 3);
      for (const t of body) counts.set(t, (counts.get(t) || 0) + 1);
      return { counts, length: title.length * 3 + body.length };
    });
    const df = new Map();
    for (const d of docs) for (const t of d.counts.keys()) df.set(t, (df.get(t) || 0) + 1);
    const avg = docs.reduce((s, d) => s + d.length, 0) / docs.length;
    return { docs, df, avg, n: docs.length };
  }

  function keywordRank(index, query) {
    const terms = [...new Set(tokens(query))];
    const k1 = 1.2;
    const b = 0.75;
    const scored = [];
    index.docs.forEach((d, i) => {
      let score = 0;
      for (const t of terms) {
        const f = d.counts.get(t);
        if (!f) continue;
        const df = index.df.get(t);
        const idf = Math.log(1 + (index.n - df + 0.5) / (df + 0.5));
        score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * d.length / index.avg));
      }
      if (score > 0) scored.push([i, score]);
    });
    scored.sort((a, c) => c[1] - a[1]);
    return scored.slice(0, POOL).map((s) => s[0]);
  }

  // ---- semantic ranking -----------------------------------------------------------------
  let modelPromise = null;
  let backend = '';

  // The checkpoint lists a vision and an audio encoder that this page never loads.
  // Count only the files that download for text.
  const downloads = new Map();
  function setProgress(info) {
    if (info.status !== 'progress' || /vision_encoder|audio_encoder/.test(info.file)) return;
    downloads.set(info.file, info);
    let loaded = 0;
    let total = 0;
    for (const d of downloads.values()) { loaded += d.loaded; total += d.total; }
    bar.hidden = false;
    bar.value = (100 * loaded) / total;
    statusEl.textContent = 'Downloading the search model: ' + Math.round(loaded / 1e6) +
      ' of ' + Math.round(total / 1e6) + ' MB. The browser keeps it for later visits.';
  }

  function loadModel(meta) {
    if (!modelPromise) {
      modelPromise = (async () => {
        const tf = await import(staticBase + 'search/transformers.min.js');
        tf.env.allowLocalModels = false;
        // ONNX Runtime Web ships several WASM builds. WebGPU needs the asyncify build. The CPU
        // backend needs the JSEP build: the asyncify build lacks the GatherBlockQuantized kernel
        // that the quantized checkpoints use.
        const runtime = (build) => {
          tf.env.backends.onnx.wasm.wasmPaths = {
            mjs: staticBase + 'search/ort-wasm-simd-threaded.' + build + '.mjs',
            wasm: staticBase + 'search/ort-wasm-simd-threaded.' + build + '.wasm',
          };
        };
        // The checkpoint is multimodal. Without these configs, only the text model downloads.
        const config = await tf.AutoConfig.from_pretrained(meta.model, { revision: meta.revision });
        delete config.vision_config;
        delete config.audio_config;
        const options = (device) => ({
          device, dtype: meta.dtype, revision: meta.revision, config, progress_callback: setProgress,
        });
        let adapter = null;
        try { adapter = navigator.gpu ? await navigator.gpu.requestAdapter() : null; } catch (e) { /* no WebGPU */ }
        let extractor;
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
          backend = 'WASM';
        }
        return extractor;
      })();
      modelPromise.catch(() => { modelPromise = null; });
    }
    return modelPromise;
  }

  async function semanticRank(data, query) {
    const extractor = await loadModel(data.meta);
    const output = await extractor(data.meta.query_prefix + query, { pooling: 'mean', normalize: true });
    const full = output.data;
    const dim = data.meta.dim;
    const q = new Float32Array(dim);
    let norm = 0;
    for (let j = 0; j < dim; j++) { q[j] = full[j]; norm += q[j] * q[j]; }
    norm = Math.sqrt(norm);
    const scored = [];
    for (let i = 0; i < data.chunks.length; i++) {
      let dot = 0;
      const base = i * dim;
      for (let j = 0; j < dim; j++) dot += q[j] * data.codes[base + j];
      scored.push([i, dot * data.scales[i] / norm]);
    }
    scored.sort((a, c) => c[1] - a[1]);
    return scored.slice(0, POOL).map((s) => s[0]);
  }

  // ---- fusion and rendering -------------------------------------------------------------
  function fuse(...lists) {
    const score = new Map();
    for (const l of lists) {
      l.forEach((id, rank) => score.set(id, (score.get(id) || 0) + 1 / (RRF_K + rank + 1)));
    }
    return [...score.entries()].sort((a, c) => c[1] - a[1]).map((e) => e[0]);
  }

  function snippet(chunk, terms) {
    const text = chunk.t.replace(/\s+/g, ' ');
    const lower = text.toLowerCase();
    let at = -1;
    for (const t of terms) {
      const i = lower.indexOf(t);
      if (i >= 0 && (at < 0 || i < at)) at = i;
    }
    const start = Math.max(0, at - 60);
    const piece = text.slice(start, start + 240);
    return (start > 0 ? '... ' : '') + piece + (start + 240 < text.length ? ' ...' : '');
  }

  function appendHighlighted(node, text, terms) {
    if (!terms.length) { node.append(text); return; }
    const pattern = new RegExp('(' + terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'gi');
    text.split(pattern).forEach((part, i) => {
      if (i % 2) {
        const mark = document.createElement('mark');
        mark.textContent = part;
        node.append(mark);
      } else {
        node.append(part);
      }
    });
  }

  function render(data, ids, query) {
    list.replaceChildren();
    const terms = [...new Set(query.toLowerCase().match(/[a-z0-9_]{3,}/g) || [])].filter((t) => !STOP.has(t));
    const perPage = new Map();
    const seen = new Set();
    let shown = 0;
    for (const id of ids) {
      const chunk = data.chunks[id];
      const page = chunk.u.split('#')[0];
      const used = perPage.get(page) || 0;
      if (used >= PER_PAGE || seen.has(chunk.u)) continue;
      perPage.set(page, used + 1);
      seen.add(chunk.u);
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = siteRoot + chunk.u;
      a.textContent = chunk.s ? chunk.p + ' › ' + chunk.s : chunk.p;
      const p = document.createElement('p');
      appendHighlighted(p, snippet(chunk, terms), terms);
      li.append(a, p);
      list.append(li);
      if (++shown >= SHOWN) break;
    }
  }

  // ---- query loop -----------------------------------------------------------------------
  let current = 0;

  async function run(query) {
    const ticket = ++current;
    query = query.trim();
    list.replaceChildren();
    bar.hidden = true;
    if (!query) { statusEl.textContent = ''; return; }
    let data;
    try {
      data = await dataReady;
    } catch (e) {
      statusEl.textContent = 'The search index is not available.';
      return;
    }
    if (ticket !== current) return;
    const keyword = keywordRank(data.bm25, query);
    render(data, fuse(keyword), query);
    if (!toggle.checked) {
      statusEl.textContent = 'Keyword matches. Semantic search is off.';
      return;
    }
    if (!modelPromise) statusEl.textContent = 'Keyword matches. Loading semantic search...';
    try {
      const semantic = await semanticRank(data, query);
      if (ticket !== current) return;
      bar.hidden = true;
      render(data, fuse(keyword, semantic), query);
      statusEl.textContent = 'Keyword and semantic matches (' + backend + ').';
    } catch (e) {
      console.error(e);
      if (ticket !== current) return;
      bar.hidden = true;
      statusEl.textContent = 'Keyword matches. Semantic search failed in this browser.';
    }
  }

  toggle.addEventListener('change', () => {
    store('diffct-semantic-search', toggle.checked ? 'on' : 'off');
    run(input.value);
  });

  input.form.addEventListener('submit', (event) => {
    event.preventDefault();
    const query = input.value.trim();
    const url = new URL(window.location.href);
    url.searchParams.set('q', query);
    history.replaceState(null, '', url);
    run(query);
    if (window.Search && typeof Search.performSearch === 'function') {
      results.replaceChildren();
      if (query) Search.performSearch(query);
    }
  });

  const initial = new URLSearchParams(window.location.search).get('q');
  if (initial) {
    input.value = initial;
    run(initial);
  }
})();

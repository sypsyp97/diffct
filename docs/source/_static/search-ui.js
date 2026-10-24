/*
 * Search interface for the landing page box, the Ctrl+K / Cmd+K dialog and the search page.
 * All three share one engine: one index, one model worker, one set of preferences.
 *
 * Keyword results show while the user types. The model loads on the first query, never on page
 * load, in a Web Worker, and semantic results merge in when the worker answers.
 */
import * as core from './search-core.js';

const staticBase = new URL('.', import.meta.url).href;
const siteRoot = new URL('../', staticBase).href;
const MODEL_MB = 377;
const SEMANTIC_DEBOUNCE_MS = 250;
const EXAMPLES = ['如何迭代重建?', 'how do I use a curved detector', 'multi GPU', 'install CUDA 13'];
const PREF_KEY = 'diffct-search';

// ---- engine -------------------------------------------------------------------------------

function readPrefs() {
  try { return JSON.parse(localStorage.getItem(PREF_KEY)) || {}; } catch (e) { return {}; }
}
function writePrefs(change) {
  try { localStorage.setItem(PREF_KEY, JSON.stringify({ ...readPrefs(), ...change })); } catch (e) { /* storage is optional */ }
}

const engine = {
  index: null,
  indexError: false,
  indexPromise: null,
  // idle | consent | loading | ready | off | error
  model: readPrefs().semantic === 'off' ? 'off' : 'idle',
  loaded: 0,
  total: 0,
  fromCache: false,
  backend: '',
  worker: null,
  listeners: new Set(),
  pending: null,
  nextId: 0,
};

function emit() { engine.listeners.forEach((f) => f()); }

function loadIndex() {
  if (!engine.indexPromise) {
    engine.indexPromise = (async () => {
      const [meta, chunks, vectors] = await Promise.all([
        fetch(staticBase + 'search-meta.json').then((r) => r.json()),
        fetch(staticBase + 'search-chunks.json').then((r) => r.json()),
        fetch(staticBase + 'search-vectors.bin').then((r) => r.arrayBuffer()),
      ]);
      engine.index = core.createIndex(chunks, meta, vectors);
      emit();
      return engine.index;
    })();
    engine.indexPromise.catch(() => { engine.indexError = true; emit(); });
  }
  return engine.indexPromise;
}

function spawnWorker(index, forceWasm) {
  const worker = new Worker(new URL('search-worker.js', import.meta.url), { type: 'module' });
  engine.worker = worker;
  worker.onmessage = ({ data }) => {
    if (data.type === 'fallback') {
      // The worker keeps WebGPU state after a failure: retry the CPU backend in a fresh one.
      worker.terminate();
      spawnWorker(index, true);
      return;
    }
    if (data.type === 'progress') {
      engine.loaded = data.loaded;
      engine.total = data.total;
    } else if (data.type === 'ready') {
      engine.model = 'ready';
      engine.backend = data.backend;
      writePrefs({ semantic: 'on', loaded: true });
    } else if (data.type === 'vector') {
      if (engine.pending && engine.pending.id === data.id) engine.pending.resolve(data.vector);
      return;
    } else if (data.type === 'error') {
      console.error(data.message);
      if (data.id === undefined) { engine.model = 'error'; worker.terminate(); engine.worker = null; }
      if (engine.pending && engine.pending.id === data.id) engine.pending.reject(new Error(data.message));
    }
    emit();
  };
  worker.onerror = (event) => {
    console.error(event.message);
    engine.model = 'error';
    engine.worker = null;
    emit();
  };
  worker.postMessage({ type: 'load', meta: index.meta, forceWasm });
}

function startModel() {
  if (engine.worker) return;
  engine.model = 'loading';
  engine.fromCache = !!readPrefs().loaded;
  engine.loaded = 0;
  engine.total = 0;
  emit();
  loadIndex().then((index) => {
    if (engine.model === 'loading' && !engine.worker) spawnWorker(index, false);
  }).catch(() => { engine.model = 'error'; emit(); });
}

function stopModel() {
  if (engine.worker) engine.worker.terminate();
  engine.worker = null;
  if (engine.pending) engine.pending.reject(new Error('stale'));
  engine.pending = null;
  engine.model = 'off';
  writePrefs({ semantic: 'off' });
  emit();
}

function enableModel() {
  writePrefs({ semantic: 'on' });
  engine.model = 'idle';
  startModel();
}

/** Start the model on the first query. Large downloads on a metered connection need a click. */
function onFirstQuery() {
  if (engine.model !== 'idle') return;
  const prefs = readPrefs();
  const connection = navigator.connection || {};
  if (!prefs.loaded && (connection.saveData || connection.type === 'cellular')) {
    engine.model = 'consent';
    emit();
  } else {
    startModel();
  }
}

/** Read the model from the browser cache on focus, only when an earlier visit loaded it. */
function onFocus() {
  loadIndex();
  const prefs = readPrefs();
  if (prefs.loaded && prefs.semantic !== 'off' && engine.model === 'idle') startModel();
}

function embed(text) {
  if (engine.pending) engine.pending.reject(new Error('stale'));
  const id = ++engine.nextId;
  return new Promise((resolve, reject) => {
    engine.pending = { id, resolve, reject };
    engine.worker.postMessage({ type: 'embed', id, text });
  });
}

// ---- view ---------------------------------------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function appendHighlighted(node, text, terms) {
  if (!terms.length) { node.append(text); return; }
  const pattern = new RegExp('(' + terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'gi');
  text.split(pattern).forEach((part, i) => {
    if (i % 2) node.append(el('mark', '', part));
    else node.append(part);
  });
}

const TAGS = {
  keyword: ['Keyword', 'Matches the words of the query'],
  semantic: ['Semantic', 'Matches the meaning of the query'],
  both: ['Both', 'Matches the words and the meaning of the query'],
};

let hostCount = 0;

function attach({ input, host, mode, maxGroups }) {
  const uid = ++hostCount;
  const listId = `srch-list-${uid}`;
  host.classList.add('srch-host');
  host.dataset.mode = mode;
  const bar = el('div', 'srch-bar');
  const status = el('span', 'srch-status');
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  const actions = el('span', 'srch-actions');
  const count = el('span', 'srch-sr');
  count.setAttribute('role', 'status');
  bar.append(status, actions, count);
  const progress = el('progress', 'srch-progress');
  progress.max = 100;
  progress.hidden = true;
  const message = el('div', 'srch-message');
  const list = el('ul', 'srch-list');
  list.id = listId;
  list.setAttribute('role', 'listbox');
  list.setAttribute('aria-label', 'Search results');
  const foot = el('div', 'srch-foot');
  host.append(bar, progress, message, list, foot);

  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-controls', listId);
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('aria-expanded', 'false');

  let query = '';
  let keywordIds = [];
  let semantic = null; // result for `query`, or null
  let semanticPending = false;
  let timer = 0;
  let ticket = 0;
  let groups = [];
  let activeId = null;
  const expanded = new Set();

  const options = () => [...list.querySelectorAll('.srch-item:not([hidden])')];

  function setActive(option) {
    for (const o of list.querySelectorAll('.srch-item[aria-selected="true"]')) o.setAttribute('aria-selected', 'false');
    activeId = option ? option.id : null;
    if (option) {
      option.setAttribute('aria-selected', 'true');
      option.scrollIntoView({ block: 'nearest' });
      input.setAttribute('aria-activedescendant', option.id);
    } else {
      input.removeAttribute('aria-activedescendant');
    }
  }

  function button(label, onClick) {
    const b = el('button', 'srch-button', label);
    b.type = 'button';
    b.addEventListener('click', onClick);
    return b;
  }

  // Buttons are created once. A progress tick must not rebuild them, or a focused button loses focus.
  const keywordOnly = button('Keyword only', stopModel);
  const download = button('Download', enableModel);
  const turnOn = button(`Turn on semantic search (\u2248${MODEL_MB} MB download)`, enableModel);
  let actionsKey = '';
  function setActions(key, ...nodes) {
    if (key === actionsKey) return;
    actionsKey = key;
    actions.replaceChildren(...nodes);
  }

  function renderStatus() {
    progress.hidden = engine.model !== 'loading';
    const mb = (bytes) => Math.round(bytes / 1e6);
    const total = groups.reduce((n, g) => n + g.items.length + g.more.length, 0);
    // The status line shows only what the visitor needs to act on or wait for.
    let text = '';
    if (engine.indexError) {
      text = 'The search index is not available. Use the page search below.';
    } else if (engine.model === 'loading') {
      text = engine.fromCache
        ? 'Loading semantic search model from the browser cache'
        : `Loading semantic search model (\u2248${MODEL_MB} MB, one time)`;
      if (engine.total) text += `: ${mb(engine.loaded)} of ${mb(engine.total)} MB`;
      progress.value = engine.total ? (100 * engine.loaded) / engine.total : 0;
      setActions('loading', keywordOnly);
    } else if (engine.model === 'consent') {
      text = `Semantic search matches meaning, also across languages. It needs a one-time download of \u2248${MODEL_MB} MB.`;
      setActions('consent', download, keywordOnly);
    } else if (engine.model === 'off') {
      text = 'Keyword search only.';
      setActions('off', turnOn);
    } else if (engine.model === 'error') {
      text = 'Semantic search does not work in this browser. These are keyword matches.';
    }
    if (!['loading', 'consent', 'off'].includes(engine.model)) setActions('none');
    // Screen readers still hear the result count.
    count.textContent = query && !semanticPending ? `${total} result${total === 1 ? '' : 's'}` : '';
    host.dataset.state = semanticPending ? 'pending' : engine.model;
    status.textContent = text;
  }

  function renderEmpty() {
    setActive(null);
    list.replaceChildren();
    foot.replaceChildren();
    message.replaceChildren();
    input.setAttribute('aria-expanded', 'false');
    const label = el('span', 'srch-examples-label', 'Try:');
    const chips = el('span', 'srch-chips');
    for (const example of EXAMPLES) {
      chips.append(button(example, () => { input.value = example; input.focus(); onInput(); }));
    }
    message.append(label, chips);
    message.className = 'srch-message srch-examples';
    renderStatus();
  }

  function renderItem(item, first) {
    const chunk = engine.index.chunks[item.id];
    const sn = core.snippet(engine.index, item, core.highlightTerms(query), semantic);
    const a = el('a', 'srch-item' + (first ? '' : ' srch-sub'));
    a.id = `srch-${uid}-${item.id}`;
    a.href = siteRoot + chunk.u;
    a.setAttribute('role', 'option');
    a.setAttribute('aria-selected', a.id === activeId ? 'true' : 'false');
    const head = el('span', 'srch-head');
    if (first || !chunk.s) head.append(el('span', 'srch-title', chunk.p));
    if (chunk.s) head.append(el('span', 'srch-crumb', (first ? ' › ' : '') + chunk.s.split(' > ').join(' › ')));
    const [label, hint] = TAGS[item.tag];
    const tag = el('span', 'srch-tag srch-tag-' + item.tag, label);
    tag.title = hint;
    const text = el('span', 'srch-snippet' + (sn.code ? ' srch-code' : ''));
    appendHighlighted(text, sn.text, core.highlightTerms(query));
    a.append(head, tag, text);
    return a;
  }

  function renderResults() {
    const held = list.offsetHeight;
    message.replaceChildren();
    message.className = 'srch-message';
    list.replaceChildren();
    groups = engine.index
      ? core.combine(engine.index, keywordIds, semantic, { maxGroups, query })
      : [];
    for (const g of groups) {
      const li = el('li', 'srch-group');
      li.setAttribute('role', 'presentation');
      g.items.forEach((item, i) => li.append(renderItem(item, i === 0)));
      if (g.more.length) {
        const open = expanded.has(g.page);
        const extra = g.more.map((item) => {
          const a = renderItem(item, false);
          a.hidden = !open;
          li.append(a);
          return a;
        });
        const toggle = button(open ? 'Show fewer sections' : `${g.more.length} more section${g.more.length === 1 ? '' : 's'} on this page`, () => {
          const show = !expanded.has(g.page);
          if (show) expanded.add(g.page); else expanded.delete(g.page);
          extra.forEach((a) => { a.hidden = !show; });
          toggle.textContent = show ? 'Show fewer sections' : `${g.more.length} more section${g.more.length === 1 ? '' : 's'} on this page`;
          toggle.setAttribute('aria-expanded', String(show));
        });
        toggle.classList.add('srch-more');
        toggle.setAttribute('aria-expanded', String(open));
        li.append(toggle);
      }
      list.append(li);
    }
    // Hold the height while results merge, so the page below does not jump.
    list.style.minHeight = held && semantic ? `${held}px` : '';
    const current = activeId && document.getElementById(activeId);
    if (current && !current.hidden) setActive(current); else setActive(null);
    input.setAttribute('aria-expanded', String(groups.length > 0));
    if (!groups.length && !semanticPending) {
      const parts = [`No matches for “${query}”. Try other words or a shorter question.`];
      if (engine.model !== 'ready' && engine.model !== 'loading') parts.push('Semantic search can match by meaning.');
      message.append(el('p', '', parts.join(' ')));
      const plain = el('a', '', 'Try the standard keyword search');
      plain.href = `${siteRoot}search.html?q=${encodeURIComponent(query)}`;
      message.append(plain);
    }
    foot.replaceChildren();
    if (groups.length && mode !== 'page') {
      const all = el('a', '', 'All results');
      all.href = `${siteRoot}search.html?q=${encodeURIComponent(query)}`;
      foot.append(all);
    }
    renderStatus();
  }

  async function runSemantic(myTicket) {
    if (engine.model !== 'ready' || !query) return;
    semanticPending = true;
    renderStatus();
    try {
      const vector = await embed(engine.index.meta.query_prefix + query);
      if (myTicket !== ticket) return;
      semantic = core.semanticSearch(engine.index, core.queryVector(engine.index, vector));
      semanticPending = false;
      renderResults();
    } catch (e) {
      if (e.message === 'stale') return; // a newer query is already in flight
      console.error(e);
      if (myTicket !== ticket) return;
      semanticPending = false;
      renderResults();
    }
  }

  function onInput() {
    const myTicket = ++ticket;
    clearTimeout(timer);
    query = input.value.trim();
    semantic = null;
    semanticPending = false;
    list.style.minHeight = '';
    if (mode === 'page') {
      const url = new URL(window.location.href);
      if (query) url.searchParams.set('q', query); else url.searchParams.delete('q');
      history.replaceState(null, '', url);
      const native = document.getElementById('search-results');
      if (native) native.classList.toggle('srch-stale', query !== nativeQuery);
    }
    if (!query) { renderEmpty(); return; }
    onFirstQuery();
    loadIndex().then(() => {
      if (myTicket !== ticket) return;
      keywordIds = core.keywordSearch(engine.index, query);
      semanticPending = engine.model === 'ready' || engine.model === 'loading';
      renderResults();
      if (engine.model === 'ready') timer = setTimeout(() => runSemantic(myTicket), SEMANTIC_DEBOUNCE_MS);
    }).catch(() => renderStatus());
  }

  // The model becomes ready while a query waits: run it now.
  let lastModel = engine.model;
  engine.listeners.add(() => {
    if (lastModel !== 'ready' && engine.model === 'ready' && query && !semantic) {
      semanticPending = true;
      runSemantic(ticket);
    }
    if (engine.model !== 'ready' && engine.model !== 'loading') semanticPending = false;
    const changed = lastModel !== engine.model;
    lastModel = engine.model;
    // Progress ticks update the status line only, so the list does not flicker.
    if (!changed) renderStatus();
    else if (query && engine.index) renderResults();
    else if (!query) renderEmpty();
  });

  input.addEventListener('input', onInput);
  input.addEventListener('focus', onFocus);
  if (document.activeElement === input) onFocus();
  input.addEventListener('keydown', (event) => {
    // Selecting an input-method candidate sends Enter and arrow keys. They are not search commands.
    if (event.isComposing || event.keyCode === 229) return;
    const all = options();
    const at = all.findIndex((o) => o.id === activeId);
    if (event.key === 'ArrowDown' && all.length) {
      event.preventDefault();
      setActive(all[Math.min(at + 1, all.length - 1)]);
    } else if (event.key === 'ArrowUp' && all.length) {
      event.preventDefault();
      setActive(at > 0 ? all[at - 1] : null);
    } else if (event.key === 'Enter' && at >= 0) {
      event.preventDefault();
      all[at].click();
    } else if (event.key === 'Escape' && mode !== 'dialog' && input.value) {
      event.preventDefault();
      input.value = '';
      onInput();
    }
  });

  if (mode === 'dialog') {
    // Sphinx handlers on `document` cancel Escape when focus is outside an input, which stops
    // the native close. Close the dialog from here when focus is on a result.
    host.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || event.isComposing) return;
      event.preventDefault();
      host.closest('dialog').close();
    });
  }

  let nativeQuery = '';
  function runNative(q) {
    const native = document.getElementById('search-results');
    // Sphinx declares `const Search` at top level, so it is a global binding, not `window.Search`.
    if (!native || typeof Search === 'undefined' || typeof Search.performSearch !== 'function') return;
    if (typeof Search.stopPulse === 'function') Search.stopPulse();
    nativeQuery = q;
    native.replaceChildren();
    native.classList.remove('srch-stale');
    if (q) Search.performSearch(q);
  }
  if (mode === 'page') {
    input.form.addEventListener('submit', (event) => {
      event.preventDefault();
      clearTimeout(timer);
      onInput();
      runNative(input.value.trim());
    });
    const initial = new URLSearchParams(window.location.search).get('q');
    nativeQuery = (initial || '').trim();
    if (initial) { input.value = initial; onInput(); }
  }
  if (!input.value) renderEmpty(); else onInput();
}

// ---- attach to the three places -------------------------------------------------------------

function init() {
  const landing = document.querySelector('.landing-search input');
  if (landing) {
    // The landing box replaces the navbar search button and the Ctrl+K dialog on this page.
    document.documentElement.classList.add('srch-landing');
    window.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && !event.altKey && /^k$/i.test(event.key)) {
        event.preventDefault();
        event.stopImmediatePropagation();
        landing.focus();
        landing.select();
      }
    }, true);
    attach({ input: landing, host: document.querySelector('.landing-search .srch-mount'), mode: 'landing', maxGroups: 6 });
  }
  const dialogInput = document.querySelector('#pst-search-dialog input[name="q"]');
  if (dialogInput) {
    const host = el('div', 'srch-mount');
    dialogInput.closest('form').after(host);
    attach({ input: dialogInput, host, mode: 'dialog', maxGroups: 6 });
  }
  const pageInput = document.querySelector('.bd-search-container form.bd-search input[name="q"]');
  if (pageInput) {
    const host = el('div', 'srch-mount');
    pageInput.closest('form').after(host);
    attach({ input: pageInput, host, mode: 'page', maxGroups: 10 });
  }
}
init();

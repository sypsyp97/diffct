/*
 * Test harness for the documentation search front end.
 *
 * - A static server (one per Playwright worker, free port) serves the built HTML from DOCS_HTML.
 * - The Hugging Face Hub is replaced by a fake that serves a tiny ONNX model from
 *   fixtures/tiny-model, so the real transformers.js runtime and the real browser cache run
 *   without the 377 MB download. Tests can hold, delay or fail each model file.
 * - The runtime module `_static/search/transformers.min.js` is served through a thin wrapper that
 *   re-exports the real module and reports each `pipeline()` call, each finished model session and
 *   each embedding call to `/__probe/*` on the static server. Tests use these probes to count
 *   sessions and to delay the semantic answer of a given query.
 */
import { test as base, expect } from '@playwright/test';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const TINY_MODEL = path.join(HERE, 'fixtures', 'tiny-model');
export const FULL_MODEL = process.env.FULL_MODEL === '1';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm', '.txt': 'text/plain',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.gif': 'image/gif',
  '.woff': 'font/woff', '.woff2': 'font/woff2', '.ttf': 'font/ttf', '.map': 'application/json',
  '.bin': 'application/octet-stream', '.inv': 'application/octet-stream', '.ico': 'image/x-icon',
  '.mp4': 'video/mp4', '.webm': 'video/webm',
};

const WRAPPER = `
import * as real from './transformers.min.js?real=1';
export * from './transformers.min.js?real=1';
const probe = (kind, params = {}) =>
  fetch(new URL('/__probe/' + kind + '?' + new URLSearchParams(params), self.location.href)).then((r) => r.text());
const control = JSON.parse(await probe('init'));
if (control.webgpu === 'absent') {
  Object.defineProperty(self.navigator, 'gpu', { value: undefined, configurable: true });
} else if (control.webgpu === 'adapter') {
  Object.defineProperty(self.navigator, 'gpu', { value: { requestAdapter: async () => ({ simulated: true }) }, configurable: true });
} else if (control.webgpu === 'reject') {
  Object.defineProperty(self.navigator, 'gpu', {
    value: { requestAdapter: async () => { throw new Error('simulated WebGPU adapter failure'); } }, configurable: true });
}
export async function pipeline(task, model, options = {}) {
  await probe('pipeline', { device: String(options.device) });
  if ((control.failDevices || []).includes(String(options.device))) {
    throw new Error('simulated ' + options.device + ' initialization failure');
  }
  const extractor = await real.pipeline(task, model, options);
  await probe('session', { device: String(options.device) });
  return async (text, opts) => {
    await probe('embed', { text });
    return extractor(text, opts);
  };
}
`;

export function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- static server ------------------------------------------------------------------------

export async function startServer(root) {
  if (!root) throw new Error('Set DOCS_HTML to the built HTML directory (docs/_build/html).');
  root = path.resolve(root);
  if (!fs.existsSync(path.join(root, 'index.html'))) throw new Error(`DOCS_HTML has no index.html: ${root}`);
  const events = new EventEmitter();
  events.setMaxListeners(0);
  const state = {
    overlays: new Map(), // pathname -> { body, type }
    wrapper: true,
    control: { webgpu: 'absent', failDevices: [] },
    probes: [],
    gates: [], // { kind, match(params), gate: deferred, seen: deferred }
    probeDelay: null, // (entry) => milliseconds
  };
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const send = (status, body, type) => {
      res.writeHead(status, { 'content-type': type, 'cache-control': 'no-store' });
      res.end(body);
    };
    try {
      if (url.pathname.startsWith('/__probe/')) {
        const entry = { kind: url.pathname.slice('/__probe/'.length), params: Object.fromEntries(url.searchParams), t: Date.now() };
        if (entry.kind === 'init') return send(200, JSON.stringify(state.control), 'application/json');
        state.probes.push(entry);
        events.emit('probe', entry);
        if (state.probeDelay) await sleep(state.probeDelay(entry) || 0);
        for (const g of state.gates) {
          if (g.kind === entry.kind && g.match(entry.params)) {
            g.seen.resolve(entry);
            await g.gate.promise;
          }
        }
        return send(200, 'ok', 'text/plain');
      }
      if (state.overlays.has(url.pathname)) {
        const o = state.overlays.get(url.pathname);
        return send(200, o.body, o.type);
      }
      if (state.wrapper && url.pathname.endsWith('/_static/search/transformers.min.js') && !url.searchParams.has('real')) {
        return send(200, WRAPPER, 'text/javascript');
      }
      let file = path.join(root, decodeURIComponent(url.pathname));
      if (!file.startsWith(root)) return send(403, 'forbidden', 'text/plain');
      if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
      if (!fs.existsSync(file)) return send(404, 'not found', 'text/plain');
      return send(200, fs.readFileSync(file), MIME[path.extname(file)] || 'application/octet-stream');
    } catch (e) {
      if (!res.headersSent) send(500, String(e), 'text/plain');
    }
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const origin = `http://localhost:${server.address().port}`;
  return {
    origin,
    root,
    events,
    state,
    reset() {
      for (const g of state.gates) g.gate.resolve();
      state.overlays.clear();
      state.wrapper = true;
      state.control = { webgpu: 'absent', failDevices: [] };
      state.probes = [];
      state.gates = [];
      state.probeDelay = null;
    },
    probes(kind) { return state.probes.filter((p) => p.kind === kind); },
    /** Hold the matching probe until `release()`; `seen` resolves when the probe arrives. */
    gate(kind, match) {
      const g = { kind, match, gate: deferred(), seen: deferred() };
      state.gates.push(g);
      return { seen: g.seen.promise, release: () => g.gate.resolve() };
    },
    /** Resolves with the first probe of `kind` that matches, including earlier ones. */
    waitProbe(kind, match = () => true, timeout = 60_000) {
      const found = state.probes.find((p) => p.kind === kind && match(p.params));
      if (found) return Promise.resolve(found);
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => { events.off('probe', on); reject(new Error(`no ${kind} probe within ${timeout} ms`)); }, timeout);
        const on = (p) => {
          if (p.kind === kind && match(p.params)) { clearTimeout(timer); events.off('probe', on); resolve(p); }
        };
        events.on('probe', on);
      });
    },
    close() { return new Promise((r) => server.close(r)); },
  };
}

// ---- fake Hugging Face Hub ----------------------------------------------------------------

const CORS = {
  'access-control-allow-origin': '*',
  'access-control-expose-headers': 'content-length, content-type, etag, x-repo-commit, x-linked-size',
};

const isMetadataProbe = (r) => r.range === 'bytes=0-0';

export class FakeHub extends EventEmitter {
  constructor() {
    super();
    this.setMaxListeners(0);
    this.log = []; // every request that reached the network layer
    this.rules = new Map(); // file -> { hold: deferred, delay, fail, seen: deferred }
  }

  /**
   * Downloads per file name (path after /resolve/<revision>/), any revision. A one-byte Range
   * request (`bytes=0-0`) reads only the file size, so it does not count as a download.
   */
  downloads(file) { return this.log.filter((r) => r.file === file && !isMetadataProbe(r)).length; }
  count(file) { return this.log.filter((r) => r.file === file).length; }
  downloadLog() { return this.log.filter((r) => !isMetadataProbe(r)).map((r) => r.url); }

  hold(file) {
    const rule = { ...(this.rules.get(file) || {}), hold: deferred(), seen: deferred() };
    this.rules.set(file, rule);
    return { seen: rule.seen.promise, release: () => rule.hold.resolve() };
  }
  fail(file, how) { this.rules.set(file, { ...(this.rules.get(file) || {}), fail: how }); }
  delay(file, ms) { this.rules.set(file, { ...(this.rules.get(file) || {}), delay: ms }); }
  clear() {
    for (const r of this.rules.values()) r.hold?.resolve();
    this.rules.clear();
  }

  waitFor(file, timeout = 60_000) {
    if (this.downloads(file)) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`no download of ${file} within ${timeout} ms`)), timeout);
      const on = (r) => { if (r.file === file && !isMetadataProbe(r)) { clearTimeout(timer); this.off('request', on); resolve(); } };
      this.on('request', on);
    });
  }

  async handle(route) {
    const request = route.request();
    const url = new URL(request.url());
    const m = url.pathname.match(/\/resolve\/[^/]+\/(.+)$/);
    const file = m ? decodeURIComponent(m[1]) : url.pathname;
    const range = (await request.allHeaders()).range || null;
    const entry = { file, url: url.href, method: request.method(), range, t: Date.now() };
    this.log.push(entry);
    this.emit('request', entry);
    const rule = this.rules.get(file) || this.rules.get('*') || {};
    if (!isMetadataProbe(entry)) {
      rule.seen?.resolve(entry);
      if (rule.hold) await rule.hold.promise;
      if (rule.delay) await sleep(rule.delay);
    }
    try {
      if (rule.fail === 'abort') return await route.abort('internetdisconnected');
      if (typeof rule.fail === 'number') {
        return await route.fulfill({ status: rule.fail, headers: CORS, body: 'simulated failure' });
      }
      const local = path.join(TINY_MODEL, file);
      if (!m || !local.startsWith(TINY_MODEL) || !fs.existsSync(local)) {
        return await route.fulfill({ status: 404, headers: CORS, body: 'Entry not found' });
      }
      const body = fs.readFileSync(local);
      if (entry.range === 'bytes=0-0') {
        return await route.fulfill({
          status: 206,
          body: body.subarray(0, 1),
          headers: { ...CORS, 'content-range': `bytes 0-0/${body.length}`, 'content-length': '1',
            'access-control-expose-headers': `${CORS['access-control-expose-headers']}, content-range` },
        });
      }
      return await route.fulfill({
        status: 200,
        body,
        headers: {
          ...CORS,
          'content-type': file.endsWith('.json') ? 'application/json' : 'application/octet-stream',
          'content-length': String(body.length),
        },
      });
    } catch (e) {
      // The page or the worker that made the request is gone.
    }
  }
}

// ---- page helpers -------------------------------------------------------------------------

export function landingUI(page) {
  const input = page.locator('.landing-search input[name="q"]');
  return { input, host: page.locator('.landing-search .srch-host'), listbox: listboxOf(input) };
}

export function dialogUI(page) {
  const input = page.locator('#pst-search-dialog input[name="q"]');
  return {
    dialog: page.locator('#pst-search-dialog'),
    input,
    host: page.locator('#pst-search-dialog .srch-host'),
    trigger: page.locator('button.search-button__button:visible').first(),
    listbox: listboxOf(input),
  };
}

export function searchPageUI(page) {
  const input = page.locator('.bd-search-container form.bd-search input[name="q"]');
  return { input, host: page.locator('.bd-search-container .srch-host'), listbox: listboxOf(input) };
}

/** The element that the combobox names in aria-controls. */
export function listboxOf(input) {
  const page = input.page();
  return {
    async locator() {
      const id = await input.getAttribute('aria-controls');
      expect(id, 'combobox has aria-controls').toBeTruthy();
      return page.locator(`[id="${id}"]`);
    },
  };
}

/** Visible options of the listbox, as "href<TAB>text" lines, in order. */
export async function snapshot(listbox) {
  const lb = await listbox.locator();
  return lb.evaluate((el) => [...el.querySelectorAll('[role="option"]')]
    .filter((o) => !o.closest('[hidden]'))
    .map((o) => `${o.getAttribute('href')}\t${o.textContent.replace(/\s+/g, ' ').trim()}`));
}

/** Wait until the visible options stop changing for `quietMs` and return them. */
export async function stableSnapshot(listbox, { quietMs = 600, timeout = 20_000, allowEmpty = false } = {}) {
  const end = Date.now() + timeout;
  let last = null;
  let since = Date.now();
  while (Date.now() < end) {
    const now = await snapshot(listbox);
    if (JSON.stringify(now) !== JSON.stringify(last)) { last = now; since = Date.now(); }
    else if (Date.now() - since >= quietMs && (allowEmpty || now.length)) return now;
    await sleep(100);
  }
  throw new Error(`results did not settle within ${timeout} ms; last: ${JSON.stringify(last)}`);
}

export async function typeQuery(input, text) {
  await input.click();
  await input.press('ControlOrMeta+a');
  await input.press('Backspace');
  await input.pressSequentially(text, { delay: 5 });
}

export async function clearQuery(input) {
  await input.press('ControlOrMeta+a');
  await input.press('Backspace');
}

/** Status text that the search host shows (role=status regions). */
export async function statusText(host) {
  const texts = await host.locator('[role="status"]').allTextContents();
  return texts.join(' ').replace(/\s+/g, ' ').trim();
}

/** A message for a visitor: no stack trace, no exception name, no raw error text. */
export function expectPlainMessage(text) {
  expect(text.length, `message "${text}"`).toBeGreaterThan(0);
  expect(text.length, `message "${text}"`).toBeLessThan(300);
  expect(text).not.toMatch(/\b(Error|TypeError|RangeError|Exception)\b|\bat \S+:\d+|undefined|\[object |simulated|ERR_|onnx|wasm/i);
}

export function trackWorkers(page) {
  const workers = [];
  page.on('worker', (w) => {
    const info = { url: w.url(), closed: false, closedAt: null, worker: w };
    workers.push(info);
    w.on('close', () => { info.closed = true; info.closedAt = Date.now(); });
  });
  return workers;
}

export function trackConsole(page) {
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(String(e)));
  return errors;
}

// ---- CDP helpers --------------------------------------------------------------------------

/**
 * CDP session on the page that also auto-attaches (non-flattened) to its dedicated workers,
 * so heap and GC commands can be sent to the worker target.
 */
export async function cdpWithWorkers(page) {
  const cdp = await page.context().newCDPSession(page);
  const workers = new Map(); // sessionId -> targetInfo
  const pending = new Map();
  let nextId = 1;
  cdp.on('Target.attachedToTarget', (e) => { if (e.targetInfo.type === 'worker') workers.set(e.sessionId, e.targetInfo); });
  cdp.on('Target.detachedFromTarget', (e) => workers.delete(e.sessionId));
  cdp.on('Target.receivedMessageFromTarget', (e) => {
    const msg = JSON.parse(e.message);
    const p = msg.id && pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    if (msg.error) p.reject(new Error(`${msg.error.message}`)); else p.resolve(msg.result);
  });
  await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false, flatten: false });
  const sendToWorker = (sessionId, method, params = {}) => new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    cdp.send('Target.sendMessageToTarget', { sessionId, message: JSON.stringify({ id, method, params }) }).catch(reject);
  });
  return {
    cdp,
    workers,
    sendToWorker,
    async pageHeap() {
      await cdp.send('HeapProfiler.collectGarbage');
      await cdp.send('HeapProfiler.collectGarbage');
      return (await cdp.send('Runtime.getHeapUsage')).usedSize;
    },
    async workerHeap(sessionId) {
      await sendToWorker(sessionId, 'HeapProfiler.collectGarbage');
      await sendToWorker(sessionId, 'HeapProfiler.collectGarbage');
      return (await sendToWorker(sessionId, 'Runtime.getHeapUsage')).usedSize;
    },
    async metrics() {
      await cdp.send('Performance.enable');
      const { metrics } = await cdp.send('Performance.getMetrics');
      return Object.fromEntries(metrics.map((m) => [m.name, m.value]));
    },
    /** Listener count on the object that `expression` evaluates to; depth -1 includes the subtree. */
    async listeners(expression, depth = 0) {
      const { result } = await cdp.send('Runtime.evaluate', { expression, objectGroup: 'listeners' });
      const { listeners } = await cdp.send('DOMDebugger.getEventListeners', { objectId: result.objectId, depth, pierce: true });
      await cdp.send('Runtime.releaseObjectGroup', { objectGroup: 'listeners' });
      return listeners;
    },
  };
}

const isHub = (url) => /(^|\.)(huggingface\.co|hf\.co)$/.test(url.hostname);

/** Send Hub requests of `context` to `hub`; block every other external request. */
export async function routeFakeHub(context, hub) {
  await context.route(isHub, (route) => hub.handle(route));
  await context.route((url) => !['localhost', '127.0.0.1'].includes(url.hostname) && !isHub(url),
    (route) => route.abort('blockedbyclient'));
}

/**
 * The results that `query` shows in a fresh browser context, as a reference.
 * mode 'keyword': the model download never finishes, so only keyword results show.
 * mode 'semantic': the model loads and the semantic answer for `query` merges in.
 * where: 'landing' (index.html) or 'dialog' (quickstart.html, Ctrl+K dialog).
 */
export async function reference(browser, server, query, mode, where = 'landing') {
  const context = await browser.newContext({ baseURL: server.origin });
  const hub = new FakeHub();
  await routeFakeHub(context, hub);
  if (mode === 'keyword') hub.hold('onnx/model_quantized.onnx');
  try {
    const page = await context.newPage();
    let ui;
    if (where === 'landing') {
      await page.goto('/index.html');
      ui = landingUI(page);
    } else {
      await page.goto('/quickstart.html');
      ui = dialogUI(page);
      await ui.trigger.click();
      await expect(ui.dialog).toBeVisible();
    }
    await typeQuery(ui.input, query);
    if (mode === 'semantic') await server.waitProbe('embed', (p) => p.text.endsWith(query));
    return await stableSnapshot(ui.listbox, { quietMs: 1000 });
  } finally {
    hub.clear();
    await context.close();
  }
}

// ---- fixtures -----------------------------------------------------------------------------

export const test = base.extend({
  server: [async ({}, use) => {
    const server = await startServer(process.env.DOCS_HTML);
    await use(server);
    await server.close();
  }, { scope: 'worker' }],
  baseURL: async ({ server }, use) => { await use(server.origin); },
  site: async ({ server }, use) => {
    server.reset();
    await use(server);
    server.reset();
  },
  hub: async ({ context, site }, use) => {
    const hub = new FakeHub();
    if (!FULL_MODEL) await routeFakeHub(context, hub);
    await use(hub);
    hub.clear();
  },
});

export { expect };

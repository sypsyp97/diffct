// Contract "semantic search failure handling" (C1-C5): a model file that fails or stalls, a query
// that never gets a semantic answer, a WebGPU failure after load, and text-only model downloads.
// The page always falls back to keyword results with one plain message, never loops, offers a
// retry, and never downloads the vision or audio encoders.
import {
  test, expect, landingUI, typeQuery, stableSnapshot, statusText, expectPlainMessage, reference,
  trackWorkers, sleep, TINY_MODEL,
} from './harness.mjs';
import fs from 'node:fs';
import path from 'node:path';

test.use({ semanticSearch: true });

const CONFIG = 'config.json';
const WEIGHTS = 'onnx/model_quantized.onnx';
const DATA = 'onnx/model_quantized.onnx_data';
const MODEL_FILES = [CONFIG, 'tokenizer.json', 'tokenizer_config.json', WEIGHTS, DATA];
const FIRST = 'fan beam';
const NEXT = 'cone beam geometry';
const MORE = ['multi GPU', 'install CUDA', 'how do I reconstruct iteratively'];
const LAST = MORE[MORE.length - 1];

// ---- helpers ------------------------------------------------------------------------------

/** The C1 failure message: semantic search failed, keyword results are still shown. */
function isFailureMessage(text) {
  return /semantic/i.test(text) && /keyword/i.test(text) && !/loading/i.test(text)
    && /fail|not work|unavailable|could not|couldn.t|cannot|can.t|did not|didn.t|stopped|no longer/i.test(text);
}

function trackPageErrors(page) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  return errors;
}

const liveWorkers = (workers) => workers.filter((w) => !w.closed).length;

/** References are computed before a test changes the shared server state (control, gates). */
async function keywordRefs(browser, site, queries) {
  const out = {};
  for (const q of queries) out[q] = await reference(browser, site, q, 'keyword');
  return out;
}

function retryControl(ui) {
  const name = /retry|try again/i;
  return ui.host.getByRole('button', { name }).or(ui.host.getByRole('link', { name }));
}

async function expectFailureShown(ui, timeout) {
  let message = '';
  try {
    await expect.poll(async () => {
      message = await statusText(ui.host);
      return isFailureMessage(message);
    }, { timeout }).toBe(true);
  } catch (e) {
    throw new Error(`the failure message did not show within ${timeout} ms; last status: "${message}"`);
  }
  expectPlainMessage(message);
  await expect(ui.host.getByText(message.slice(0, 40), { exact: false }).first()).toBeVisible();
  return message;
}

async function expectProgressHidden(ui) {
  await expect(ui.host.locator('progress, [role="progressbar"]').filter({ visible: true })).toHaveCount(0);
}

async function expectKeywordResults(ui, refs, current) {
  expect(await stableSnapshot(ui.listbox, { quietMs: 1000 }), `results for "${current}"`).toEqual(refs[current]);
  await typeQuery(ui.input, NEXT);
  expect(await stableSnapshot(ui.listbox, { quietMs: 1000 }), `results for "${NEXT}"`).toEqual(refs[NEXT]);
}

/** C1.4: while idle for 5 s and during 3 more queries, nothing starts and nothing downloads. */
async function expectNoRetryLoop({ ui, site, hub, workers }) {
  const requests = hub.log.length;
  const attempts = site.probes('pipeline').length;
  const started = workers.length;
  const check = (when) => {
    expect(hub.log.slice(requests).map((r) => r.url), `model requests ${when}`).toEqual([]);
    expect(site.probes('pipeline').length - attempts, `pipeline() calls ${when}`).toBe(0);
    expect(workers.length - started, `workers started ${when}`).toBe(0);
  };
  await sleep(5000);
  check('while idle');
  for (const q of MORE) {
    await typeQuery(ui.input, q);
    await sleep(700);
  }
  await sleep(1000);
  check('during 3 more queries');
}

/** C1.5: one activation of the retry control starts one load; semantic results merge in. */
async function expectRetryWorks({ ui, site, workers, query, semanticRef }) {
  const control = retryControl(ui);
  await expect(control, 'a control to retry semantic search').toBeVisible();
  const started = workers.length;
  const attempts = site.probes('pipeline').length;
  const t0 = Date.now();
  await control.first().click();
  await expect.poll(() => site.probes('embed').filter((p) => p.t >= t0 && p.text.endsWith(query)).length,
    { timeout: 30_000, message: `semantic answer for "${query}" after the retry` }).toBeGreaterThan(0);
  expect(await stableSnapshot(ui.listbox, { quietMs: 1000 }), `results for "${query}"`).toEqual(semanticRef);
  await sleep(2000);
  expect(workers.length - started, 'workers started by one retry').toBe(1);
  expect(site.probes('pipeline').length - attempts, 'load attempts started by one retry').toBe(1);
}

// ---- C1: any model file fails while the model loads ----------------------------------------

test.describe('C1: a model file fails while the model loads', () => {
  for (const file of MODEL_FILES) {
    for (const how of ['abort', 404, 500]) {
      test(`${how} on ${file}`, async ({ browser, page, site, hub }) => {
        const refs = await keywordRefs(browser, site, [FIRST, NEXT]);
        const semanticRef = await reference(browser, site, LAST, 'semantic');
        const errors = trackPageErrors(page);
        const workers = trackWorkers(page);
        hub.fail(file, how);
        await page.goto('/index.html');
        const ui = landingUI(page);
        await typeQuery(ui.input, FIRST);

        await test.step('1. the failure message shows within 30 s', () => expectFailureShown(ui, 30_000));
        await test.step('2. the progress bar is hidden', () => expectProgressHidden(ui));
        await test.step('3. keyword results as with semantic search off', () => expectKeywordResults(ui, refs, FIRST));
        await test.step('4. no retry loop', () => expectNoRetryLoop({ ui, site, hub, workers }));
        await test.step('5. retry: one new load, semantic results merge in', async () => {
          hub.clear();
          await expectRetryWorks({ ui, site, workers, query: LAST, semanticRef });
        });
        await test.step('6. no unhandled error reaches the page', () => {
          expect(errors, 'pageerror events').toEqual([]);
        });
      });
    }
  }
});

// ---- C2: the model load stalls ------------------------------------------------------------

test.describe('C2: the model load stalls', () => {
  /** Hold the weights forever, fast-forward the page clock, and check the C1 failure state. */
  async function stall({ page, site, hub, refs }) {
    const workers = trackWorkers(page);
    await page.clock.install();
    const held = hub.hold(WEIGHTS);
    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, FIRST);
    await held.seen;
    await sleep(1500); // the other files finish
    await page.clock.fastForward(170_000);
    await sleep(1000);
    expect(isFailureMessage(await statusText(ui.host)), '170 s without progress: no failure yet').toBe(false);
    expect(liveWorkers(workers), '170 s without progress: the worker still runs').toBe(1);
    await page.clock.fastForward(15_000);
    await expectFailureShown(ui, 10_000);
    await expectProgressHidden(ui);
    await expect.poll(() => liveWorkers(workers), { message: 'the stalled worker is terminated' }).toBe(0);
    await expectKeywordResults(ui, refs, FIRST);
    return { ui, workers };
  }

  test('1. no progress for 180 s: failure message, worker terminated, keyword results', async ({ browser, page, site, hub }) => {
    const refs = await keywordRefs(browser, site, [FIRST, NEXT]);
    const errors = trackPageErrors(page);
    await stall({ page, site, hub, refs });
    expect(errors, 'pageerror events').toEqual([]);
  });

  test('2. growing progress restarts the window: a slow download never fails', async ({ page, site, hub }) => {
    const workers = trackWorkers(page);
    await page.clock.install();
    const steps = ['tokenizer.json', WEIGHTS, DATA].map((file) => ({ file, held: hub.hold(file) }));
    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, FIRST);
    for (const { file, held } of steps) {
      await held.seen;
      await sleep(1000);
      await page.clock.fastForward(150_000);
      await sleep(1000);
      expect(isFailureMessage(await statusText(ui.host)), `150 s after the last progress, before ${file}`).toBe(false);
      expect(liveWorkers(workers), `the worker still runs before ${file}`).toBe(1);
      held.release();
    }
    await site.waitProbe('embed', (p) => p.text.endsWith(FIRST), 30_000);
    expect(isFailureMessage(await statusText(ui.host)), 'no failure after the download completes').toBe(false);
    expect(workers.length, 'workers started').toBe(1);
  });

  test('3. retry after a stall works as in C1.5', async ({ browser, page, site, hub }) => {
    const refs = await keywordRefs(browser, site, [FIRST, NEXT]);
    const semanticRef = await reference(browser, site, NEXT, 'semantic');
    const { ui, workers } = await stall({ page, site, hub, refs });
    hub.clear();
    await expectRetryWorks({ ui, site, workers, query: NEXT, semanticRef });
  });
});

// ---- C3: a query never gets a semantic answer ----------------------------------------------

test.describe('C3: a query never gets a semantic answer', () => {
  test('1. 120 s after the query: failure message, keyword results stay', async ({ browser, page, site, hub }) => {
    const refs = await keywordRefs(browser, site, [FIRST, NEXT]);
    const errors = trackPageErrors(page);
    await page.clock.install();
    const embed = site.gate('embed', () => true); // no embedding call ever returns
    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, FIRST);
    await embed.seen;
    await page.clock.fastForward(110_000);
    await sleep(1000);
    expect(isFailureMessage(await statusText(ui.host)), '110 s: no failure yet').toBe(false);
    expect(await stableSnapshot(ui.listbox, { quietMs: 1000 }), 'keyword results while waiting').toEqual(refs[FIRST]);
    await page.clock.fastForward(15_000);
    await expectFailureShown(ui, 10_000);
    await expectKeywordResults(ui, refs, FIRST);
    expect(errors, 'pageerror events').toEqual([]);
  });

  test('2. a newer query replaces the old one: only the newest is waited for', async ({ browser, page, site, hub }) => {
    const refs = await keywordRefs(browser, site, [FIRST, NEXT]);
    await page.clock.install();
    const embed = site.gate('embed', () => true);
    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, FIRST);
    await embed.seen;
    await page.clock.fastForward(100_000);
    await typeQuery(ui.input, NEXT);
    await sleep(1500); // the debounced query reaches the worker
    await page.clock.fastForward(30_000); // 130 s after the first query, 30 s after the newer one
    await sleep(1000);
    expect(isFailureMessage(await statusText(ui.host)), 'the replaced query does not time out').toBe(false);
    expect(await stableSnapshot(ui.listbox, { quietMs: 1000 })).toEqual(refs[NEXT]);
    await page.clock.fastForward(100_000); // 130 s after the newer query
    await expectFailureShown(ui, 10_000);
    expect(await stableSnapshot(ui.listbox, { quietMs: 1000 })).toEqual(refs[NEXT]);
  });
});

// ---- C4: WebGPU fails after the model loaded -----------------------------------------------

test.describe('C4: WebGPU fails after the model loaded', () => {
  test('1. one fresh WASM worker loads the model and the semantic answer merges in', async ({ browser, page, site, hub }) => {
    const semanticRef = await reference(browser, site, FIRST, 'semantic');
    const errors = trackPageErrors(page);
    site.state.control = { webgpu: 'adapter', failDevices: [], embedFailDevices: ['webgpu'] };
    const workers = trackWorkers(page);
    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, FIRST);
    await site.waitProbe('embed', (p) => p.device === 'wasm' && p.text.endsWith(FIRST), 30_000);
    expect(await stableSnapshot(ui.listbox, { quietMs: 1000 })).toEqual(semanticRef);
    await sleep(2000);
    expect(site.probes('pipeline').map((p) => p.params.device), 'load attempts').toEqual(['webgpu', 'wasm']);
    expect(workers.length, 'workers started').toBe(2);
    expect(liveWorkers(workers), 'live workers').toBe(1);
    expect(await statusText(ui.host), 'the status reports the WASM backend').toMatch(/wasm/i);
    expect(errors, 'pageerror events').toEqual([]);
  });

  for (const [name, control] of [
    ['the WASM load fails', { failDevices: ['wasm'], embedFailDevices: ['webgpu'] }],
    ['the WASM embedding also throws', { failDevices: [], embedFailDevices: ['webgpu', 'wasm'] }],
  ]) {
    test(`2. ${name}: C1 applies, no further fallback`, async ({ browser, page, site, hub }) => {
      const refs = await keywordRefs(browser, site, [FIRST, NEXT]);
      const errors = trackPageErrors(page);
      site.state.control = { webgpu: 'adapter', ...control };
      const workers = trackWorkers(page);
      await page.goto('/index.html');
      const ui = landingUI(page);
      await typeQuery(ui.input, FIRST);
      await expectFailureShown(ui, 30_000);
      await expectProgressHidden(ui);
      await expectKeywordResults(ui, refs, FIRST);
      await expectNoRetryLoop({ ui, site, hub, workers });
      expect(site.probes('pipeline').map((p) => p.params.device), 'load attempts').toEqual(['webgpu', 'wasm']);
      expect(workers.length, 'workers started: one WebGPU, one WASM').toBe(2);
      expect(isFailureMessage(await statusText(ui.host)), 'the failure message stays').toBe(true);
      await expect(retryControl(ui), 'a control to retry semantic search').toBeVisible();
      expect(errors, 'pageerror events').toEqual([]);
    });
  }
});

// ---- C5: only the text model loads ---------------------------------------------------------

test.describe('C5: only the text model loads', () => {
  // A config with the vision and audio encoders of the multimodal checkpoint. transformers.js
  // requests their ONNX files unless the page drops these configs.
  const multimodalConfig = () => {
    const config = JSON.parse(fs.readFileSync(path.join(TINY_MODEL, CONFIG), 'utf8'));
    return JSON.stringify({
      ...config,
      model_type: 'embedding_gemma2',
      architectures: ['EmbeddingGemma2Model'],
      vision_config: { model_type: 'gemma3n_vision' },
      audio_config: { model_type: 'gemma3n_audio' },
      'transformers.js_config': {
        use_external_data_format: {
          'model_quantized.onnx': 1, 'vision_encoder_quantized.onnx': 1, 'audio_encoder_quantized.onnx': 1,
        },
      },
    });
  };
  const encoderRequests = (hub) => hub.log
    .filter((r) => /vision|audio/i.test(r.file) && /\.onnx(_data)?$/.test(r.file))
    .map((r) => `${r.method} ${r.file}${r.range ? ` (Range: ${r.range})` : ''}`);

  /** Wait until the load attempt ends: a session exists or the page left the loading state. */
  async function loadSettles(ui, site) {
    await expect.poll(async () => site.probes('session').length > 0
      || (!/loading/i.test(await statusText(ui.host)) && (await statusText(ui.host)) !== ''), { timeout: 30_000 }).toBe(true);
    await sleep(2000);
  }

  test('WASM backend', async ({ page, site, hub }) => {
    hub.serve(CONFIG, multimodalConfig());
    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, FIRST);
    await loadSettles(ui, site);
    expect(hub.downloads(WEIGHTS), 'the text model was requested').toBeGreaterThan(0);
    expect(encoderRequests(hub)).toEqual([]);
  });

  test('WebGPU backend, then the WASM fallback', async ({ page, site, hub }) => {
    hub.serve(CONFIG, multimodalConfig());
    site.state.control = { webgpu: 'adapter', failDevices: [] };
    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, FIRST);
    await site.waitProbe('pipeline', (p) => p.device === 'webgpu', 30_000);
    await loadSettles(ui, site);
    await sleep(3000);
    expect(hub.downloads(WEIGHTS), 'the text model was requested').toBeGreaterThan(0);
    expect(encoderRequests(hub)).toEqual([]);
  });

  test('after a failure and a retry', async ({ page, site, hub }) => {
    hub.serve(CONFIG, multimodalConfig());
    hub.fail(WEIGHTS, 'abort');
    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, FIRST);
    await expectFailureShown(ui, 30_000);
    hub.clear();
    hub.serve(CONFIG, multimodalConfig());
    const attempts = site.probes('pipeline').length;
    await expect(retryControl(ui), 'a control to retry semantic search').toBeVisible();
    await retryControl(ui).first().click();
    await expect.poll(() => site.probes('pipeline').length, { timeout: 30_000 }).toBeGreaterThan(attempts);
    await sleep(3000);
    expect(hub.downloads(WEIGHTS), 'the text model was requested').toBeGreaterThan(1);
    expect(encoderRequests(hub)).toEqual([]);
  });
});

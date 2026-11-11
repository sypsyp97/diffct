// Actual search-ui.js entrypoint + DOM events, with a finite test index and controlled Worker.
// This fixture does not claim to be a built Sphinx page or to exercise model inference.
import { test, expect } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';

const STATIC = new URL('../../source/_static/', import.meta.url);
const FIXTURE = new URL('./fixtures/runtime-cancel/search.html', import.meta.url);
const ORIGIN = 'https://runtime-cancel.test';
const CHUNKS = [
  { u: 'alpha.html', p: 'Alpha guide', s: '', t: 'Alpha fixture keyword result.', c: '', o: [0] },
  { u: 'beta.html', p: 'Beta guide', s: '', t: 'Beta fixture keyword result.', c: '', o: [0] },
  { u: 'gamma.html', p: 'Gamma guide', s: '', t: 'Gamma fixture semantic result.', c: '', o: [0] },
];

// createIndex's format: chunk, sentence, then page vectors; float32 scales followed by int8 codes.
function vectors() {
  const count = CHUNKS.length * 3;
  const buffer = new ArrayBuffer(count * (4 + 2));
  new Float32Array(buffer, 0, count).fill(1);
  const codes = new Int8Array(buffer, count * 4);
  for (let i = 0; i < count; i++) codes.set(i % 3 === 2 ? [0, 1] : [1, 0], i * 2);
  return Buffer.from(buffer);
}

async function openFixture(page) {
  // Stryker's command runner selects the real source mutant in this browser realm.
  if (process.env.__STRYKER_ACTIVE_MUTANT__ !== undefined) {
    await page.addInitScript((id) => { globalThis.__stryker__ = { activeMutant: id }; }, process.env.__STRYKER_ACTIVE_MUTANT__);
  }
  await page.addInitScript(() => {
    localStorage.setItem('diffct-search', JSON.stringify({ semantic: 'on' }));
    const workers = [];
    window.searchWorkerControl = {
      messages: [],
      deliver(data) {
        const worker = workers.at(-1);
        if (!worker || worker.terminated) throw new Error('No live fixture worker');
        worker.onmessage({ data });
      },
    };
    window.Worker = class {
      constructor(url, options) {
        this.url = String(url);
        this.options = options;
        this.terminated = false;
        workers.push(this);
      }
      postMessage(message) { window.searchWorkerControl.messages.push(message); }
      terminate() { this.terminated = true; }
    };
  });
  await page.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== ORIGIN) return route.abort('blockedbyclient');
    if (url.pathname === '/search.html') {
      return route.fulfill({ contentType: 'text/html', body: await readFile(FIXTURE) });
    }
    if (['search-ui.js', 'search-core.js'].some((name) => url.pathname === '/_static/' + name)) {
      return route.fulfill({ contentType: 'text/javascript', body: await readFile(new URL(url.pathname.split('/').at(-1), STATIC)) });
    }
    if (url.pathname === '/_static/search-meta.json') {
      return route.fulfill({ json: { dim: 2, query_prefix: 'fixture: ' } });
    }
    if (url.pathname === '/_static/search-chunks.json') return route.fulfill({ json: CHUNKS });
    if (url.pathname === '/_static/search-vectors.bin') {
      return route.fulfill({ contentType: 'application/octet-stream', body: vectors() });
    }
    return route.fulfill({ status: 404, body: 'Unknown test fixture asset' });
  });
  await page.coverage.startJSCoverage({ resetOnNavigation: false });
  await page.goto(ORIGIN + '/search.html');
  const ui = (selector) => ({
    input: page.locator(selector + ' input[name="q"]'),
    host: page.locator(selector + ' .srch-host'),
  });
  const dialog = ui('#pst-search-dialog');
  const searchPage = ui('.bd-search-container');
  await expect(dialog.input).toHaveAttribute('role', 'combobox');
  await expect(searchPage.input).toHaveAttribute('role', 'combobox');
  return { dialog, searchPage };
}

const options = (ui) => ui.host.getByRole('option');
const tags = (ui) => ui.host.locator('.srch-tag');
async function embeds(page) {
  return page.evaluate(() => window.searchWorkerControl.messages.filter((m) => m.type === 'embed'));
}
async function waitEmbeds(page, count) { await expect.poll(async () => (await embeds(page)).length).toBe(count); }
async function deliver(page, data) { await page.evaluate((message) => window.searchWorkerControl.deliver(message), data); }
async function ready(page) {
  await expect.poll(() => page.evaluate(() => window.searchWorkerControl.messages.filter((m) => m.type === 'load').length)).toBe(1);
  await deliver(page, { type: 'ready', backend: 'fixture' });
}
async function answer(page, request, vector = [0, 1]) {
  await page.evaluate(({ id, vector }) => window.searchWorkerControl.deliver({ type: 'vector', id, vector: new Float32Array(vector) }), { id: request.id, vector });
}
async function expectKeyword(ui, title) {
  await expect(options(ui)).toHaveCount(1);
  await expect(options(ui)).toContainText(title);
  await expect(tags(ui)).toHaveText(['Keyword']);
}

test.afterEach(async ({ page }, testInfo) => {
  const coverage = await page.coverage.stopJSCoverage();
  await writeFile(testInfo.outputPath('js-coverage.json'), JSON.stringify(coverage));
});

test('a current request canceled by the other host settles and preserves its keyword results', async ({ page }) => {
  const { dialog, searchPage } = await openFixture(page);
  await dialog.input.fill('alpha');
  await expectKeyword(dialog, 'Alpha guide');
  await searchPage.input.fill('beta');
  await expectKeyword(searchPage, 'Beta guide');
  await expect(dialog.host).toHaveAttribute('data-state', 'pending');
  await expect(searchPage.host).toHaveAttribute('data-state', 'pending');

  // Both real ready listeners run. The second embed cancels the first global pending request.
  await ready(page);
  await waitEmbeds(page, 2);
  const [canceled, current] = await embeds(page);
  expect([canceled.text, current.text]).toEqual(['fixture: alpha', 'fixture: beta']);
  await expect(dialog.host).toHaveAttribute('data-state', 'ready');
  await expect(searchPage.host).toHaveAttribute('data-state', 'pending');
  await expectKeyword(dialog, 'Alpha guide');

  // A canceled worker response cannot publish through either host.
  await answer(page, canceled);
  await expectKeyword(dialog, 'Alpha guide');
  await expectKeyword(searchPage, 'Beta guide');
  await expect(searchPage.host).toHaveAttribute('data-state', 'pending');
  await answer(page, current);
  await expect(searchPage.host).toHaveAttribute('data-state', 'ready');
  await expect(searchPage.host.locator('.srch-tag-semantic')).not.toHaveCount(0);
  await expectKeyword(dialog, 'Alpha guide');
});

test('an obsolete same-host cancellation cannot clear the newer request or publish an old result', async ({ page }) => {
  const { searchPage } = await openFixture(page);
  await searchPage.input.fill('alpha');
  await expectKeyword(searchPage, 'Alpha guide');
  await ready(page);
  await waitEmbeds(page, 1);
  const [old] = await embeds(page);
  await searchPage.input.fill('beta');
  await expectKeyword(searchPage, 'Beta guide');
  await waitEmbeds(page, 2);
  const [, current] = await embeds(page);
  await expect(searchPage.host).toHaveAttribute('data-state', 'pending');
  await answer(page, old);
  await expectKeyword(searchPage, 'Beta guide');
  await expect(searchPage.host).toHaveAttribute('data-state', 'pending');
  await answer(page, current);
  await expect(searchPage.host).toHaveAttribute('data-state', 'ready');
  await expect(searchPage.host.locator('.srch-tag-semantic')).not.toHaveCount(0);
});

test('an obsolete success before the replacement debounce preserves the newer keyword query and pending state', async ({ page }) => {
  await page.clock.install({ time: new Date('2026-11-11T00:00:00Z') });
  await page.clock.pauseAt(new Date('2026-11-11T00:01:00Z'));
  const { searchPage } = await openFixture(page);
  await searchPage.input.fill('alpha');
  await expectKeyword(searchPage, 'Alpha guide');
  await ready(page);
  await waitEmbeds(page, 1);
  const [old] = await embeds(page);
  await searchPage.input.fill('beta');
  await expectKeyword(searchPage, 'Beta guide');
  await answer(page, old);
  await expectKeyword(searchPage, 'Beta guide');
  await expect(searchPage.host).toHaveAttribute('data-state', 'pending');
  expect(await embeds(page)).toHaveLength(1);
  await page.clock.runFor(250);
  await waitEmbeds(page, 2);
  const [, current] = await embeds(page);
  await answer(page, current);
  await expect(searchPage.host).toHaveAttribute('data-state', 'ready');
  await expect(searchPage.host.locator('.srch-tag-semantic')).not.toHaveCount(0);
});

test('a current embedding failure settles while retaining keyword results', async ({ page }) => {
  const { searchPage } = await openFixture(page);
  await searchPage.input.fill('beta');
  await expectKeyword(searchPage, 'Beta guide');
  await ready(page);
  await waitEmbeds(page, 1);
  const [current] = await embeds(page);
  await deliver(page, { type: 'error', id: current.id, message: 'fixture embedding rejected' });
  await expect(searchPage.host).toHaveAttribute('data-state', 'ready');
  await expectKeyword(searchPage, 'Beta guide');
});

test('stopping a loading model settles both waiting hosts without embedding', async ({ page }) => {
  const { dialog, searchPage } = await openFixture(page);
  await dialog.input.fill('alpha');
  await searchPage.input.fill('beta');
  await expectKeyword(dialog, 'Alpha guide');
  await expectKeyword(searchPage, 'Beta guide');
  await dialog.host.getByRole('button', { name: 'Keyword only', exact: true }).click();
  await expect(dialog.host).toHaveAttribute('data-state', 'off');
  await expect(searchPage.host).toHaveAttribute('data-state', 'off');
  await expectKeyword(dialog, 'Alpha guide');
  await expectKeyword(searchPage, 'Beta guide');
  expect(await embeds(page)).toEqual([]);
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('diffct-search')).semantic)).toBe('off');
});

test('a model failure retires a debounced query without making its settled host pending again', async ({ page }) => {
  await page.clock.install({ time: new Date('2026-11-11T00:00:00Z') });
  await page.clock.pauseAt(new Date('2026-11-11T00:01:00Z'));
  const { searchPage } = await openFixture(page);
  await searchPage.input.fill('alpha');
  await expectKeyword(searchPage, 'Alpha guide');
  await ready(page);
  await waitEmbeds(page, 1);
  const [first] = await embeds(page);
  await answer(page, first);
  await expect(searchPage.host).toHaveAttribute('data-state', 'ready');

  await searchPage.input.fill('beta');
  await expectKeyword(searchPage, 'Beta guide');
  await deliver(page, { type: 'error', message: 'fixture model failed' });
  await expect(searchPage.host).toHaveAttribute('data-state', 'error');
  // Retain the observable attribute history: a brief pending transition also counts.
  await searchPage.host.evaluate((host) => {
    window.searchHostStates = [host.dataset.state];
    new MutationObserver((records) => {
      window.searchHostStates.push(...records.map((record) => record.oldValue), host.dataset.state);
    }).observe(host, { attributes: true, attributeFilter: ['data-state'], attributeOldValue: true });
  });
  await page.clock.runFor(250);
  await expect(searchPage.host).toHaveAttribute('data-state', 'error');
  await expectKeyword(searchPage, 'Beta guide');
  expect(await page.evaluate(() => window.searchHostStates)).not.toContain('pending');
  expect(await embeds(page)).toHaveLength(1);
});

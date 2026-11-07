// Semantic search requires consent on every search surface. Connection hints do not add
// another consent state. These tests deliberately use fresh storage, without preference seeding.
import {
  test, expect, landingUI, dialogUI, searchPageUI, typeQuery, stableSnapshot,
  enableSemanticButton, trackWorkers, sleep,
} from './harness.mjs';

async function openSearch(page, where, reload = false) {
  if (reload) await page.reload();
  else await page.goto(where === 'landing' ? '/index.html' : where === 'dialog' ? '/quickstart.html' : '/search.html');
  if (where === 'dialog') {
    await page.keyboard.press('ControlOrMeta+k');
    const ui = dialogUI(page);
    await expect(ui.dialog).toBeVisible();
    return ui;
  }
  return where === 'landing' ? landingUI(page) : searchPageUI(page);
}

async function expectKeywordOnly(ui, site, hub, workers) {
  expect((await stableSnapshot(ui.listbox)).length, 'keyword results').toBeGreaterThan(0);
  await expect(enableSemanticButton(ui.host)).toBeVisible();
  await expect(ui.host.getByRole('button', { name: 'Download', exact: true })).toHaveCount(0);
  const lb = await ui.listbox.locator();
  await expect(lb.getByText(/^(Semantic|Both)$/)).toHaveCount(0);
  // Observe beyond the query debounce, including any delayed automatic model load.
  await sleep(1500);
  expect(hub.log, 'all Hub requests, including metadata probes').toEqual([]);
  expect(site.probes('pipeline'), 'model load attempts').toEqual([]);
  expect(site.probes('session'), 'model sessions').toEqual([]);
  expect(site.probes('embed'), 'semantic queries').toEqual([]);
  expect(workers, 'workers before consent').toEqual([]);
}

async function expectSemantic(ui) {
  const lb = await ui.listbox.locator();
  await expect(lb.getByText(/^(Semantic|Both)$/).first()).toBeVisible({ timeout: 30_000 });
}

for (const where of ['landing', 'dialog', 'page']) {
  test(`fresh visitor: keyword results and zero model requests (${where})`, async ({ page, site, hub }) => {
    const workers = trackWorkers(page);
    const ui = await openSearch(page, where);
    expect(await page.evaluate(() => localStorage.getItem('diffct-search'))).toBeNull();
    await typeQuery(ui.input, 'fan beam');
    await expectKeywordOnly(ui, site, hub, workers);
    await typeQuery(ui.input, 'cone beam');
    await expectKeywordOnly(ui, site, hub, workers);
  });

  test(`enable click merges the current query and persists across reload (${where})`, async ({ page, site, hub }) => {
    const workers = trackWorkers(page);
    let ui = await openSearch(page, where);
    await typeQuery(ui.input, 'fan beam');
    await expectKeywordOnly(ui, site, hub, workers);
    await enableSemanticButton(ui.host).click();
    await expect.poll(() => page.evaluate(() => JSON.parse(localStorage.getItem('diffct-search')))).toMatchObject({ semantic: 'on' });
    await site.waitProbe('embed', (p) => p.text.endsWith('fan beam'));
    await expectSemantic(ui);
    await expect(ui.input).toHaveValue('fan beam');
    expect(hub.downloads('onnx/model_quantized.onnx')).toBe(1);
    expect(workers.length, 'model loads in one Web Worker').toBe(1);
    expect(site.probes('session').length).toBe(1);

    ui = await openSearch(page, where, true);
    await typeQuery(ui.input, 'curved detector');
    await site.waitProbe('embed', (p) => p.text.endsWith('curved detector'));
    await expectSemantic(ui);
    expect(site.probes('session').length, 'reload automatically loads a new session').toBe(2);
    expect(await page.evaluate(() => JSON.parse(localStorage.getItem('diffct-search')).semantic)).toBe('on');
  });
}

for (const connection of [{ saveData: true, type: 'wifi' }, { saveData: false, type: 'cellular' }]) {
  test(`connection ${JSON.stringify(connection)} uses the same enable button without Download consent`, async ({ page, site, hub }) => {
    await page.addInitScript((value) => {
      Object.defineProperty(navigator, 'connection', { configurable: true, value });
    }, connection);
    const workers = trackWorkers(page);
    const ui = await openSearch(page, 'landing');
    await typeQuery(ui.input, 'fan beam');
    await expectKeywordOnly(ui, site, hub, workers);
    await enableSemanticButton(ui.host).click();
    await site.waitProbe('embed', (p) => p.text.endsWith('fan beam'));
    await expectSemantic(ui);
    await expect(ui.host.getByRole('button', { name: 'Download', exact: true })).toHaveCount(0);
  });
}

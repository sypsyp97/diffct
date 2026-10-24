// Contract item 2 (single model load): concurrent queries during model download trigger exactly
// one download and one model session; reloading the page uses the browser cache (no second
// network download of model files).
import {
  test, expect, dialogUI, landingUI, typeQuery, stableSnapshot, trackWorkers, sleep,
} from './harness.mjs';

const WEIGHTS = 'onnx/model_quantized.onnx';
const FILES = ['onnx/model_quantized.onnx_data', 'tokenizer.json', 'tokenizer_config.json'];

async function queriesDuringDownload(page, site, hub) {
  const held = hub.hold(WEIGHTS);
  await page.goto('/quickstart.html');
  const ui = dialogUI(page);
  await ui.trigger.click();
  await expect(ui.dialog).toBeVisible();
  await typeQuery(ui.input, 'fan beam');
  await held.seen; // the download is in progress
  for (const q of ['cone beam', 'multi GPU', 'install CUDA']) {
    await typeQuery(ui.input, q);
    await sleep(300);
  }
  await ui.input.press('Escape');
  await expect(ui.dialog).toBeHidden();
  await ui.trigger.click();
  await expect(ui.dialog).toBeVisible();
  await typeQuery(ui.input, 'iterative reconstruction');
  await sleep(500);
  held.release();
  await site.waitProbe('embed', (p) => p.text.endsWith('iterative reconstruction'));
  await stableSnapshot(ui.listbox);
  return ui;
}

test.describe('item 2: single model load', () => {
  test('concurrent queries during the download: one download of each file, one session, one worker', async ({ page, site, hub }) => {
    const workers = trackWorkers(page);
    await queriesDuringDownload(page, site, hub);
    await sleep(1000);
    expect(hub.downloads(WEIGHTS), `downloads: ${JSON.stringify(hub.downloadLog())}`).toBe(1);
    for (const f of FILES) expect(hub.downloads(f), f).toBe(1);
    expect(site.probes('session').length, 'model sessions').toBe(1);
    expect(workers.length, 'model workers').toBe(1);
  });

  test('no model file downloads twice, across revisions', async ({ page, site, hub }) => {
    await queriesDuringDownload(page, site, hub);
    await sleep(1000);
    const files = [...new Set(hub.log.map((r) => r.file))];
    const twice = files.filter((f) => hub.downloads(f) > 1);
    expect(twice, `downloads: ${JSON.stringify(hub.downloadLog(), null, 1)}`).toEqual([]);
  });

  test('reload reads the model from the browser cache, with no network download', async ({ page, site, hub }) => {
    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, 'fan beam');
    await site.waitProbe('embed', (p) => p.text.endsWith('fan beam'));
    await stableSnapshot(ui.listbox);
    const before = hub.downloadLog().length;
    const sessionsBefore = site.probes('session').length;

    await page.reload();
    const again = landingUI(page);
    await typeQuery(again.input, 'cone beam');
    await site.waitProbe('embed', (p) => p.text.endsWith('cone beam'));
    await stableSnapshot(again.listbox);
    expect(site.probes('session').length, 'the reloaded page has a model session').toBe(sessionsBefore + 1);
    expect(hub.downloadLog().slice(before), 'model downloads after reload').toEqual([]);
  });
});

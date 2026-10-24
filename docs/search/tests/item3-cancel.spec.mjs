// Contract item 3 (cancel): "Keyword only" during download stops further model network requests;
// the choice persists across reloads; turning semantic search back on resumes cleanly.
import {
  test, expect, landingUI, typeQuery, snapshot, stableSnapshot, statusText, trackWorkers, trackConsole, sleep,
} from './harness.mjs';

const WEIGHTS = 'onnx/model_quantized.onnx';

test.describe('item 3: cancel', () => {
  test('"Keyword only" during download stops model requests, persists across reload, and resumes cleanly', async ({ page, site, hub }) => {
    const workers = trackWorkers(page);
    const errors = trackConsole(page);
    const held = hub.hold(WEIGHTS);
    await page.goto('/index.html');
    let ui = landingUI(page);
    await typeQuery(ui.input, 'fan beam');
    await held.seen;

    let requestsAtCancel = 0;
    await test.step('"Keyword only" stops further model requests', async () => {
      await ui.host.getByRole('button', { name: /keyword only/i }).click();
      requestsAtCancel = hub.log.length;
      held.release(); // the request in flight may finish; nothing new may start
      await sleep(2000);
      await typeQuery(ui.input, 'cone beam');
      await sleep(2000);
      expect(hub.log.slice(requestsAtCancel).map((r) => r.url), 'model requests after "Keyword only"').toEqual([]);
      expect(site.probes('session').length, 'model sessions').toBe(0);
      expect((await stableSnapshot(ui.listbox)).length, 'keyword results').toBeGreaterThan(0);
    });

    await test.step('the choice persists across a reload', async () => {
      const workersBefore = workers.length;
      await page.reload();
      ui = landingUI(page);
      await typeQuery(ui.input, 'multi GPU');
      await sleep(3000);
      expect(hub.log.slice(requestsAtCancel).map((r) => r.url), 'model requests after reload').toEqual([]);
      expect(workers.length - workersBefore, 'model workers started after reload').toBe(0);
      expect((await stableSnapshot(ui.listbox)).length, 'keyword results').toBeGreaterThan(0);
      await expect(ui.host.getByRole('button', { name: /semantic/i })).toBeVisible();
    });

    await test.step('turning semantic search back on resumes cleanly', async () => {
      const errorsBefore = errors.length;
      const downloadsBefore = hub.downloads(WEIGHTS);
      await ui.host.getByRole('button', { name: /semantic/i }).click();
      await site.waitProbe('session', () => true, 30_000);
      await site.waitProbe('embed', (p) => p.text.endsWith('multi GPU'), 30_000);
      await stableSnapshot(ui.listbox);
      expect(site.probes('session').length, 'model sessions').toBe(1);
      expect(hub.downloads(WEIGHTS) - downloadsBefore, 'weights downloads after resume').toBe(1);
      expect(workers.filter((w) => !w.closed).length, 'live model workers').toBe(1);
      expect(errors.slice(errorsBefore), 'console errors after resume').toEqual([]);
      const status = await statusText(ui.host);
      expect(status).not.toMatch(/does not work|failed|error|keyword search only/i);
      // A new query gets a semantic answer.
      await typeQuery(ui.input, 'curved detector');
      await site.waitProbe('embed', (p) => p.text.endsWith('curved detector'), 15_000);
    });

    await test.step('after resume the choice persists as "on" across a reload', async () => {
      await page.reload();
      ui = landingUI(page);
      const sessions = site.probes('session').length;
      await typeQuery(ui.input, 'fan beam');
      await site.waitProbe('embed', (p) => p.text.endsWith('fan beam') && site.probes('session').length > sessions, 30_000);
      expect(site.probes('session').length).toBe(sessions + 1);
    });
  });

  test('cancel and turn back on in the same page: one live worker, one session, semantic answers', async ({ page, site, hub }) => {
    const workers = trackWorkers(page);
    const errors = trackConsole(page);
    const held = hub.hold(WEIGHTS);
    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, 'fan beam');
    await held.seen;
    await ui.host.getByRole('button', { name: /keyword only/i }).click();
    await expect(ui.host.getByRole('button', { name: /semantic/i })).toBeVisible();
    hub.clear(); // later downloads go through at once
    await ui.host.getByRole('button', { name: /semantic/i }).click();
    held.release();
    await site.waitProbe('embed', (p) => p.text.endsWith('fan beam'), 30_000);
    await stableSnapshot(ui.listbox);
    await sleep(1000);
    expect(site.probes('session').length, 'model sessions').toBe(1);
    expect(workers.filter((w) => !w.closed).length, 'live model workers').toBe(1);
    expect(errors, 'console errors').toEqual([]);
    expect(await snapshot(ui.listbox)).not.toEqual([]);
  });
});

// Contract item 5 (errors): model load failure (network error, WebGPU init failure) shows a plain
// message, keeps keyword results working, and does not retry in a loop.
import {
  test, expect, landingUI, typeQuery, stableSnapshot, statusText, expectPlainMessage, reference,
  trackWorkers, sleep,
} from './harness.mjs';

const WEIGHTS = 'onnx/model_quantized.onnx';
const NEXT = 'cone beam geometry';

/** Type a query, wait for the failure message, then check keyword results and retries. */
async function expectGracefulFailure({ browser, page, site, hub, setup }) {
  const keywordNext = await reference(browser, site, NEXT, 'keyword');
  await setup();
  const workers = trackWorkers(page);
  await page.goto('/index.html');
  const ui = landingUI(page);
  await typeQuery(ui.input, 'fan beam');

  let message = '';
  await test.step('a plain message shows', async () => {
    await expect.poll(async () => {
      message = await statusText(ui.host);
      return message && !/loading/i.test(message) ? message : '';
    }, { timeout: 30_000 }).not.toBe('');
    expectPlainMessage(message);
    await expect(ui.host.getByText(message.slice(0, 40), { exact: false }).first()).toBeVisible();
  });

  await test.step('keyword results keep working', async () => {
    expect((await stableSnapshot(ui.listbox)).length).toBeGreaterThan(0);
    await typeQuery(ui.input, NEXT);
    expect(await stableSnapshot(ui.listbox)).toEqual(keywordNext);
  });

  await test.step('no retry loop', async () => {
    const requests = hub.log.length;
    const attempts = site.probes('pipeline').length;
    const started = workers.length;
    await sleep(5000);
    expect(hub.log.slice(requests).map((r) => r.url), 'model requests while idle').toEqual([]);
    expect(site.probes('pipeline').length - attempts, 'model load attempts while idle').toBe(0);
    expect(workers.length - started, 'workers started while idle').toBe(0);
    for (const q of ['multi GPU', 'install CUDA', 'gradient']) {
      await typeQuery(ui.input, q);
      await sleep(700);
    }
    // At most one more attempt (one worker, one pipeline call per backend).
    expect(site.probes('pipeline').length - attempts, 'pipeline calls during 3 more queries').toBeLessThanOrEqual(2);
    expect(workers.length - started, 'workers started during 3 more queries').toBeLessThanOrEqual(1);
  });
  return { message, ui };
}

test.describe('item 5: errors', () => {
  test('network error on the model weights', async ({ browser, page, site, hub }) => {
    await expectGracefulFailure({ browser, page, site, hub, setup: () => hub.fail(WEIGHTS, 'abort') });
    expect(site.probes('session').length).toBe(0);
  });

  test('HTTP 503 on the tokenizer', async ({ browser, page, site, hub }) => {
    await expectGracefulFailure({ browser, page, site, hub, setup: () => hub.fail('tokenizer.json', 503) });
  });

  test('network error on the model config', async ({ browser, page, site, hub }) => {
    await expectGracefulFailure({ browser, page, site, hub, setup: () => hub.fail('config.json', 'abort') });
  });

  test('WebGPU init failure and WASM failure', async ({ browser, page, site, hub }) => {
    await expectGracefulFailure({
      browser, page, site, hub,
      setup: () => { site.state.control = { webgpu: 'adapter', failDevices: ['webgpu', 'wasm'] }; },
    });
  });

  test('WebGPU init failure with a working WASM fallback: no loop, semantic or a plain message', async ({ page, site, hub }) => {
    site.state.control = { webgpu: 'adapter', failDevices: ['webgpu'] };
    const workers = trackWorkers(page);
    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, 'fan beam');
    await expect.poll(async () => site.probes('session').length > 0
      || !/loading/i.test(await statusText(ui.host)) && (await statusText(ui.host)) !== '', { timeout: 30_000 }).toBe(true);
    await sleep(3000);
    const devices = site.probes('pipeline').map((p) => p.params.device);
    expect(devices.filter((d) => d === 'webgpu').length, `load attempts: ${devices}`).toBe(1);
    expect(devices.length, `load attempts: ${devices}`).toBeLessThanOrEqual(2);
    // One fallback is allowed (for example a fresh worker for WASM), not more.
    expect(workers.length, 'workers started').toBeLessThanOrEqual(2);
    expect(workers.filter((w) => !w.closed).length, 'live workers').toBeLessThanOrEqual(1);
    if (site.probes('session').length) {
      await site.waitProbe('embed', (p) => p.text.endsWith('fan beam'));
    } else {
      expectPlainMessage(await statusText(ui.host));
    }
    expect((await stableSnapshot(ui.listbox)).length).toBeGreaterThan(0);
  });

  test('WebGPU adapter request throws: no loop, semantic or a plain message', async ({ page, site, hub }) => {
    site.state.control = { webgpu: 'reject', failDevices: [] };
    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, 'fan beam');
    await expect.poll(async () => site.probes('session').length > 0
      || !/loading/i.test(await statusText(ui.host)) && (await statusText(ui.host)) !== '', { timeout: 30_000 }).toBe(true);
    await sleep(3000);
    expect(site.probes('pipeline').length).toBeLessThanOrEqual(1);
    if (site.probes('session').length) {
      await site.waitProbe('embed', (p) => p.text.endsWith('fan beam'));
    } else {
      expectPlainMessage(await statusText(ui.host));
    }
    expect((await stableSnapshot(ui.listbox)).length).toBeGreaterThan(0);
  });
});

// Contract item 4 (no leaks): after 200 queries (typed, cleared, dialog opened/closed 50 times) the
// page JS heap and the worker heap do not grow beyond a small bound vs after the first 20 queries;
// event listener count and DOM node count stay bounded; closing the dialog removes its listeners;
// navigating away terminates the worker.
import {
  test, expect, dialogUI, typeQuery, clearQuery, stableSnapshot, cdpWithWorkers, trackWorkers, sleep,
} from './harness.mjs';

// Bounds fixed before the first run. Both measurements are taken in the same UI state
// (dialog closed, input empty) after forced garbage collection.
const HEAP_BOUND = 1024 * 1024; // bytes, page and worker each
const NODE_BOUND = 50;
const LISTENER_BOUND = 10;
// Chrome's `Nodes` counter also counts nodes of parsed helper documents (for example the
// snippets of the theme's native search), so it moves by a few dozen nodes before it levels
// off. "Bounded" is checked as: the nodes in the page document are the same after 20 and after
// 200 queries, and the counter grows by at most NODE_BOUND over the second half of the run.

const WORDS = ['fan beam', 'cone beam', 'gradient', 'multi GPU', 'curved detector', 'install CUDA',
  'iterative reconstruction', 'trajectory', 'FDK weights', 'parallel projector'];

async function measure(probe) {
  const [sessionId] = [...probe.workers.keys()];
  expect(sessionId, 'the model worker is attached').toBeTruthy();
  const pageHeap = await probe.pageHeap();
  const workerHeap = await probe.workerHeap(sessionId);
  const m = await probe.metrics();
  const documentNodes = await probe.cdp.send('Runtime.evaluate', {
    expression: `(() => { let c = 0; const w = document.createTreeWalker(document, NodeFilter.SHOW_ALL); while (w.nextNode()) c++; return c; })()`,
    returnByValue: true,
  }).then((r) => r.result.value);
  return { pageHeap, workerHeap, nodes: m.Nodes, documentNodes, listeners: m.JSEventListeners };
}

test.describe('item 4: no leaks', () => {
  test('200 queries and 50 dialog open/close cycles keep heaps, listeners and nodes bounded', async ({ page, site, hub }) => {
    test.setTimeout(900_000);
    const probe = await cdpWithWorkers(page);
    await page.goto('/quickstart.html');
    const ui = dialogUI(page);
    let n = 0;
    const samples = {};
    for (let cycle = 1; cycle <= 50; cycle++) {
      await ui.trigger.click();
      await expect(ui.dialog).toBeVisible();
      for (let k = 0; k < 4; k++) {
        const q = `${WORDS[n % WORDS.length]} ${n}`;
        await typeQuery(ui.input, q);
        await site.waitProbe('embed', (p) => p.text.endsWith(q), 30_000);
        await stableSnapshot(ui.listbox, { quietMs: 300, allowEmpty: true });
        n++;
      }
      await clearQuery(ui.input);
      await ui.input.press('Escape');
      await expect(ui.dialog).toBeHidden();
      if (n === 20 || n === 112 || n === 200) {
        await sleep(500);
        samples[n] = await measure(probe);
      }
    }
    expect(n).toBe(200);
    const a = samples[20];
    const mid = samples[112];
    const b = samples[200];
    const all = { after20: a, after112: mid, after200: b };
    test.info().annotations.push({ type: 'measurements', description: JSON.stringify(all) });
    console.log('item4 measurements', JSON.stringify(all));
    expect.soft(b.pageHeap - a.pageHeap, 'page JS heap growth (bytes)').toBeLessThanOrEqual(HEAP_BOUND);
    expect.soft(b.workerHeap - a.workerHeap, 'worker JS heap growth (bytes)').toBeLessThanOrEqual(HEAP_BOUND);
    expect.soft(b.documentNodes, 'nodes in the page document').toBe(a.documentNodes);
    expect.soft(b.nodes - mid.nodes, 'DOM node counter growth over the second half').toBeLessThanOrEqual(NODE_BOUND);
    expect.soft(b.listeners - a.listeners, 'event listener growth').toBeLessThanOrEqual(LISTENER_BOUND);
  });

  test('closing the dialog removes the listeners that opening it added', async ({ page, site, hub }) => {
    const probe = await cdpWithWorkers(page);
    await page.goto('/quickstart.html');
    const ui = dialogUI(page);
    // Load the model once, so that first-use setup is not counted.
    await ui.trigger.click();
    await typeQuery(ui.input, 'fan beam');
    await site.waitProbe('embed', (p) => p.text.endsWith('fan beam'));
    await stableSnapshot(ui.listbox);
    await clearQuery(ui.input);
    await ui.input.press('Escape');
    await expect(ui.dialog).toBeHidden();

    const count = async () => ({
      window: (await probe.listeners('window')).length,
      document: (await probe.listeners('document')).length,
      dialog: (await probe.listeners('document.getElementById("pst-search-dialog")', -1)).length,
      total: (await probe.metrics()).JSEventListeners,
    });
    await probe.pageHeap(); // collect garbage
    const closed = await count();

    await ui.trigger.click();
    await expect(ui.dialog).toBeVisible();
    await typeQuery(ui.input, 'cone beam');
    await site.waitProbe('embed', (p) => p.text.endsWith('cone beam'));
    await stableSnapshot(ui.listbox);
    await clearQuery(ui.input);
    await ui.input.press('Escape');
    await expect(ui.dialog).toBeHidden();
    await probe.pageHeap();
    const closedAgain = await count();
    console.log('item4 listeners', JSON.stringify({ closed, closedAgain }));
    expect(closedAgain).toEqual(closed);
  });

  test('navigating away terminates the worker', async ({ page, site, hub }) => {
    const workers = trackWorkers(page);
    await page.goto('/quickstart.html');
    const ui = dialogUI(page);
    await ui.trigger.click();
    await typeQuery(ui.input, 'fan beam');
    await site.waitProbe('embed', (p) => p.text.endsWith('fan beam'));
    expect(workers.length).toBe(1);
    expect(workers[0].closed).toBe(false);
    await page.goto('/installation.html');
    await expect.poll(() => workers[0].closed, { timeout: 10_000 }).toBe(true);
  });
});

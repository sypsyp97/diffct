// Contract item 7 (security): query text and document snippets are rendered as text (no HTML
// injection), including queries like `<img src=x onerror=alert(1)>`.
import fs from 'node:fs';
import path from 'node:path';
import {
  test, expect, landingUI, dialogUI, searchPageUI, typeQuery, stableSnapshot, sleep,
} from './harness.mjs';

const PAYLOADS = [
  '<img src=x onerror=alert(1)>',
  '<img src=x onerror="window.__xss=1">',
  '"><svg onload="window.__xss=2">',
  '<b>bold</b> fan beam',
];

/** No injected element, no script effect, no alert. */
async function expectNoInjection(page, dialogs) {
  await sleep(500);
  expect(dialogs, 'alert() calls').toEqual([]);
  expect(await page.evaluate(() => window.__xss), 'window.__xss').toBeUndefined();
  const injected = await page.evaluate(() => ({
    img: document.querySelectorAll('img[src="x"]').length,
    svg: document.querySelectorAll('.srch-host svg[onload], [role="listbox"] svg').length,
    bold: [...document.querySelectorAll('.srch-host b, [role="listbox"] b')].map((b) => b.outerHTML),
    handlers: [...document.querySelectorAll('[onerror], [onload]')].map((e) => e.outerHTML),
  }));
  expect(injected).toEqual({ img: 0, svg: 0, bold: [], handlers: [] });
}

for (const where of ['landing', 'dialog', 'page']) {
  test(`item 7: query text renders as text (${where})`, async ({ page, site, hub }) => {
    const dialogs = [];
    page.on('dialog', (d) => { dialogs.push(d.message()); d.dismiss(); });
    let ui;
    if (where === 'landing') {
      await page.goto('/index.html');
      ui = landingUI(page);
    } else if (where === 'dialog') {
      await page.goto('/quickstart.html');
      ui = dialogUI(page);
      await ui.trigger.click();
    } else {
      await page.goto('/search.html');
      ui = searchPageUI(page);
    }
    for (const q of PAYLOADS) {
      await typeQuery(ui.input, q);
      await sleep(800);
      await stableSnapshot(ui.listbox, { allowEmpty: true });
      await expectNoInjection(page, dialogs);
      if (where === 'page') await ui.input.press('Enter'); // also run the page search for the query
    }
    await expectNoInjection(page, dialogs);
    // The no-match message shows the query as literal text.
    await typeQuery(ui.input, '<img src=x onerror=alert(1)> qqqzzzx');
    await sleep(800);
    const text = await ui.host.evaluate((h) => h.textContent);
    if (/qqqzzzx/.test(text)) expect(text).toContain('<img src=x onerror=alert(1)> qqqzzzx');
    await expectNoInjection(page, dialogs);
  });
}

test('item 7: query from the search page URL renders as text', async ({ page, site, hub }) => {
  const dialogs = [];
  page.on('dialog', (d) => { dialogs.push(d.message()); d.dismiss(); });
  for (const q of PAYLOADS) {
    await page.goto(`/search.html?q=${encodeURIComponent(q)}`);
    await sleep(1500);
    await expectNoInjection(page, dialogs);
    await expect(searchPageUI(page).input).toHaveValue(q);
  }
});

test('item 7: document titles, sections and snippets render as text', async ({ page, site, hub }) => {
  const dialogs = [];
  page.on('dialog', (d) => { dialogs.push(d.message()); d.dismiss(); });
  const file = path.join(site.root, '_static', 'search-chunks.json');
  const chunks = JSON.parse(fs.readFileSync(file, 'utf8'));
  const prose = chunks.findIndex((c) => c.t && c.o.length && c.s);
  const evil = '<img src=x onerror="window.__xss=3"><b>zqxjvorp</b>';
  const c = chunks[prose];
  const head = `Zqxjvorp ${evil} start. `;
  chunks[prose] = {
    ...c,
    p: `Title ${evil}`,
    s: `Section ${evil}`,
    t: head + c.t,
    o: c.o.map((x, i) => (i === 0 ? 0 : x + head.length)),
  };
  site.state.overlays.set('/_static/search-chunks.json', { body: JSON.stringify(chunks), type: 'application/json' });

  await page.goto('/index.html');
  const ui = landingUI(page);
  await typeQuery(ui.input, 'zqxjvorp');
  const snap = await stableSnapshot(ui.listbox);
  expect(snap.length).toBeGreaterThan(0);
  const shown = snap.join('\n');
  expect(shown, 'the snippet shows the markup as text').toContain('<img src=x onerror="window.__xss=3">');
  expect(shown, 'the title shows the markup as text').toContain('Title <img src=x');
  await expectNoInjection(page, dialogs);
});

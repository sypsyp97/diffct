// Contract item 1 (ordering): a result list shown for query Q is never replaced by results of an
// older query; a slow semantic answer for an old query is discarded when a newer query was typed.
// Keyword results for the current text are always shown.
import {
  test, expect, landingUI, dialogUI, typeQuery, snapshot, stableSnapshot, reference, sleep,
} from './harness.mjs';

test.describe('item 1: ordering', () => {
  test('slow semantic answer of an older query does not replace the newer query results', async ({ browser, page, site, hub }) => {
    const Q1 = 'cone beam geometry';
    const Q2 = 'parallel projector gradient';
    const keyword2 = await reference(browser, site, Q2, 'keyword');
    const semantic2 = await reference(browser, site, Q2, 'semantic');
    // The test can only see a mix-up when the two lists differ.
    expect(keyword2.length).toBeGreaterThan(0);
    expect(semantic2).not.toEqual(keyword2);

    await page.goto('/index.html');
    const ui = landingUI(page);
    await typeQuery(ui.input, 'warm up');
    await site.waitProbe('embed', (p) => p.text.endsWith('warm up'));
    await stableSnapshot(ui.listbox);

    const slow1 = site.gate('embed', (p) => p.text.endsWith(Q1));
    const slow2 = site.gate('embed', (p) => p.text.endsWith(Q2));
    await typeQuery(ui.input, Q1);
    await slow1.seen; // the semantic request for Q1 is in flight
    await typeQuery(ui.input, Q2);

    await test.step('keyword results for the current text show while semantic answers wait', async () => {
      await expect.poll(() => snapshot(ui.listbox), { timeout: 3000 }).toEqual(keyword2);
    });

    await test.step('the late answer for Q1 is discarded', async () => {
      slow1.release();
      await sleep(1500);
      expect(await ui.input.inputValue()).toBe(Q2);
      expect(await snapshot(ui.listbox)).toEqual(keyword2);
    });

    await test.step('the answer for Q2 merges in and stays', async () => {
      slow2.release();
      await expect.poll(() => snapshot(ui.listbox), { timeout: 15_000 }).toEqual(semantic2);
      await sleep(1000);
      expect(await snapshot(ui.listbox)).toEqual(semantic2);
    });
  });

  test('typing with uneven semantic latency never goes back to an older list (dialog)', async ({ browser, page, site, hub }) => {
    const parts = ['fan', ' beam', ' geometry'];
    const final = parts.join('');
    const keywordFinal = await reference(browser, site, final, 'keyword', 'dialog');
    const semanticFinal = await reference(browser, site, final, 'semantic', 'dialog');
    expect(semanticFinal).not.toEqual(keywordFinal);

    await page.goto('/quickstart.html');
    const ui = dialogUI(page);
    await ui.trigger.click();
    await expect(ui.dialog).toBeVisible();
    await typeQuery(ui.input, 'warm up');
    await site.waitProbe('embed', (p) => p.text.endsWith('warm up'));
    await stableSnapshot(ui.listbox);
    await ui.input.press('ControlOrMeta+a');
    await ui.input.press('Backspace');

    // Older prefixes answer late; the full query answers fast.
    site.state.probeDelay = (e) => {
      if (e.kind !== 'embed') return 0;
      if (e.params.text.endsWith(final)) return 50;
      if (e.params.text.endsWith('fan beam')) return 900;
      if (e.params.text.endsWith('fan')) return 1600;
      return 0;
    };

    // Record every list the visitor sees, with the input text at that time.
    const listId = await ui.input.getAttribute('aria-controls');
    await page.evaluate((id) => {
      const list = document.getElementById(id);
      const input = document.querySelector('#pst-search-dialog input[name="q"]');
      window.__snaps = [];
      const take = () => window.__snaps.push({
        text: input.value,
        items: [...list.querySelectorAll('[role="option"]')].filter((o) => !o.closest('[hidden]'))
          .map((o) => `${o.getAttribute('href')}\t${o.textContent.replace(/\s+/g, ' ').trim()}`),
      });
      new MutationObserver(take).observe(list, { childList: true, subtree: true, attributes: true, characterData: true });
    }, listId);

    for (const part of parts) {
      await ui.input.pressSequentially(part, { delay: 40 });
      await sleep(400); // longer than a typing pause, so each prefix asks for a semantic answer
    }
    await site.waitProbe('embed', (p) => p.text.endsWith(final));
    await sleep(2500); // every delayed answer has arrived by now
    const end = await stableSnapshot(ui.listbox, { quietMs: 1000 });
    expect(end).toEqual(semanticFinal);

    const snaps = await page.evaluate(() => window.__snaps);
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    const first = snaps.findIndex((s) => same(s.items, keywordFinal) || same(s.items, semanticFinal));
    expect(first, 'the final query results appeared').toBeGreaterThanOrEqual(0);
    const later = snaps.slice(first).filter((s) => s.items.length);
    const foreign = later.filter((s) => !same(s.items, keywordFinal) && !same(s.items, semanticFinal));
    expect(foreign.map((s) => s.items.slice(0, 2)), 'lists shown after the final query results appeared').toEqual([]);
  });
});

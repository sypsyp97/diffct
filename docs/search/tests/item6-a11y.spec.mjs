// Contract item 6 (accessibility): combobox/listbox roles, aria-activedescendant follows arrow
// keys, Esc behaviour, focus returns to the trigger after closing the dialog.
import {
  test, expect, landingUI, dialogUI, searchPageUI, typeQuery, stableSnapshot, sleep,
} from './harness.mjs';

async function open(page, where) {
  if (where === 'landing') {
    await page.goto('/index.html');
    return landingUI(page);
  }
  if (where === 'dialog') {
    await page.goto('/quickstart.html');
    const ui = dialogUI(page);
    await ui.trigger.click();
    await expect(ui.dialog).toBeVisible();
    return ui;
  }
  await page.goto('/search.html');
  return searchPageUI(page);
}

/** Ids of the visible options, in order. */
async function optionIds(ui) {
  const lb = await ui.listbox.locator();
  return lb.evaluate((el) => [...el.querySelectorAll('[role="option"]')].filter((o) => !o.closest('[hidden]')).map((o) => o.id));
}

async function expectActive(ui, id) {
  await expect(ui.input).toHaveAttribute('aria-activedescendant', id);
  const lb = await ui.listbox.locator();
  const option = lb.locator(`[id="${id}"]`);
  await expect(option).toHaveCount(1);
  await expect(option).toHaveAttribute('role', 'option');
  await expect(option).toHaveAttribute('aria-selected', 'true');
  await expect(lb.locator('[role="option"][aria-selected="true"]')).toHaveCount(1);
}

for (const where of ['landing', 'dialog', 'page']) {
  test.describe(`item 6: accessibility (${where})`, () => {
    test('combobox and listbox roles', async ({ page, site, hub }) => {
      const ui = await open(page, where);
      await expect(ui.input).toHaveAttribute('role', 'combobox');
      const lb = await ui.listbox.locator();
      await expect(lb).toHaveAttribute('role', 'listbox');
      await expect(ui.input).toHaveAttribute('aria-expanded', 'false');
      await typeQuery(ui.input, 'fan beam');
      await stableSnapshot(ui.listbox);
      await expect(ui.input).toHaveAttribute('aria-expanded', 'true');
      const options = lb.locator('[role="option"]');
      expect(await options.count()).toBeGreaterThan(0);
      // Every option has an id (needed for aria-activedescendant) and an accessible name.
      for (const id of await optionIds(ui)) {
        expect(id).toBeTruthy();
        const name = await lb.locator(`[id="${id}"]`).evaluate((o) => o.textContent.trim());
        expect(name.length).toBeGreaterThan(0);
      }
      // Options sit in the listbox directly or through presentational groups only.
      const owners = await lb.evaluate((el) => [...el.querySelectorAll('[role="option"]')].map((o) => {
        const roles = [];
        for (let p = o.parentElement; p && p !== el; p = p.parentElement) roles.push(p.getAttribute('role') || p.tagName.toLowerCase());
        return roles;
      }));
      for (const chain of owners) for (const r of chain) expect(['presentation', 'none', 'group']).toContain(r);
    });

    test('aria-activedescendant follows the arrow keys', async ({ page, site, hub }) => {
      const ui = await open(page, where);
      await typeQuery(ui.input, 'fan beam');
      await site.waitProbe('embed', (p) => p.text.endsWith('fan beam'));
      await stableSnapshot(ui.listbox);
      const ids = await optionIds(ui);
      expect(ids.length).toBeGreaterThanOrEqual(3);
      await ui.input.press('ArrowDown');
      await expectActive(ui, ids[0]);
      await ui.input.press('ArrowDown');
      await expectActive(ui, ids[1]);
      await ui.input.press('ArrowDown');
      await expectActive(ui, ids[2]);
      await ui.input.press('ArrowUp');
      await expectActive(ui, ids[1]);
      await expect(ui.input).toBeFocused();
    });
  });
}

test.describe('item 6: accessibility (dynamic results)', () => {
  test('aria-activedescendant never points at a removed option when results merge', async ({ page, site, hub }) => {
    const ui = await open(page, 'landing');
    await typeQuery(ui.input, 'warm up');
    await site.waitProbe('embed', (p) => p.text.endsWith('warm up'));
    const slow = site.gate('embed', (p) => p.text.endsWith('detector geometry'));
    await typeQuery(ui.input, 'detector geometry');
    await slow.seen;
    await stableSnapshot(ui.listbox);
    await ui.input.press('ArrowDown');
    await ui.input.press('ArrowDown');
    slow.release();
    await sleep(1500);
    const active = await ui.input.getAttribute('aria-activedescendant');
    if (active) {
      const lb = await ui.listbox.locator();
      await expect(lb.locator(`[id="${active}"][role="option"]`), `active ${active}`).toHaveCount(1);
      await expect(lb.locator(`[id="${active}"]`)).toHaveAttribute('aria-selected', 'true');
    }
  });

  test('aria-activedescendant never points at a removed option after a new query', async ({ page, site, hub }) => {
    const ui = await open(page, 'landing');
    await typeQuery(ui.input, 'fan beam');
    await stableSnapshot(ui.listbox);
    await ui.input.press('ArrowDown');
    await ui.input.press('ArrowDown');
    await ui.input.pressSequentially(' xyzzy', { delay: 5 });
    await stableSnapshot(ui.listbox, { allowEmpty: true });
    const active = await ui.input.getAttribute('aria-activedescendant');
    if (active) {
      const lb = await ui.listbox.locator();
      await expect(lb.locator(`[id="${active}"][role="option"]`), `aria-activedescendant="${active}"`).toHaveCount(1);
    }
  });

  test('Esc in the landing box clears the query and collapses the list', async ({ page, site, hub }) => {
    const ui = await open(page, 'landing');
    await typeQuery(ui.input, 'fan beam');
    await stableSnapshot(ui.listbox);
    await ui.input.press('ArrowDown');
    await ui.input.press('Escape');
    await expect(ui.input).toHaveValue('');
    await expect(ui.input).toHaveAttribute('aria-expanded', 'false');
    await expect(ui.input).toBeFocused();
    // aria-activedescendant may only name an option that exists.
    const active = await ui.input.getAttribute('aria-activedescendant');
    if (active) {
      const lb = await ui.listbox.locator();
      await expect(lb.locator(`[id="${active}"][role="option"]`), `aria-activedescendant="${active}"`).toHaveCount(1);
    }
  });

  test('Esc closes the dialog and focus returns to the search button', async ({ page, site, hub }) => {
    const ui = await open(page, 'dialog');
    await expect(ui.input).toBeFocused();
    await typeQuery(ui.input, 'fan beam');
    await stableSnapshot(ui.listbox);
    await ui.input.press('ArrowDown');
    await ui.input.press('Escape');
    await expect(ui.dialog).toBeHidden();
    await expect(ui.trigger).toBeFocused();
  });

  test('Esc closes the dialog when focus is on a result, and focus returns to the search button', async ({ page, site, hub }) => {
    const ui = await open(page, 'dialog');
    await typeQuery(ui.input, 'fan beam');
    await stableSnapshot(ui.listbox);
    const lb = await ui.listbox.locator();
    await lb.locator('[role="option"]').first().focus();
    await page.keyboard.press('Escape');
    await expect(ui.dialog).toBeHidden();
    await expect(ui.trigger).toBeFocused();
  });

  test('closing the dialog by a click outside returns focus to the search button', async ({ page, site, hub }) => {
    const ui = await open(page, 'dialog');
    await typeQuery(ui.input, 'fan beam');
    await stableSnapshot(ui.listbox);
    await page.mouse.click(5, 5);
    await expect(ui.dialog).toBeHidden();
    await expect(ui.trigger).toBeFocused();
  });

  test('Ctrl/Cmd+K opens the dialog, Esc closes it, focus returns to the element that had it', async ({ page, site, hub }) => {
    await page.goto('/quickstart.html');
    const ui = dialogUI(page);
    const link = page.locator('a.navbar-brand');
    await link.focus();
    await page.keyboard.press('ControlOrMeta+k');
    await expect(ui.dialog).toBeVisible();
    await expect(ui.input).toBeFocused();
    await typeQuery(ui.input, 'fan beam');
    await stableSnapshot(ui.listbox);
    await ui.input.press('Escape');
    await expect(ui.dialog).toBeHidden();
    await expect(link).toBeFocused();
  });
});

// Opt-in test with the real EmbeddingGemma 2 checkpoint from the Hugging Face Hub (about 350 MB).
// Run with FULL_MODEL=1. It checks contract item 2 end to end: semantic results arrive with the
// real model, and a reload reads the model from the browser cache with no second download.
// It uses a persistent browser profile on disk, as a visitor's browser has; the default
// Playwright context keeps storage in memory with a small quota.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from '@playwright/test';
import { test, expect, landingUI, typeQuery, FULL_MODEL } from './harness.mjs';

const HUB = /(^|\.)(huggingface\.co|hf\.co)$/;

test.skip(!FULL_MODEL, 'set FULL_MODEL=1 to download the real model');

test('real model: semantic results, then a reload with no second download', async ({ site }) => {
  test.setTimeout(30 * 60_000);
  site.state.wrapper = false; // the real runtime, unchanged
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'diffct-search-profile-'));
  const context = await chromium.launchPersistentContext(profile, { headless: true, baseURL: site.origin });
  try {
    const requests = [];
    context.on('requestfinished', async (request) => {
      const url = new URL(request.url());
      if (!HUB.test(url.hostname)) return;
      const headers = await request.allHeaders();
      const sizes = await request.sizes().catch(() => ({ responseBodySize: -1 }));
      requests.push({ url: `${url.origin}${url.pathname}`, range: headers.range || null, bytes: sizes.responseBodySize });
    });
    const downloads = (from = 0) => requests.slice(from).filter((r) => r.range !== 'bytes=0-0');
    const megabytes = (list) => list.reduce((s, r) => s + Math.max(r.bytes, 0), 0) / 1e6;

    const page = context.pages()[0] || await context.newPage();
    await page.goto('/index.html');
    let ui = landingUI(page);
    await typeQuery(ui.input, 'how do I use a curved detector');
    let lb = await ui.listbox.locator();
    await expect(lb.getByText(/^(Semantic|Both)$/).first()).toBeVisible({ timeout: 25 * 60_000 });
    const first = downloads();
    const mark = requests.length;

    await page.reload();
    ui = landingUI(page);
    await typeQuery(ui.input, '如何迭代重建?');
    lb = await ui.listbox.locator();
    await expect(lb.getByText(/^(Semantic|Both)$/).first()).toBeVisible({ timeout: 25 * 60_000 });
    const again = downloads(mark);

    const evidence = { first, firstMegabytes: megabytes(first), again, againMegabytes: megabytes(again) };
    fs.writeFileSync(test.info().outputPath('requests.json'), JSON.stringify(evidence, null, 1));
    console.log('full model:', JSON.stringify({ firstMegabytes: evidence.firstMegabytes, againMegabytes: evidence.againMegabytes,
      again: again.map((r) => `${r.bytes} ${r.url.slice(0, 120)}`) }, null, 1));
    expect(evidence.firstMegabytes, 'the first load downloads the model').toBeGreaterThan(50);
    expect(again.map((r) => `${r.bytes} ${r.url}`), 'model downloads after reload').toEqual([]);
  } finally {
    await context.close();
    fs.rmSync(profile, { recursive: true, force: true });
  }
});

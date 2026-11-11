// Fixture-only suite: no DOCS_HTML, model assets, or Sphinx build prerequisite.
import { fileURLToPath } from 'node:url';
import { defineConfig } from '@playwright/test';

const evidence = (name) => fileURLToPath(new URL('./test-results/runtime-cancel/' + name, import.meta.url));
export default defineConfig({
  testDir: '.',
  testMatch: 'runtime-cancel.spec.mjs',
  workers: 1,
  retries: 0,
  timeout: 15_000,
  expect: { timeout: 2_000 },
  outputDir: evidence(''),
  reporter: [['line'], ['json', { outputFile: evidence('results.json') }]],
  use: {
    browserName: 'chromium',
    headless: true,
    launchOptions: process.env.SEARCH_CHROMIUM ? { executablePath: process.env.SEARCH_CHROMIUM } : {},
  },
});

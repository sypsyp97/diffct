// Browser tests for the documentation search front end.
// Run from docs/search: DOCS_HTML=<built html dir> npx playwright test -c tests/playwright.config.mjs
import { fileURLToPath } from 'node:url';
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  testMatch: /.*\.spec\.mjs$/,
  timeout: 180_000,
  expect: { timeout: 15_000 },
  workers: Number(process.env.PW_WORKERS || 2),
  retries: 0,
  outputDir: fileURLToPath(new URL('test-results/', import.meta.url)),
  reporter: [['list'], ['json', { outputFile: fileURLToPath(new URL('test-results/results.json', import.meta.url)) }]],
  use: { ...devices['Desktop Chrome'], headless: true, trace: 'retain-on-failure' },
  projects: [{ name: 'chromium' }],
});

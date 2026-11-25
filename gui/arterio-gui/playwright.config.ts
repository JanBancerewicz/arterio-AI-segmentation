import {defineConfig, devices} from '@playwright/test';

export default defineConfig({
    testDir: './e2e',
    globalSetup: './e2e/globalSetup.ts',
    timeout: 60_000,
    expect: {timeout: 15_000},
    fullyParallel: true,
    workers: 2,
    reporter: [['list'], ['html', {open: 'never'}]],
    use: {
        baseURL: process.env.E2E_BASE_URL ?? 'http://localhost:8080',
        locale: 'pl-PL',
        screenshot: 'on',
        trace: 'retain-on-failure',
    },
    projects: [
        {name: 'chromium', use: {...devices['Desktop Chrome']}},
    ],
});

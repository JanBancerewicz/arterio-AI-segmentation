import '@testing-library/jest-dom/vitest';
import {cleanup} from '@testing-library/react';
import {afterAll, afterEach, beforeAll} from 'vitest';
import {server} from '../mocks/server.ts';

beforeAll(() => {
    server.listen({onUnhandledFrame: 'error'});
    const interceptedFetch = globalThis.fetch;
    globalThis.fetch = (input, init) =>
        interceptedFetch(typeof input === 'string' ? new URL(input, window.location.origin) : input, init);
});
afterEach(() => {
    cleanup();
    server.resetHandlers();
});
afterAll(() => server.close());

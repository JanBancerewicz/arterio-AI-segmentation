import {QueryClient} from '@tanstack/react-query';
import {describe, expect, it} from 'vitest';
import {ApiError} from './client.ts';
import {expireSession, isPasswordChangeRequired, isUnauthenticated, retryApiQuery} from './sessionCache.ts';

const error = (status: number, code: string) => new ApiError(status, code, 'test', null, null);

describe('session errors', () => {
    it('recognizes an expired session', () => {
        expect(isUnauthenticated(error(401, 'UNAUTHENTICATED'))).toBe(true);
        expect(isUnauthenticated(error(401, 'INVALID_CREDENTIALS'))).toBe(false);
        expect(isUnauthenticated(new Error('UNAUTHENTICATED'))).toBe(false);
    });

    it('recognizes a required password change', () => {
        expect(isPasswordChangeRequired(error(403, 'PASSWORD_CHANGE_REQUIRED'))).toBe(true);
        expect(isPasswordChangeRequired(error(403, 'FORBIDDEN_ORIGIN'))).toBe(false);
    });
});

describe('retryApiQuery', () => {
    it('does not retry client errors', () => {
        expect(retryApiQuery(0, error(400, 'VALIDATION_ERROR'))).toBe(false);
        expect(retryApiQuery(0, error(404, 'ANALYSIS_NOT_FOUND'))).toBe(false);
    });

    it('retries server and network errors twice', () => {
        expect(retryApiQuery(0, error(503, 'QUEUE_FULL'))).toBe(true);
        expect(retryApiQuery(1, new Error('network'))).toBe(true);
        expect(retryApiQuery(2, new Error('network'))).toBe(false);
    });
});

describe('expireSession', () => {
    it('removes private data and signs the user out', () => {
        const client = new QueryClient();
        client.setQueryData(['session'], {id: '1'});
        client.setQueryData(['analyses', 'list'], [{id: 'private'}]);
        client.setQueryData(['segmentation-models'], [{id: 'unet'}]);
        expireSession(client);
        expect(client.getQueryData(['session'])).toBeNull();
        expect(client.getQueryData(['analyses', 'list'])).toBeUndefined();
        expect(client.getQueryData(['segmentation-models'])).toBeUndefined();
    });
});

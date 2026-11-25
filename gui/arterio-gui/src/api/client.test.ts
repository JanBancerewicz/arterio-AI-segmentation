import {http, HttpResponse} from 'msw/http';
import {describe, expect, it} from 'vitest';
import {apiError, server} from '../mocks/server.ts';
import {ApiError, apiImageRequest, apiRequest} from './client.ts';

describe('apiRequest', () => {
    it('returns JSON and sends session cookies with the request', async () => {
        let credentials: RequestCredentials | undefined;
        let accept: string | null = null;
        server.use(http.get('/api/v1/models', ({request}) => {
            credentials = request.credentials;
            accept = request.headers.get('Accept');
            return HttpResponse.json([{id: 'unet'}]);
        }));
        expect(await apiRequest('/models')).toEqual([{id: 'unet'}]);
        expect(credentials).toBe('same-origin');
        expect(accept).toBe('application/json');
    });

    it('returns undefined for 204 No Content', async () => {
        server.use(http.post('/api/v1/auth/logout', () => new HttpResponse(null, {status: 204})));
        expect(await apiRequest('/auth/logout', {method: 'POST'})).toBeUndefined();
    });

    it('maps the error contract to ApiError', async () => {
        server.use(http.post('/api/v1/auth/login', () =>
            apiError(429, 'RATE_LIMITED', 'Zbyt wiele prób logowania.', {'Retry-After': '120'})));
        const error = await apiRequest<never>('/auth/login', {method: 'POST'}).catch((e: ApiError) => e);
        expect(error).toBeInstanceOf(ApiError);
        expect(error).toMatchObject({status: 429, code: 'RATE_LIMITED', message: 'Zbyt wiele prób logowania.', retryAfterSeconds: 120});
    });

    it('uses a generic error for responses outside the contract', async () => {
        server.use(http.get('/api/v1/models', () => new HttpResponse('<html>Bad gateway</html>', {status: 502})));
        const error = await apiRequest<never>('/models').catch((e: ApiError) => e);
        expect(error).toMatchObject({status: 502, code: 'HTTP_ERROR', details: null, retryAfterSeconds: null});
        expect(error.message).toContain('Spróbuj ponownie');
    });

    it('ignores an invalid Retry-After header', async () => {
        server.use(http.get('/api/v1/models', () => apiError(503, 'QUEUE_FULL', 'Kolejka pełna.', {'Retry-After': 'soon'})));
        const error = await apiRequest<never>('/models').catch((e: ApiError) => e);
        expect(error.retryAfterSeconds).toBeNull();
    });

    it('reports a network failure in user language', async () => {
        server.use(http.get('/api/v1/models', () => HttpResponse.error()));
        await expect(apiRequest('/models')).rejects.toThrow('Nie można połączyć się z serwerem');
    });

    it('rejects a successful response without JSON body', async () => {
        server.use(http.get('/api/v1/models', () => new HttpResponse('not json', {status: 200})));
        await expect(apiRequest('/models')).rejects.toThrow('nieprawidłową odpowiedź');
    });
});

describe('apiImageRequest', () => {
    it('returns a PNG blob', async () => {
        server.use(http.get('/api/v1/analyses/1/files/mask', () =>
            new HttpResponse(new Uint8Array([1, 2, 3]), {headers: {'Content-Type': 'image/png'}})));
        const blob = await apiImageRequest('/analyses/1/files/mask');
        expect(blob.size).toBe(3);
    });

    it('rejects content that is not PNG', async () => {
        server.use(http.get('/api/v1/analyses/1/files/mask', () =>
            new HttpResponse('<html></html>', {headers: {'Content-Type': 'text/html'}})));
        await expect(apiImageRequest('/analyses/1/files/mask')).rejects.toThrow('nie zwrócił obrazu PNG');
    });
});

import {http, HttpResponse} from 'msw/http';
import {setupServer} from 'msw/node';
import type {User} from '../api/auth.ts';

export const testUser: User = {
    id: '00000000-0000-0000-0000-000000000001',
    username: 'jan.kowalski',
    displayName: 'Jan Kowalski',
    mustChangePassword: false,
};

export const apiError = (status: number, code: string, message: string, headers?: Record<string, string>) =>
    HttpResponse.json({error: {code, message, details: null, requestId: 'test'}}, {status, headers});

export const server = setupServer(
    http.get('/api/v1/auth/me', () => HttpResponse.json(testUser)),
);

import {screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {http, HttpResponse} from 'msw/http';
import {describe, expect, it} from 'vitest';
import {apiError, server, testUser} from '../mocks/server.ts';
import {renderWithRouter} from '../test/renderWithRouter.tsx';
import {RequireSession} from './RequireSession.tsx';

const routes = [
    {
        element: <RequireSession />,
        children: [
            {path: '/', element: <h1>Panel analiz</h1>},
            {path: '/change-password', element: <h1>Zmiana hasła</h1>},
        ],
    },
    {path: '/login', element: <h1>Logowanie</h1>},
];

describe('RequireSession', () => {
    it('shows protected content to a signed-in user', async () => {
        renderWithRouter(routes, '/');
        expect(screen.getByRole('status')).toHaveTextContent('Sprawdzanie sesji');
        expect(await screen.findByRole('heading', {name: 'Panel analiz'})).toBeInTheDocument();
    });

    it('redirects an anonymous user to login', async () => {
        server.use(http.get('/api/v1/auth/me', () => apiError(401, 'UNAUTHENTICATED', 'Sesja wygasła.')));
        const {router} = renderWithRouter(routes, '/');
        expect(await screen.findByRole('heading', {name: 'Logowanie'})).toBeInTheDocument();
        expect(router.state.location.pathname).toBe('/login');
    });

    it('forces a temporary password change before anything else', async () => {
        server.use(http.get('/api/v1/auth/me', () => HttpResponse.json({...testUser, mustChangePassword: true})));
        const {router} = renderWithRouter(routes, '/');
        expect(await screen.findByRole('heading', {name: 'Zmiana hasła'})).toBeInTheDocument();
        expect(router.state.location.pathname).toBe('/change-password');
    });

    it('lets a signed-in user open the password page voluntarily', async () => {
        renderWithRouter(routes, '/change-password');
        expect(await screen.findByRole('heading', {name: 'Zmiana hasła'})).toBeInTheDocument();
    });

    it('offers a retry when the session cannot be checked', async () => {
        let attempts = 0;
        server.use(http.get('/api/v1/auth/me', () => {
            attempts++;
            return attempts <= 2 ? apiError(500, 'INTERNAL_ERROR', 'Błąd serwera.') : HttpResponse.json(testUser);
        }));
        renderWithRouter(routes, '/');
        expect(await screen.findByRole('alert', {}, {timeout: 3000})).toHaveTextContent('Nie udało się sprawdzić sesji');
        await userEvent.click(screen.getByRole('button', {name: 'Spróbuj ponownie'}));
        expect(await screen.findByRole('heading', {name: 'Panel analiz'})).toBeInTheDocument();
    });
});

import {screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {http, HttpResponse} from 'msw/http';
import {describe, expect, it} from 'vitest';
import {apiError, server, testUser} from '../mocks/server.ts';
import {renderWithRouter} from '../test/renderWithRouter.tsx';
import {LoginPage} from './LoginPage.tsx';

const routes = [
    {path: '/login', element: <LoginPage />},
    {path: '/', element: <h1>Panel analiz</h1>},
];

const fillAndSubmit = async (username: string, password: string) => {
    const user = userEvent.setup();
    await user.type(screen.getByLabelText('Identyfikator'), username);
    await user.type(screen.getByLabelText('Hasło'), password);
    await user.click(screen.getByRole('button', {name: 'Zaloguj się'}));
    return user;
};

describe('LoginPage', () => {
    it('signs in, stores the session and opens the application', async () => {
        let body: unknown;
        server.use(http.post('/api/v1/auth/login', async ({request}) => {
            body = await request.json();
            return HttpResponse.json(testUser);
        }));
        const {router, queryClient} = renderWithRouter(routes, '/login');
        await fillAndSubmit('  jan.kowalski  ', ' hasło ze spacjami ');
        expect(await screen.findByRole('heading', {name: 'Panel analiz'})).toBeInTheDocument();
        expect(body).toEqual({username: 'jan.kowalski', password: ' hasło ze spacjami '});
        expect(router.state.location.pathname).toBe('/');
        expect(queryClient.getQueryData(['session'])).toEqual(testUser);
    });

    it('shows the server message for invalid credentials', async () => {
        server.use(http.post('/api/v1/auth/login', () =>
            apiError(401, 'INVALID_CREDENTIALS', 'Nieprawidłowy identyfikator lub hasło.')));
        const {router} = renderWithRouter(routes, '/login');
        await fillAndSubmit('jan.kowalski', 'zle-haslo-123');
        expect(await screen.findByRole('alert')).toHaveTextContent('Nieprawidłowy identyfikator lub hasło.');
        expect(router.state.location.pathname).toBe('/login');
    });

    it('blocks the form while the request is pending', async () => {
        let release: () => void = () => {};
        server.use(http.post('/api/v1/auth/login', () => new Promise<Response>(resolve => {
            release = () => resolve(HttpResponse.json(testUser));
        })));
        renderWithRouter(routes, '/login');
        await fillAndSubmit('jan.kowalski', 'poprawne-haslo-1');
        expect(await screen.findByRole('button', {name: 'Logowanie…'})).toBeDisabled();
        expect(screen.getByLabelText('Identyfikator')).toBeDisabled();
        release();
        expect(await screen.findByRole('heading', {name: 'Panel analiz'})).toBeInTheDocument();
    });

    it('toggles password visibility', async () => {
        renderWithRouter(routes, '/login');
        const password = screen.getByLabelText('Hasło');
        expect(password).toHaveAttribute('type', 'password');
        await userEvent.click(screen.getByRole('button', {name: 'Pokaż hasło'}));
        expect(password).toHaveAttribute('type', 'text');
        await userEvent.click(screen.getByRole('button', {name: 'Ukryj hasło'}));
        expect(password).toHaveAttribute('type', 'password');
    });

    it('requires both fields before sending anything', async () => {
        let requests = 0;
        server.use(http.post('/api/v1/auth/login', () => {
            requests++;
            return HttpResponse.json(testUser);
        }));
        renderWithRouter(routes, '/login');
        await userEvent.click(screen.getByRole('button', {name: 'Zaloguj się'}));
        expect(screen.getByLabelText('Identyfikator')).toBeInvalid();
        expect(requests).toBe(0);
    });
});

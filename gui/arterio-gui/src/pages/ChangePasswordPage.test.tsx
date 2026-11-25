import {screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {http, HttpResponse} from 'msw/http';
import {describe, expect, it} from 'vitest';
import {apiError, server, testUser} from '../mocks/server.ts';
import {renderWithRouter} from '../test/renderWithRouter.tsx';
import {ChangePasswordPage} from './ChangePasswordPage.tsx';

const routes = [
    {path: '/change-password', element: <ChangePasswordPage />},
    {path: '/', element: <h1>Panel analiz</h1>},
];

const submit = async (current: string, next: string, confirmation = next) => {
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/Obecne hasło|Hasło tymczasowe/), current);
    await user.type(screen.getByLabelText('Nowe hasło'), next);
    await user.type(screen.getByLabelText('Powtórz nowe hasło'), confirmation);
    await user.click(screen.getByRole('button', {name: 'Zmień hasło'}));
};

const countRequests = () => {
    const counter = {value: 0};
    server.use(http.post('/api/v1/auth/password', () => {
        counter.value++;
        return HttpResponse.json(testUser);
    }));
    return counter;
};

describe('ChangePasswordPage', () => {
    it.each([
        ['krótkie', 'krótkie', 'od 12 do 128 znaków'],
        ['stare-haslo-123', 'stare-haslo-123', 'musi różnić się od obecnego'],
        ['stare-haslo-123', 'nowe-haslo-12345', 'nie są identyczne', 'inne-haslo-12345'],
    ])('validates locally before contacting the server (%s → %s)', async (current, next, message, confirmation = next) => {
        const requests = countRequests();
        renderWithRouter(routes, '/change-password');
        await screen.findByRole('heading', {name: 'Zmień hasło'});
        await submit(current, next, confirmation);
        expect(screen.getByRole('alert')).toHaveTextContent(message);
        expect(requests.value).toBe(0);
    });

    it('sends the change and returns to the application', async () => {
        let body: unknown;
        server.use(http.post('/api/v1/auth/password', async ({request}) => {
            body = await request.json();
            return HttpResponse.json(testUser);
        }));
        const {router} = renderWithRouter(routes, '/change-password');
        await screen.findByRole('heading', {name: 'Zmień hasło'});
        await submit('stare-haslo-123', 'nowe-haslo-12345');
        expect(await screen.findByRole('heading', {name: 'Panel analiz'})).toBeInTheDocument();
        expect(body).toEqual({currentPassword: 'stare-haslo-123', newPassword: 'nowe-haslo-12345'});
        expect(router.state.location.pathname).toBe('/');
    });

    it('shows a server rejection of the current password', async () => {
        server.use(http.post('/api/v1/auth/password', () =>
            apiError(403, 'INVALID_CURRENT_PASSWORD', 'Obecne hasło jest nieprawidłowe.')));
        renderWithRouter(routes, '/change-password');
        await screen.findByRole('heading', {name: 'Zmień hasło'});
        await submit('zle-haslo-1234', 'nowe-haslo-12345');
        expect(await screen.findByRole('alert')).toHaveTextContent('Obecne hasło jest nieprawidłowe.');
    });

    it('explains a forced change of a temporary password', async () => {
        server.use(http.get('/api/v1/auth/me', () => HttpResponse.json({...testUser, mustChangePassword: true})));
        renderWithRouter(routes, '/change-password');
        expect(await screen.findByRole('heading', {name: 'Ustaw własne hasło'})).toBeInTheDocument();
        expect(screen.getByLabelText('Hasło tymczasowe')).toBeInTheDocument();
        expect(screen.getByRole('button', {name: 'Wyloguj'})).toBeInTheDocument();
        expect(screen.queryByRole('link', {name: 'Anuluj'})).not.toBeInTheDocument();
    });

    it('shows all password fields on request', async () => {
        renderWithRouter(routes, '/change-password');
        await screen.findByRole('heading', {name: 'Zmień hasło'});
        await userEvent.click(screen.getByLabelText('Pokaż hasła'));
        for (const label of ['Obecne hasło', 'Nowe hasło', 'Powtórz nowe hasło']) {
            expect(screen.getByLabelText(label)).toHaveAttribute('type', 'text');
        }
    });
});

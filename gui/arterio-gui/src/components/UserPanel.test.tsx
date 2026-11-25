import {screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {http, HttpResponse} from 'msw/http';
import {describe, expect, it, vi} from 'vitest';
import {MaskEditContext} from '../context/MaskEditContext.ts';
import {apiError, server, testUser} from '../mocks/server.ts';
import {renderWithRouter} from '../test/renderWithRouter.tsx';
import {UserPanel} from './UserPanel.tsx';

const routes = (dirty = false) => [
    {path: '/', element: <MaskEditContext.Provider value={{dirty, setDirty: () => {}}}><UserPanel /></MaskEditContext.Provider>},
];

describe('UserPanel', () => {
    it('shows the signed-in user with initials', async () => {
        server.use(http.get('/api/v1/auth/me', () => HttpResponse.json({...testUser, displayName: ' Anna Maria Nowak '})));
        renderWithRouter(routes(), '/');
        expect(await screen.findByText('Anna Maria Nowak')).toBeInTheDocument();
        expect(screen.getByText('AM')).toBeInTheDocument();
        expect(screen.getByRole('link', {name: 'Zmień hasło'})).toHaveAttribute('href', '/change-password');
    });

    it('signs out and clears the session', async () => {
        server.use(http.post('/api/v1/auth/logout', () => new HttpResponse(null, {status: 204})));
        const {queryClient} = renderWithRouter(routes(), '/');
        await userEvent.click(await screen.findByRole('button', {name: 'Wyloguj'}));
        await vi.waitFor(() => expect(queryClient.getQueryData(['session'])).toBeNull());
    });

    it('reports a failed sign-out', async () => {
        server.use(http.post('/api/v1/auth/logout', () => apiError(403, 'FORBIDDEN_ORIGIN', 'Niedozwolone źródło.')));
        renderWithRouter(routes(), '/');
        await userEvent.click(await screen.findByRole('button', {name: 'Wyloguj'}));
        expect(await screen.findByRole('alert')).toHaveTextContent('Nie udało się wylogować.');
    });

    it('prevents signing out with unsaved mask edits', async () => {
        renderWithRouter(routes(true), '/');
        expect(await screen.findByRole('button', {name: 'Wyloguj'})).toBeDisabled();
        expect(screen.getByText('Zakończ edycję maski przed wylogowaniem.')).toBeInTheDocument();
    });
});

import {screen, within} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {http, HttpResponse} from 'msw/http';
import {describe, expect, it} from 'vitest';
import {apiError, server} from '../mocks/server.ts';
import {analysisFixture, pageFixture} from '../test/analysisFixtures.ts';
import {renderWithRouter} from '../test/renderWithRouter.tsx';
import {AnalysesList} from './AnalysesList.tsx';

const routes = [{path: '/', element: <AnalysesList />}];

describe('AnalysesList', () => {
    it('lists analyses with their status', async () => {
        server.use(http.get('/api/v1/analyses', () => HttpResponse.json(pageFixture([
            analysisFixture({id: 'a', filename: 'pierwsza.png', status: 'completed'}),
            analysisFixture({id: 'b', filename: 'druga.png', status: 'queued'}),
        ]))));
        renderWithRouter(routes, '/');
        const first = await screen.findByRole('link', {name: /pierwsza\.png/});
        expect(first).toHaveAttribute('href', '/analyses/a');
        expect(within(first).getByText('Zakończona')).toBeInTheDocument();
        expect(within(screen.getByRole('link', {name: /druga\.png/})).getByText('W kolejce')).toBeInTheDocument();
    });

    it('shows an empty history', async () => {
        server.use(http.get('/api/v1/analyses', () => HttpResponse.json(pageFixture([]))));
        renderWithRouter(routes, '/');
        expect(await screen.findByText('Brak analiz.')).toBeInTheDocument();
    });

    it('filters by status on the server', async () => {
        const statuses: (string | null)[] = [];
        server.use(http.get('/api/v1/analyses', ({request}) => {
            statuses.push(new URL(request.url).searchParams.get('status'));
            return HttpResponse.json(pageFixture([]));
        }));
        renderWithRouter(routes, '/');
        await screen.findByText('Brak analiz.');
        await userEvent.selectOptions(screen.getByLabelText('Filtruj po statusie'), 'W toku');
        expect(await screen.findByText('Brak analiz o tym statusie.')).toBeInTheDocument();
        expect(statuses).toEqual([null, 'queued,processing']);
    });

    it('loads further pages without duplicates', async () => {
        server.use(http.get('/api/v1/analyses', ({request}) => {
            const page = Number(new URL(request.url).searchParams.get('page'));
            const items = page === 1
                ? [analysisFixture({id: 'a', filename: 'a.png'}), analysisFixture({id: 'b', filename: 'b.png'})]
                : [analysisFixture({id: 'b', filename: 'b.png'}), analysisFixture({id: 'c', filename: 'c.png'})];
            return HttpResponse.json({...pageFixture(items, page, 40), pageSize: 20});
        }));
        renderWithRouter(routes, '/');
        await userEvent.click(await screen.findByRole('button', {name: 'Pokaż więcej'}));
        expect(await screen.findByRole('link', {name: /c\.png/})).toBeInTheDocument();
        expect(screen.getAllByRole('link')).toHaveLength(3);
        expect(screen.queryByRole('button', {name: 'Pokaż więcej'})).not.toBeInTheDocument();
    });

    it('reports a failure to load the history', async () => {
        server.use(http.get('/api/v1/analyses', () => apiError(400, 'VALIDATION_ERROR', 'Błąd.')));
        renderWithRouter(routes, '/');
        expect(await screen.findByRole('alert')).toHaveTextContent('Nie udało się pobrać historii.');
        expect(screen.getByRole('button', {name: 'Spróbuj ponownie'})).toBeInTheDocument();
    });
});

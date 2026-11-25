import {screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {http, HttpResponse} from 'msw/http';
import {beforeEach, describe, expect, it, vi} from 'vitest';
import type {Analysis} from '../api/Analysis.ts';
import {MaskEditContext} from '../context/MaskEditContext.ts';
import {apiError, server} from '../mocks/server.ts';
import {analysisFixture} from '../test/analysisFixtures.ts';
import {renderWithRouter} from '../test/renderWithRouter.tsx';
import {AnalysisPage} from './AnalysisPage.tsx';

const id = '11111111-1111-1111-1111-111111111111';

const routes = (dirty = false) => [
    {
        path: '/analyses/:id',
        element: <MaskEditContext.Provider value={{dirty, setDirty: () => {}}}><AnalysisPage /></MaskEditContext.Provider>,
    },
    {path: '/', element: <h1>Nowa analiza</h1>},
];

const serve = (analysis: Analysis) =>
    server.use(http.get(`/api/v1/analyses/${id}`, () => HttpResponse.json(analysis)));

beforeEach(() => {
    server.use(http.get(`/api/v1/analyses/${id}/files/:kind`, () =>
        new HttpResponse(new Uint8Array([1]), {headers: {'Content-Type': 'image/png'}})));
    URL.createObjectURL = vi.fn(() => 'blob:image');
    URL.revokeObjectURL = vi.fn();
});

describe('AnalysisPage', () => {
    it('shows a finished analysis with its metadata and results', async () => {
        serve(analysisFixture());
        renderWithRouter(routes(), `/analyses/${id}`);
        expect(await screen.findByText('Analiza została zakończona.')).toBeInTheDocument();
        expect(screen.getByText('Zakończona')).toBeInTheDocument();
        expect(screen.getByText('512 × 512 px')).toBeInTheDocument();
        expect(screen.getByText('1.23 s')).toBeInTheDocument();
        expect(screen.getByRole('button', {name: 'Nałożenie maski'})).toHaveAttribute('aria-pressed', 'true');
        await userEvent.click(screen.getByRole('button', {name: 'Maska'}));
        expect(screen.getByRole('button', {name: 'Maska'})).toHaveAttribute('aria-pressed', 'true');
        expect(screen.getByRole('button', {name: 'Edytuj maskę'})).toBeInTheDocument();
    });

    it('shows the queue position and lets the user cancel', async () => {
        serve(analysisFixture({status: 'queued', queuePosition: 3, finishedAt: null, processingMs: null}));
        server.use(http.post(`/api/v1/analyses/${id}/cancel`, () =>
            HttpResponse.json(analysisFixture({status: 'cancelled'}))));
        renderWithRouter(routes(), `/analyses/${id}`);
        expect(await screen.findByText('Pozycja w kolejce:')).toHaveTextContent('3');
        expect(screen.getByText('Maska i nałożenie będą dostępne po zakończeniu analizy.')).toBeInTheDocument();
        expect(screen.queryByRole('button', {name: 'Usuń analizę'})).not.toBeInTheDocument();
        serve(analysisFixture({status: 'cancelled'}));
        await userEvent.click(screen.getByRole('button', {name: 'Anuluj zadanie'}));
        expect(await screen.findByText('Analiza została anulowana i nie będzie przetwarzana.')).toBeInTheDocument();
    });

    it('names the current processing stage', async () => {
        serve(analysisFixture({status: 'processing', processingStage: 'inference', finishedAt: null}));
        renderWithRouter(routes(), `/analyses/${id}`);
        expect(await screen.findByText('Segmentacja naczyń')).toBeInTheDocument();
    });

    it('explains a failed analysis', async () => {
        serve(analysisFixture({status: 'failed', errorMessage: 'Zabrakło pamięci GPU.', files: {original: 'x', mask: null, overlay: null}}));
        renderWithRouter(routes(), `/analyses/${id}`);
        expect(await screen.findByText('Zabrakło pamięci GPU.')).toBeInTheDocument();
        expect(screen.getByText('Ta analiza nie ma wyniku segmentacji.')).toBeInTheDocument();
    });

    it('deletes only after confirmation', async () => {
        let deleted = false;
        serve(analysisFixture());
        server.use(http.delete(`/api/v1/analyses/${id}`, () => {
            deleted = true;
            return new HttpResponse(null, {status: 204});
        }));
        const {router} = renderWithRouter(routes(), `/analyses/${id}`);
        await userEvent.click(await screen.findByRole('button', {name: 'Usuń analizę'}));
        expect(screen.getByText(/Tej operacji nie można cofnąć/)).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', {name: 'Zachowaj analizę'}));
        expect(deleted).toBe(false);
        await userEvent.click(screen.getByRole('button', {name: 'Usuń analizę'}));
        await userEvent.click(screen.getByRole('button', {name: 'Usuń analizę i pliki'}));
        expect(await screen.findByRole('heading', {name: 'Nowa analiza'})).toBeInTheDocument();
        expect(deleted).toBe(true);
        expect(router.state.location.pathname).toBe('/');
    });

    it('blocks deletion while the mask has unsaved edits', async () => {
        serve(analysisFixture());
        renderWithRouter(routes(true), `/analyses/${id}`);
        expect(await screen.findByRole('button', {name: 'Usuń analizę'})).toBeDisabled();
        expect(screen.getByText('Zakończ edycję maski przed usunięciem analizy.')).toBeInTheDocument();
    });

    it('handles a missing or foreign analysis', async () => {
        server.use(http.get(`/api/v1/analyses/${id}`, () => apiError(404, 'ANALYSIS_NOT_FOUND', 'Nie znaleziono analizy.')));
        renderWithRouter(routes(), `/analyses/${id}`);
        expect(await screen.findByRole('heading', {name: 'Nie znaleziono analizy'})).toBeInTheDocument();
    });
});

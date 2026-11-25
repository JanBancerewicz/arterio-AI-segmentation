import {screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {http, HttpResponse} from 'msw/http';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {apiError, server} from '../mocks/server.ts';
import {analysisFixture, modelsFixture} from '../test/analysisFixtures.ts';
import {renderWithRouter} from '../test/renderWithRouter.tsx';
import {NewAnalysisPage} from './NewAnalysisPage.tsx';

const routes = [
    {path: '/', element: <NewAnalysisPage />},
    {path: '/analyses/:id', element: <h1>Szczegóły analizy</h1>},
];

const png = () => new File([new Uint8Array([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])], 'klatka.png', {type: 'image/png'});

const chooseFile = async (container: HTMLElement) =>
    userEvent.upload(container.querySelector('input[type="file"]') as HTMLInputElement, png());

beforeEach(() => {
    server.use(http.get('/api/v1/models', () => HttpResponse.json(modelsFixture)));
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue({width: 512, height: 512, close: vi.fn()}));
    URL.createObjectURL = vi.fn(() => 'blob:preview');
    URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
    vi.unstubAllGlobals();
});

describe('NewAnalysisPage', () => {
    it('preselects the default model and disables unavailable ones', async () => {
        renderWithRouter(routes, '/');
        expect(await screen.findByRole('radio', {name: /U-Net/})).toBeChecked();
        expect(screen.getByRole('radio', {name: /Qwen3-VL/})).toBeDisabled();
        expect(screen.getByText('domyślny')).toBeInTheDocument();
    });

    it('starts only after a valid file is chosen', async () => {
        const {container} = renderWithRouter(routes, '/');
        const submit = await screen.findByRole('button', {name: 'Rozpocznij analizę'});
        expect(submit).toBeDisabled();
        await chooseFile(container);
        expect(submit).toBeEnabled();
    });

    it('sends the file with the chosen model and opens the new analysis', async () => {
        let sent: FormData | undefined;
        server.use(http.post('/api/v1/analyses', async ({request}) => {
            sent = await request.formData();
            return HttpResponse.json(analysisFixture({id: 'abc', status: 'queued'}), {status: 201});
        }));
        const {container, router} = renderWithRouter(routes, '/');
        await userEvent.click(await screen.findByRole('radio', {name: /Test infrastruktury/}));
        await chooseFile(container);
        await userEvent.click(screen.getByRole('button', {name: 'Rozpocznij analizę'}));
        expect(await screen.findByRole('heading', {name: 'Szczegóły analizy'})).toBeInTheDocument();
        expect(router.state.location.pathname).toBe('/analyses/abc');
        expect(sent?.get('modelId')).toBe('smoke-test');
        expect((sent?.get('file') as File).size).toBe(8);
    });

    it('explains why the server refused the analysis', async () => {
        server.use(http.post('/api/v1/analyses', () =>
            apiError(429, 'TOO_MANY_ACTIVE_ANALYSES', 'Osiągnięto limit aktywnych analiz.')));
        const {container} = renderWithRouter(routes, '/');
        await screen.findByRole('radio', {name: /U-Net/});
        await chooseFile(container);
        await userEvent.click(screen.getByRole('button', {name: 'Rozpocznij analizę'}));
        expect(await screen.findByRole('alert')).toHaveTextContent('Osiągnięto limit aktywnych analiz.');
    });

    it('reports when models cannot be loaded', async () => {
        server.use(http.get('/api/v1/models', () => apiError(400, 'VALIDATION_ERROR', 'Błąd katalogu modeli.')));
        renderWithRouter(routes, '/');
        expect(await screen.findByRole('alert')).toHaveTextContent('Nie udało się pobrać listy modeli: Błąd katalogu modeli.');
        expect(screen.getByRole('button', {name: 'Rozpocznij analizę'})).toBeDisabled();
    });
});

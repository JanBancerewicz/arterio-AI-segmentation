import type {CreateAnalysisInput} from './CreateAnalysisInput.ts';
import type {Analysis, AnalysisPage} from './Analysis.ts';
import {ApiError, apiRequest} from './client.ts';

export class AnalysisNotFoundError extends Error {
    constructor() { super('Nie znaleziono analizy.'); }
}

export const fetchAnalyses = (page: number, status?: string, signal?: AbortSignal): Promise<AnalysisPage> => {
    const params = new URLSearchParams({page: String(page), pageSize: '20'});
    if (status) params.set('status', status);
    return apiRequest<AnalysisPage>(`/analyses?${params}`, {signal});
};

export const fetchAnalysis = async (id: string, signal?: AbortSignal): Promise<Analysis> => {
    try {
        return await apiRequest<Analysis>(`/analyses/${encodeURIComponent(id)}`, {signal});
    } catch (error) {
        if (error instanceof ApiError && error.status === 404) throw new AnalysisNotFoundError();
        throw error;
    }
};

export const cancelAnalysis = (id: string): Promise<Analysis> =>
    apiRequest<Analysis>(`/analyses/${encodeURIComponent(id)}/cancel`, {method: 'POST'});

export const createAnalysis = (input: CreateAnalysisInput): Promise<Analysis> => {
    const body = new FormData();
    body.append('file', input.file);
    body.append('modelId', input.modelId);
    return apiRequest<Analysis>('/analyses', {method: 'POST', body});
};

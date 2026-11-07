import type {CreateAnalysisInput} from './CreateAnalysisInput.ts';
import type {Analysis, AnalysisPage} from './Analysis.ts';
import {ApiError, apiImageRequest, apiRequest} from './client.ts';

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

export const deleteAnalysis = (id: string): Promise<void> =>
    apiRequest<void>(`/analyses/${encodeURIComponent(id)}`, {method: 'DELETE'});

export type AnalysisFileKind = 'original' | 'mask' | 'overlay';

export const fetchAnalysisFile = (id: string, kind: AnalysisFileKind, signal?: AbortSignal, download = false): Promise<Blob> =>
    apiImageRequest(`/analyses/${encodeURIComponent(id)}/files/${kind}${download ? '?download=true' : ''}`, {signal});

export const createAnalysis = (input: CreateAnalysisInput): Promise<Analysis> => {
    const body = new FormData();
    body.append('file', input.file);
    body.append('modelId', input.modelId);
    return apiRequest<Analysis>('/analyses', {method: 'POST', body});
};

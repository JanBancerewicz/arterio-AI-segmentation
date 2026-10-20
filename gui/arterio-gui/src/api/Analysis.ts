export type AnalysisStatus = 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';

export type Analysis = {
    id: string;
    filename: string;
    modelId: string;
    status: AnalysisStatus;
    createdAt: string;
    width: number;
    height: number;
    errorMessage: string | null;
}

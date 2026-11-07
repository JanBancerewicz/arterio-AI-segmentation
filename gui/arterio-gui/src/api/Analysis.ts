export type AnalysisStatus = 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';

export type Analysis = {
    id: string;
    filename: string;
    modelId: string;
    modelName: string;
    modelVersion: string;
    status: AnalysisStatus;
    createdAt: string;
    startedAt: string | null;
    finishedAt: string | null;
    processingMs: number | null;
    width: number;
    height: number;
    errorMessage: string | null;
    errorCode: string | null;
    queuePosition: number | null;
    processingStage: 'preprocessing' | 'inference' | 'postprocessing' | null;
    files: {original: string; mask: string | null; overlay: string | null};
};

export type AnalysisPage = {
    items: Analysis[];
    page: number;
    pageSize: number;
    total: number;
};

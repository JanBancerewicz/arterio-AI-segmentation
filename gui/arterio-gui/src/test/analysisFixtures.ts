import type {Analysis, AnalysisPage} from '../api/Analysis.ts';
import type {SegmentationModel} from '../api/SegmentationModel.ts';

export const analysisFixture = (overrides: Partial<Analysis> = {}): Analysis => {
    const id = overrides.id ?? '11111111-1111-1111-1111-111111111111';
    return {
        id,
        filename: 'klatka.png',
        modelId: 'unet-resnet34',
        modelName: 'U-Net (ResNet-34)',
        modelVersion: '1.0',
        status: 'completed',
        createdAt: '2026-11-20T10:00:00Z',
        startedAt: '2026-11-20T10:00:01Z',
        finishedAt: '2026-11-20T10:00:02Z',
        processingMs: 1234,
        width: 512,
        height: 512,
        errorMessage: null,
        errorCode: null,
        queuePosition: null,
        processingStage: null,
        files: {
            original: `/api/v1/analyses/${id}/files/original`,
            mask: `/api/v1/analyses/${id}/files/mask`,
            overlay: `/api/v1/analyses/${id}/files/overlay`,
        },
        ...overrides,
    };
};

export const pageFixture = (items: Analysis[], page = 1, total = items.length): AnalysisPage =>
    ({items, page, pageSize: 20, total});

export const modelsFixture: SegmentationModel[] = [
    {id: 'smoke-test', name: 'Test infrastruktury', version: '0.1.0', description: 'Progowanie', isDefault: false, available: true},
    {id: 'unet-resnet34', name: 'U-Net (ResNet-34)', version: '1.0', description: 'Segmentacja naczyń', isDefault: true, available: true},
    {id: 'qwen', name: 'Qwen3-VL', version: '0.9', description: 'Model eksperymentalny', isDefault: false, available: false},
];

import type {SegmentationModel} from './SegmentationModel.ts';
import {apiRequest} from './client.ts';

export const fetchModels = (signal?: AbortSignal): Promise<SegmentationModel[]> =>
    apiRequest<SegmentationModel[]>('/models', {signal});

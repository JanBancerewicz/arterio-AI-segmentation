import type {SegmentationModel} from "./SegmentationModel.ts";
import {segmentationModelsMock} from "../mocks/AnalysisModel.mock.ts";

export const fetchModels = async ():
    Promise<SegmentationModel[]> => {
        await new Promise(resolve => setTimeout(resolve, 400));
        return segmentationModelsMock;
}
import {useQuery} from "@tanstack/react-query";
import {fetchModels} from "../api/model.ts";

export const useSegmentationModels = () => {
    return useQuery({ queryKey: ['segmentation-models'], queryFn: fetchModels });
}
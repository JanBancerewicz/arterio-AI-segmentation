import {useMutation, useQueryClient} from "@tanstack/react-query";
import {createAnalysis} from "../api/analyses.ts";
import {analysisKeys} from "./analysisKeys.ts";

export const useCreateAnalysis = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: createAnalysis,
        onSuccess: (analysis) => {
            queryClient.setQueryData(analysisKeys.detail(analysis.id), analysis);
            return queryClient.invalidateQueries({ queryKey: analysisKeys.list() });
        },
    });
}

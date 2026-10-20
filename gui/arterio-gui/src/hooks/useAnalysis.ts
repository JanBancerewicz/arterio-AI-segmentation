import {useQuery} from "@tanstack/react-query";
import {AnalysisNotFoundError, fetchAnalysis} from "../api/analyses.ts";
import {analysisKeys} from "./analysisKeys.ts";

export const useAnalysis = (id: string) => {
    return useQuery({
        queryKey: analysisKeys.detail(id),
        queryFn: () => fetchAnalysis(id),
        enabled: Boolean(id),
        retry: (count, error) => !(error instanceof AnalysisNotFoundError) && count < 2,
        refetchInterval: query => {
            if (query.state.error instanceof AnalysisNotFoundError) return false;
            const status = query.state.data?.status;
            return status === 'queued' || status === 'processing' ? 1000 : false;
        },
    });
}

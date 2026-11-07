import {useQuery} from "@tanstack/react-query";
import {AnalysisNotFoundError, fetchAnalysis} from "../api/analyses.ts";
import {analysisKeys} from "./analysisKeys.ts";
import {retryApiQuery} from '../api/sessionCache.ts';

export const useAnalysis = (id: string) => {
    return useQuery({
        queryKey: analysisKeys.detail(id),
        queryFn: ({signal}) => fetchAnalysis(id, signal),
        enabled: Boolean(id),
        retry: (count, error) => !(error instanceof AnalysisNotFoundError) && retryApiQuery(count, error),
        refetchInterval: query => {
            if (query.state.error instanceof AnalysisNotFoundError) return false;
            const status = query.state.data?.status;
            return status === 'queued' || status === 'processing' ? 30_000 : false;
        },
    });
}

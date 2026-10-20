import {useQuery} from "@tanstack/react-query";
import {fetchAnalyses} from "../api/analyses.ts";
import {analysisKeys} from "./analysisKeys.ts";

export const useAnalyses = () => {
    return useQuery({
        queryKey: analysisKeys.list(),
        queryFn: fetchAnalyses,
        refetchInterval: query => query.state.data?.some(
            analysis => analysis.status === 'queued' || analysis.status === 'processing',
        ) ? 1000 : false,
    });
}

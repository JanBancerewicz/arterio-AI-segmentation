import {useQuery} from "@tanstack/react-query";
import {fetchAnalysis} from "../api/analyses.ts";
import {analysisKeys} from "./analysisKeys.ts";

export const useAnalysis = (id: string) => {
    return useQuery({ queryKey: analysisKeys.detail(id), queryFn: () => fetchAnalysis(id) });
}

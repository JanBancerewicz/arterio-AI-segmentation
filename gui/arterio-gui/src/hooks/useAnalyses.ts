import {useQuery} from "@tanstack/react-query";
import {fetchAnalyses} from "../api/analyses.ts";
import {analysisKeys} from "./analysisKeys.ts";

export const useAnalyses = () => {
    return useQuery({ queryKey: analysisKeys.list(), queryFn: fetchAnalyses });
}

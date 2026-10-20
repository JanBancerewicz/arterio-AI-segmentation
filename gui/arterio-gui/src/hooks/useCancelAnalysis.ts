import {useMutation, useQueryClient} from '@tanstack/react-query';
import {cancelAnalysis} from '../api/analyses.ts';
import {analysisKeys} from './analysisKeys.ts';

export const useCancelAnalysis = (id: string) => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: () => cancelAnalysis(id),
        onSuccess: analysis => {
            queryClient.setQueryData(analysisKeys.detail(id), analysis);
        },
        onSettled: () => queryClient.invalidateQueries({queryKey: analysisKeys.all}),
    });
};
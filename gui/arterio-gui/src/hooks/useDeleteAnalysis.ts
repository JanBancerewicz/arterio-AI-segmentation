import {useMutation, useQueryClient} from '@tanstack/react-query';
import {deleteAnalysis} from '../api/analyses.ts';
import {analysisKeys} from './analysisKeys.ts';

export const useDeleteAnalysis = (id: string) => {
    const client = useQueryClient();
    return useMutation({
        mutationFn: () => deleteAnalysis(id),
        onSuccess: async () => {
            await client.cancelQueries({queryKey: analysisKeys.all});
            client.removeQueries({queryKey: analysisKeys.detail(id)});
            void client.invalidateQueries({queryKey: analysisKeys.list()});
        },
        onError: () => {
            void client.invalidateQueries({queryKey: analysisKeys.all});
        },
    });
};

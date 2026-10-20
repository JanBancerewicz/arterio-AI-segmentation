import {useMutation, useQueryClient} from '@tanstack/react-query';
import {logout} from '../api/auth.ts';
import {analysisKeys} from './analysisKeys.ts';
import {sessionKey} from './useSession.ts';

export const useLogout = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: logout,

        onSuccess: async () => {
            await queryClient.cancelQueries();

            queryClient.removeQueries({
                queryKey: analysisKeys.all,
            });

            queryClient.setQueryData(sessionKey, null);
        },
    });
};
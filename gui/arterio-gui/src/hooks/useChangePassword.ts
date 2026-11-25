import {useMutation, useQueryClient} from '@tanstack/react-query';
import {changePassword} from '../api/auth.ts';
import {sessionKey} from './useSession.ts';

export const useChangePassword = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: changePassword,

        onSuccess: user => {
            queryClient.setQueryData(sessionKey, user);
        },
    });
};

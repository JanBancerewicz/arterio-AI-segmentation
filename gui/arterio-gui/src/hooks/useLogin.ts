import {useMutation, useQueryClient} from "@tanstack/react-query";
import {login} from "../api/auth.ts";
import {sessionKey} from "./useSession.ts";
import {analysisKeys} from './analysisKeys.ts';

export const useLogin = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: login,

        onSuccess: async user => {
            await queryClient.cancelQueries();
            queryClient.removeQueries({queryKey: analysisKeys.all});
            queryClient.setQueryData(sessionKey, user);
        },
    });
};

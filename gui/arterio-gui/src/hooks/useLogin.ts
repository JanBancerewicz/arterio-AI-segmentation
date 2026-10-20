import {useMutation, useQueryClient} from "@tanstack/react-query";
import {login} from "../api/auth.ts";
import {sessionKey} from "./useSession.ts";

export const useLogin = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: login,

        onSuccess: user => {
            queryClient.setQueryData(sessionKey, user);
        },
    });
};
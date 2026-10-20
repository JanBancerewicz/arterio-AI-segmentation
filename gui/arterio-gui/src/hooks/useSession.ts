import {useQuery} from '@tanstack/react-query';
import {getSession} from '../api/auth.ts';

export const sessionKey = ['session'] as const;

export const useSession = () => {
    return useQuery({
        queryKey: sessionKey,
        queryFn: getSession,
        retry: 1,
    });
};
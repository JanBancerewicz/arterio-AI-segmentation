import type {QueryClient} from '@tanstack/react-query';
import {ApiError} from './client.ts';

export const isUnauthenticated = (error: unknown) =>
    error instanceof ApiError && error.status === 401 && error.code === 'UNAUTHENTICATED';

export const isPasswordChangeRequired = (error: unknown) =>
    error instanceof ApiError && error.status === 403 && error.code === 'PASSWORD_CHANGE_REQUIRED';

export const expireSession = (client: QueryClient) => {
    // Abort requests synchronously before clearing private data.
    void client.cancelQueries();
    client.removeQueries({queryKey: ['analyses']});
    client.removeQueries({queryKey: ['segmentation-models']});
    client.setQueryData(['session'], null);
};

export const retryApiQuery = (count: number, error: unknown) =>
    !(error instanceof ApiError && error.status >= 400 && error.status < 500) && count < 2;

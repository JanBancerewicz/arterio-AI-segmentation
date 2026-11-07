import {ApiError, apiRequest} from './client.ts';

export type User = {
    id: string;
    username: string;
    displayName: string;
}

type LoginInput = {
    username: string;
    password: string;
}

export const login = async (input: LoginInput): Promise<User> => {
    return apiRequest<User>('/auth/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(input),
    });
}

export const logout = async (): Promise<void> => {
    return apiRequest<void>('/auth/logout', {method: 'POST'});
};

export const getSession = async (signal?: AbortSignal): Promise<User | null> => {
    try {
        return await apiRequest<User>('/auth/me', {signal});
    } catch (error) {
        if (error instanceof ApiError && error.status === 401 && error.code === 'UNAUTHENTICATED') return null;
        throw error;
    }
};

const API_PREFIX = '/api/v1';

export class ApiError extends Error {
    readonly status: number;
    readonly code: string;
    readonly details: Record<string, unknown> | null;
    readonly retryAfterSeconds: number | null;

    constructor(status: number, code: string, message: string,
                details: Record<string, unknown> | null, retryAfterSeconds: number | null) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.code = code;
        this.details = details;
        this.retryAfterSeconds = retryAfterSeconds;
    }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
    typeof value === 'object' && value !== null && !Array.isArray(value);

const request = async (path: string, options: RequestInit, accept: string): Promise<Response> => {
    const headers = new Headers(options.headers);
    headers.set('Accept', accept);

    let response: Response;
    try {
        response = await fetch(`${API_PREFIX}${path}`, {
            ...options,
            headers,
            credentials: 'same-origin',
            cache: 'no-store',
        });
    } catch (error) {
        if (options.signal?.aborted) throw error;
        throw new Error('Nie można połączyć się z serwerem. Sprawdź połączenie i spróbuj ponownie.', {cause: error});
    }

    if (!response.ok) {
        const payload: unknown = await response.json().catch(error => {
            if (options.signal?.aborted) throw error;
            return null;
        });
        const error = isRecord(payload) && isRecord(payload.error) ? payload.error : null;
        const retryHeader = response.headers.get('Retry-After');
        const retryAfter = retryHeader === null ? NaN : Number(retryHeader);
        throw new ApiError(
            response.status,
            typeof error?.code === 'string' ? error.code : 'HTTP_ERROR',
            typeof error?.message === 'string' ? error.message : 'Nie udało się wykonać żądania. Spróbuj ponownie.',
            isRecord(error?.details) ? error.details : null,
            Number.isFinite(retryAfter) && retryAfter >= 0 ? retryAfter : null,
        );
    }
    return response;
};

export const apiRequest = async <T>(path: string, options: RequestInit = {}): Promise<T> => {
    const response = await request(path, options, 'application/json');
    if (response.status === 204) return undefined as T;
    const payload: unknown = await response.json().catch(error => {
        if (options.signal?.aborted) throw error;
        return null;
    });
    if (payload === null) throw new Error('Serwer zwrócił nieprawidłową odpowiedź. Spróbuj ponownie.');
    return payload as T;
};

export const apiImageRequest = async (path: string, options: RequestInit = {}): Promise<Blob> => {
    const response = await request(path, options, 'image/png');
    if (response.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase() !== 'image/png') {
        throw new Error('Serwer nie zwrócił obrazu PNG. Spróbuj ponownie.');
    }
    return response.blob();
};

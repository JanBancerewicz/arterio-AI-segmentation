export const analysisKeys = {
    all: ['analyses'] as const,
    list: (status?: string) => status === undefined
        ? [...analysisKeys.all, 'list'] as const
        : [...analysisKeys.all, 'list', status] as const,
    detail: (id: string) => [...analysisKeys.all, 'detail', id] as const,
};

export const analysisKeys = {
    all: ['analyses'] as const,
    list: () => [...analysisKeys.all, 'list'] as const,
    detail: (id: string) => [...analysisKeys.all, 'detail', id] as const,
};

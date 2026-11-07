import {useInfiniteQuery} from '@tanstack/react-query';
import {fetchAnalyses} from '../api/analyses.ts';
import {analysisKeys} from './analysisKeys.ts';

export const useAnalyses = (status = '') => useInfiniteQuery({
    queryKey: analysisKeys.list(status),
    initialPageParam: 1,
    queryFn: ({pageParam, signal}) => fetchAnalyses(pageParam, status, signal),
    getNextPageParam: last => last.page * last.pageSize < last.total ? last.page + 1 : undefined,
    refetchInterval: 30_000,
});

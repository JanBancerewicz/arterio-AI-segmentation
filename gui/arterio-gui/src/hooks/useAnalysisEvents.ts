import {useEffect} from 'react';
import {useQueryClient} from '@tanstack/react-query';
import {analysisKeys} from './analysisKeys.ts';
import {getSession} from '../api/auth.ts';
import {expireSession} from '../api/sessionCache.ts';

export const useAnalysisEvents = () => {
    const client = useQueryClient();
    useEffect(() => {
        let disposed = false;
        let refreshTimer: ReturnType<typeof setTimeout> | undefined;
        let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
        let checkingSession = false;
        const controller = new AbortController();
        let source: EventSource;
        const refresh = () => {
            if (refreshTimer !== undefined) return;
            refreshTimer = setTimeout(() => {
                refreshTimer = undefined;
                if (!disposed) void client.invalidateQueries({queryKey: analysisKeys.all});
            }, 150);
        };
        const expire = () => {
            if (disposed) return;
            disposed = true;
            source.close();
            controller.abort();
            clearTimeout(refreshTimer);
            clearTimeout(reconnectTimer);
            expireSession(client);
        };
        const connect = () => {
            if (disposed) return;
            source = new EventSource('/api/v1/analyses/events');
            source.addEventListener('resync', refresh);
            source.addEventListener('analysis.changed', refresh);
            source.addEventListener('queue.changed', refresh);
            source.addEventListener('session.expired', expire);
            source.onerror = () => {
                if (disposed) return;
                // Native EventSource retries broken streams. HTTP errors may close it permanently.
                if (source.readyState === EventSource.CLOSED) {
                    clearTimeout(reconnectTimer);
                    reconnectTimer = setTimeout(connect, 5000);
                }
                if (checkingSession) return;
                checkingSession = true;
                void getSession(controller.signal).then(user => {
                    if (!user) expire();
                }).catch(() => {
                    // Network failure is not evidence of an expired session.
                }).finally(() => { checkingSession = false; });
            };
        };
        connect();
        return () => {
            disposed = true;
            source.close();
            controller.abort();
            clearTimeout(refreshTimer);
            clearTimeout(reconnectTimer);
        };
    }, [client]);
};

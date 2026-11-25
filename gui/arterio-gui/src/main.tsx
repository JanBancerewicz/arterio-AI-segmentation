import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import {router} from './App.tsx'
import {RouterProvider} from "react-router";
import {MutationCache, QueryCache, QueryClient, QueryClientProvider} from "@tanstack/react-query"
import {expireSession, isPasswordChangeRequired, isUnauthenticated, retryApiQuery} from './api/sessionCache.ts'

const onApiError = (error: unknown) => {
    if (isUnauthenticated(error)) expireSession(queryClient);
    if (isPasswordChangeRequired(error)) void queryClient.invalidateQueries({queryKey: ['session']});
};
const queryClient = new QueryClient({
    queryCache: new QueryCache({onError: onApiError}),
    mutationCache: new MutationCache({onError: onApiError}),
    defaultOptions: {queries: {retry: retryApiQuery}},
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
)

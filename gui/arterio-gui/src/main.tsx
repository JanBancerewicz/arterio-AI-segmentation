import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import {BrowserRouter} from "react-router";
import {MutationCache, QueryCache, QueryClient, QueryClientProvider} from "@tanstack/react-query"
import {expireSession, isUnauthenticated, retryApiQuery} from './api/sessionCache.ts'

const onApiError = (error: unknown) => {
    if (isUnauthenticated(error)) expireSession(queryClient);
};
const queryClient = new QueryClient({
    queryCache: new QueryCache({onError: onApiError}),
    mutationCache: new MutationCache({onError: onApiError}),
    defaultOptions: {queries: {retry: retryApiQuery}},
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
)

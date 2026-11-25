import {QueryClient, QueryClientProvider} from '@tanstack/react-query';
import {render} from '@testing-library/react';
import {createMemoryRouter, RouterProvider, type RouteObject} from 'react-router';

export const renderWithRouter = (routes: RouteObject[], initialPath: string) => {
    const queryClient = new QueryClient({
        defaultOptions: {queries: {retry: false}, mutations: {retry: false}},
    });
    const router = createMemoryRouter(routes, {initialEntries: [initialPath]});
    const view = render(
        <QueryClientProvider client={queryClient}>
            <RouterProvider router={router} />
        </QueryClientProvider>,
    );
    return {...view, router, queryClient};
};

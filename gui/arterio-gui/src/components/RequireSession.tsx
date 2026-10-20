import {Navigate, Outlet} from 'react-router';
import {useSession} from '../hooks/useSession.ts';

export const RequireSession = () => {
    const {
        data: user,
        isPending,
        isError,
        isFetching,
        refetch,
    } = useSession();

    if (isPending) {
        return <p role="status">Sprawdzanie sesji…</p>;
    }

    if (isError) {
        return (
            <div>
                <p role="alert">
                    Nie udało się sprawdzić sesji.
                </p>

                <button
                    disabled={isFetching}
                    onClick={() => void refetch()}
                >
                    Spróbuj ponownie
                </button>
            </div>
        );
    }

    if (!user) {
        return <Navigate to="/login" replace />;
    }

    return <Outlet />;
};
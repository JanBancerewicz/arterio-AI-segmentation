import {Link, useParams} from 'react-router';
import {useAnalysis} from '../hooks/useAnalysis.ts';
import {useCancelAnalysis} from '../hooks/useCancelAnalysis.ts';
import {AnalysisNotFoundError} from '../api/analyses.ts';
import {StatusBadge} from '../components/StatusBadge.tsx';
import {formatDate} from '../utils/formatDate.ts';
import styles from './AnalysisPage.module.css';

export const AnalysisPage = () => {
    const {id = ''} = useParams();
    // Keep mutation feedback scoped to the analysis when navigating through history.
    return <AnalysisDetails key={id} id={id} />;
};

const AnalysisDetails = ({id}: {id: string}) => {
    const {data: analysis, isPending, isError, error, refetch, isFetching} = useAnalysis(id);
    const cancellation = useCancelAnalysis(id);

    if (!id || error instanceof AnalysisNotFoundError) {
        return <section className={styles.page}>
            <h1>Nie znaleziono analizy</h1>
            <p>Ta analiza jest niedostępna lub została usunięta.</p>
            <Link className={styles.link} to="/">Przejdź do nowej analizy</Link>
        </section>;
    }

    if (isPending) {
        return <section className={styles.page} aria-busy="true">
            <h1>Szczegóły analizy</h1>
            <p role="status">Ładowanie analizy…</p>
        </section>;
    }

    if (!analysis) {
        return <section className={styles.page}>
            <h1>Nie udało się pobrać analizy</h1>
            <p role="alert">Sprawdź połączenie i spróbuj ponownie.</p>
            <button className={styles.button} disabled={isFetching} onClick={() => void refetch()}>
                {isFetching ? 'Ładowanie…' : 'Spróbuj ponownie'}
            </button>
            <Link className={styles.link} to="/">Przejdź do nowej analizy</Link>
        </section>;
    }

    const active = analysis.status === 'queued' || analysis.status === 'processing';

    return <section className={styles.page}>
        <header className={styles.header}>
            <div>
                <h1 className={styles.title}>Szczegóły analizy</h1>
                <p className={styles.filename}>{analysis.filename}</p>
            </div>
            <StatusBadge status={analysis.status} />
        </header>

        {isError && <div className={styles.error} role="alert">
            Nie udało się odświeżyć statusu. Wyświetlane dane mogą być nieaktualne.
            <button className={styles.button} disabled={isFetching} onClick={() => void refetch()}>Odśwież</button>
        </div>}

        <div className={styles.grid}>
            <section className={styles.card} aria-labelledby="analysis-status-heading">
                <h2 id="analysis-status-heading">Status analizy</h2>
                <div role="status" aria-live="polite" aria-atomic="true">
                    {analysis.status === 'queued' && <>
                        <p>Analiza oczekuje na rozpoczęcie przetwarzania.</p>
                        <p className={styles.detail}>Pozycja w kolejce: <strong>{analysis.queuePosition ?? 'Oczekiwanie na informację'}</strong></p>
                    </>}
                    {analysis.status === 'processing' && <>
                        <p>Trwa przetwarzanie obrazu.</p>
                        <p className={styles.detail}>Etap: <strong>{analysis.processingStage ?? 'Oczekiwanie na informację o etapie'}</strong></p>
                    </>}
                    {analysis.status === 'completed' && <>
                        <p>Analiza została zakończona.</p>
                        {/* Miejsce na osobny widok wyniku i maski. */}
                    </>}
                    {analysis.status === 'failed' && <>
                        <p>Nie udało się ukończyć analizy.</p>
                        <p className={styles.error}>{analysis.errorMessage || 'Nie podano przyczyny błędu przetwarzania.'}</p>
                    </>}
                    {analysis.status === 'cancelled' && <p>Analiza została anulowana i nie będzie przetwarzana.</p>}
                </div>
                {active && <p className={styles.hint}>Status aktualizuje się automatycznie. Możesz przejść do innej analizy i wrócić tutaj z historii.</p>}
                {analysis.status === 'queued' && <button
                    className={styles.button}
                    disabled={cancellation.isPending}
                    onClick={() => cancellation.mutate()}
                >{cancellation.isPending ? 'Anulowanie…' : 'Anuluj zadanie'}</button>}
                {cancellation.isError && <p className={styles.error} role="alert">{cancellation.error.message}</p>}
            </section>

            <section className={styles.card} aria-labelledby="analysis-info-heading">
                <h2 id="analysis-info-heading">Informacje o analizie</h2>
                <dl className={styles.metadata}>
                    <div><dt>Plik</dt><dd>{analysis.filename}</dd></div>
                    <div><dt>Rozdzielczość</dt><dd>{analysis.width} × {analysis.height} px</dd></div>
                    <div><dt>Identyfikator modelu</dt><dd>{analysis.modelId}</dd></div>
                    <div><dt>Utworzono</dt><dd>{formatDate(analysis.createdAt)}</dd></div>
                </dl>
            </section>
        </div>
        <Link className={styles.link} to="/">Przejdź do nowej analizy</Link>
    </section>;
};

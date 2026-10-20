import styles from './AnalysesList.module.css';
import {AnalysisListItem} from "./AnalysisListItem.tsx";
import {useState} from "react";
import {useAnalyses} from "../hooks/useAnalyses.ts";
import type {AnalysisStatus} from "../api/Analysis.ts";

type StatusFilter = 'all' | 'active' | 'completed' | 'failed';

const statusesByFilter: Record<StatusFilter, AnalysisStatus[] | null> = {
    all: null,
    active: ['queued', 'processing'],
    completed: ['completed'],
    failed: ['failed'],
};

export const AnalysesList = () => {
    const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
    const { data: analyses, isPending, isError } = useAnalyses();

    const allowedStatuses = statusesByFilter[statusFilter];
    const visibleAnalyses = allowedStatuses
        ? analyses?.filter(analysis => allowedStatuses.includes(analysis.status))
        : analyses;

    return (
        <section className={styles.history}>
            <div className={styles.header}>
                <h3 className={styles.heading}>Historia</h3>
                <select
                    className={styles.filter}
                    aria-label="Filtruj po statusie"
                    value={statusFilter}
                    onChange={e => setStatusFilter(e.target.value as StatusFilter)}
                >
                    <option value="all">Wszystkie</option>
                    <option value="active">W toku</option>
                    <option value="completed">Zakończone</option>
                    <option value="failed">Błędy</option>
                </select>
            </div>

            {isPending && <p className={styles.status}>Ładowanie historii…</p>}
            {isError && <p className={styles.status} role="alert">Nie udało się pobrać historii.</p>}
            {visibleAnalyses && visibleAnalyses.length === 0 && (
                <p className={styles.status}>
                    {statusFilter === 'all' ? 'Brak analiz.' : 'Brak analiz o tym statusie.'}
                </p>
            )}

            {visibleAnalyses && visibleAnalyses.length > 0 && (
                <ul className={styles.list}>
                    {visibleAnalyses.map(analysis => (
                        <li key={analysis.id}><AnalysisListItem analysis={analysis} /></li>
                    ))}
                </ul>
            )}
        </section>
    )
}

import styles from './StatusBadge.module.css';

export type AnalysisStatus = 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';

const labels: Record<AnalysisStatus, string> = {
    queued: 'W kolejce',
    processing: 'W trakcie',
    completed: 'Zakończona',
    failed: 'Błąd',
    cancelled: 'Anulowana',
};

type StatusBadgeProps = {
    status: AnalysisStatus;
}

export const StatusBadge = ({ status }: StatusBadgeProps) => {
    return <span className={`${styles.badge} ${styles[status]}`}>{labels[status]}</span>
}

import styles from './StatusBadge.module.css';
import type {AnalysisStatus} from "../api/Analysis.ts";

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

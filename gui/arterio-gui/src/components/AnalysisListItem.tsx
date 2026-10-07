import {NavLink} from "react-router";
import styles from './AnalysisListItem.module.css';
import {StatusBadge, type AnalysisStatus} from "./StatusBadge.tsx";

type AnalysisListItemProps = {
    id: string;
    filename: string;
    date: string;
    status: AnalysisStatus;
}

export const AnalysisListItem = ({ id, filename, date, status }: AnalysisListItemProps) => {
    return (
        <NavLink
            to={`/analyses/${id}`}
            className={({ isActive }) => isActive ? `${styles.item} ${styles.active}` : styles.item}
        >
            <span className={styles.filename} title={filename}>{filename}</span>
            <span className={styles.meta}>
                <span>{date}</span>
                <StatusBadge status={status} />
            </span>
        </NavLink>
    )
}

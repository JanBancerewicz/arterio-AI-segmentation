import {NavLink} from "react-router";
import styles from './AnalysisListItem.module.css';
import {StatusBadge} from "./StatusBadge.tsx";
import type {Analysis} from "../api/Analysis.ts";
import {formatDate} from "../utils/formatDate.ts";

type AnalysisListItemProps = {
    analysis: Analysis;
}

export const AnalysisListItem = ({ analysis }: AnalysisListItemProps) => {
    return (
        <NavLink
            to={`/analyses/${analysis.id}`}
            className={({ isActive }) => isActive ? `${styles.item} ${styles.active}` : styles.item}
        >
            <span className={styles.filename} title={analysis.filename}>{analysis.filename}</span>
            <span className={styles.meta}>
                <span>{formatDate(analysis.createdAt)}</span>
                <StatusBadge status={analysis.status} />
            </span>
        </NavLink>
    )
}

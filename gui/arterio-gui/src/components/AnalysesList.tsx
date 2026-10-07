import styles from './AnalysesList.module.css';
import {AnalysisListItem} from "./AnalysisListItem.tsx";

export const AnalysesList = () => {
    return (
        <section className={styles.history}>
            <div className={styles.header}>
                <h3 className={styles.heading}>Historia</h3>
                <select className={styles.filter} aria-label="Filtruj po statusie">
                    <option value="all">Wszystkie</option>
                    <option value="active">W toku</option>
                    <option value="completed">Zakończone</option>
                    <option value="failed">Błędy</option>
                </select>
            </div>

            <ul className={styles.list}>
                <li><AnalysisListItem id="4" filename="angio_04.png" date="06.10.2026 17:31" status="processing" /></li>
                <li><AnalysisListItem id="3" filename="angio_03.png" date="06.10.2026 17:11" status="completed" /></li>
                <li><AnalysisListItem id="2" filename="pacjent_lca_projekcja_rao.png" date="06.10.2026 12:28" status="failed" /></li>
                <li><AnalysisListItem id="1" filename="angio_01.png" date="05.10.2026 09:02" status="completed" /></li>
            </ul>
        </section>
    )
}

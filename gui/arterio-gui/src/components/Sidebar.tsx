import styles from './Sidebar.module.css';
import {Title} from "./Title.tsx";
import {NewAnalysisButton} from "./NewAnalysisButton.tsx";
import {AnalysesList} from "./AnalysesList.tsx";
import {UserPanel} from "./UserPanel.tsx";

export const Sidebar = () => {
    return (
        <div className={styles.sidebar}>
            <Title />
            <NewAnalysisButton />
            <AnalysesList />
            <UserPanel />
        </div>
    )
}

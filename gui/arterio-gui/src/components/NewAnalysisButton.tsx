import {NavLink} from "react-router";
import styles from './NewAnalysisButton.module.css';

export const NewAnalysisButton = () => {
    return (
        <NavLink
            to="/"
            end
            className={({ isActive }) => isActive ? `${styles.button} ${styles.active}` : styles.button}
        >
            + Nowa analiza
        </NavLink>
    )
}

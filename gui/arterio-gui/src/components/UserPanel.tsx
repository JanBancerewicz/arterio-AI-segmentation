import {Link} from "react-router";
import styles from './UserPanel.module.css';

export const UserPanel = () => {
    return (
        <div className={styles.panel}>
            <div className={styles.avatar}>JK</div>
            <div className={styles.info}>
                <span className={styles.name}>Jan Kowalski</span>
                <span className={styles.email} title="jan.kowalski@example.com">jan.kowalski@example.com</span>
            </div>
            <Link to="/login" className={styles.logout}>Wyloguj</Link>
        </div>
    )
}

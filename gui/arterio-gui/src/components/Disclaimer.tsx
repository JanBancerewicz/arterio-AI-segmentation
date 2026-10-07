import styles from './Disclaimer.module.css';

export const Disclaimer = () => {
    return (
        <div className={styles.disclaimer} role="note">
            <span className={styles.icon} aria-hidden="true">!</span>
            <p className={styles.text}>
                <strong>Narzędzie badawcze.</strong> Wyniki nie stanowią diagnozy medycznej
                i nie mogą zastąpić oceny lekarza.
            </p>
        </div>
    )
}

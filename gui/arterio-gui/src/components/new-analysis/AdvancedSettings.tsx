import styles from './AdvancedSettings.module.css';

export const AdvancedSettings = () => {
    return (
        <details className={styles.settings}>
            <summary className={styles.summary}>Parametry zaawansowane</summary>

            <div className={styles.content}>
                <label className={styles.field}>
                    <span className={styles.fieldHeader}>
                        <span className={styles.label}>Próg binaryzacji</span>
                        <output className={styles.value}>0.50</output>
                    </span>
                    <input
                        type="range"
                        min={0.05}
                        max={0.95}
                        step={0.05}
                        defaultValue={0.5}
                        className={styles.slider}
                    />
                    <span className={styles.help}>Niższy próg oznacza więcej pikseli uznanych za naczynie.</span>
                </label>

                <label className={styles.checkbox}>
                    <input type="checkbox" className={styles.checkboxInput} />
                    <span>
                        <span className={styles.label}>TTA (test-time augmentation)</span>
                        <span className={styles.help}>Dokładniejszy wynik kosztem dłuższego przetwarzania.</span>
                    </span>
                </label>
            </div>
        </details>
    )
}

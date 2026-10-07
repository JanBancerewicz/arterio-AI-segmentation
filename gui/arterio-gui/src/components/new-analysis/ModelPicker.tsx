import styles from './ModelPicker.module.css';

export const ModelPicker = () => {
    return (
        <fieldset className={styles.picker}>
            <legend className={styles.legend}>Model</legend>

            <div className={styles.options}>
                <label className={styles.card}>
                    <input type="radio" name="model" value="qwen-unet" defaultChecked className={styles.radio} />
                    <span className={styles.body}>
                        <span className={styles.header}>
                            <span className={styles.name}>Qwen3-VL + U-Net</span>
                            <span className={styles.version}>v1.0</span>
                            <span className={styles.tag}>domyślny</span>
                        </span>
                        <span className={styles.description}>Najdokładniejszy, dłuższy czas przetwarzania.</span>
                    </span>
                </label>

                <label className={styles.card}>
                    <input type="radio" name="model" value="unet" className={styles.radio} />
                    <span className={styles.body}>
                        <span className={styles.header}>
                            <span className={styles.name}>U-Net</span>
                            <span className={styles.version}>v1.0</span>
                        </span>
                        <span className={styles.description}>Lżejszy i szybszy, nieco mniej dokładny.</span>
                    </span>
                </label>

                <label className={styles.card}>
                    <input type="radio" name="model" value="gemma-unet" disabled className={styles.radio} />
                    <span className={styles.body}>
                        <span className={styles.header}>
                            <span className={styles.name}>Gemma + U-Net</span>
                            <span className={styles.version}>v0.1</span>
                        </span>
                        <span className={styles.description}>Chwilowo niedostępny.</span>
                    </span>
                </label>
            </div>
        </fieldset>
    )
}

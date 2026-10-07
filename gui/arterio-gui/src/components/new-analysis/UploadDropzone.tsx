import styles from './UploadDropzone.module.css';

export const UploadDropzone = () => {
    return (
        <label className={styles.dropzone}>
            <input type="file" accept="image/png" className={styles.input} />
            <span className={styles.icon} aria-hidden="true">⬆</span>
            <span className={styles.title}>Przeciągnij plik PNG tutaj</span>
            <span className={styles.subtitle}>
                lub <span className={styles.link}>kliknij, aby wybrać z dysku</span>
            </span>
            <span className={styles.requirements}>PNG · pojedyncza klatka</span>
        </label>
    )
}

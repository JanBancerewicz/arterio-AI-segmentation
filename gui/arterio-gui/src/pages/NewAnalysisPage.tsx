import styles from './NewAnalysisPage.module.css';
import {UploadDropzone} from "../components/new-analysis/UploadDropzone.tsx";
import {ModelPicker} from "../components/new-analysis/ModelPicker.tsx";
import {useState} from "react";

export const NewAnalysisPage = () => {

    const [uploadedFile, setUploadedFile] = useState<File | null>(null);

    return (
        <div className={styles.page}>
            <header className={styles.header}>
                <h1 className={styles.title}>Nowa analiza</h1>
                <p className={styles.subtitle}>
                    Prześlij klatkę angiografii w formacie PNG, wybierz model i uruchom segmentację naczyń.
                </p>
            </header>

            <form className={styles.form}>
                <UploadDropzone uploadedFile={uploadedFile} onFileChange={setUploadedFile} />

                <div className={styles.settings}>
                    <ModelPicker />

                    <button type="submit" className={styles.submit} disabled={!uploadedFile}>
                        Rozpocznij analizę
                    </button>
                    <p className={styles.hint}>
                        Analiza trafi do kolejki, a wynik pojawi się automatycznie.
                    </p>
                </div>
            </form>
        </div>
    )
}

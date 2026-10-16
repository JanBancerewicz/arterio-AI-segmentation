import styles from './NewAnalysisPage.module.css';
import {UploadDropzone} from "../components/new-analysis/UploadDropzone.tsx";
import {ModelPicker} from "../components/new-analysis/ModelPicker.tsx";
import {useState} from "react";
import {useSegmentationModels} from "../hooks/useSegmentationModels.ts";

export const NewAnalysisPage = () => {

    const [uploadedFile, setUploadedFile] = useState<File | null>(null);
    const [chosenModelId, setChosenModelId] = useState<string | null>(null);

    const { data: models, isPending, isError, error } = useSegmentationModels();

    const defaultModelId = models?.find(model => model.isDefault)?.id ?? null;
    const selectedModelId = chosenModelId ?? defaultModelId;

    const canSubmit = uploadedFile !== null && selectedModelId !== null;

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
                    {isPending && <p className={styles.modelsStatus}>Ładowanie modeli…</p> }
                    {isError && (
                        <p className={`${styles.modelsStatus} ${styles.modelsError}`} role="alert">
                            Nie udało się pobrać listy modeli: {error.message}
                        </p>
                    )}
                    {models && <ModelPicker segmentationModels={models} selectedModelId={selectedModelId} onSelect={setChosenModelId} />}

                    <button type="submit" className={styles.submit} disabled={!canSubmit}>
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

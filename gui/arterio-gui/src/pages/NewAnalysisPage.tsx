import styles from './NewAnalysisPage.module.css';
import {UploadDropzone} from "../components/new-analysis/UploadDropzone.tsx";
import {ModelPicker} from "../components/new-analysis/ModelPicker.tsx";
import {type FormEvent, useState} from "react";
import {useNavigate} from "react-router";
import {useSegmentationModels} from "../hooks/useSegmentationModels.ts";
import {useCreateAnalysis} from "../hooks/useCreateAnalysis.ts";

export const NewAnalysisPage = () => {

    const [uploadedFile, setUploadedFile] = useState<File | null>(null);
    const [chosenModelId, setChosenModelId] = useState<string | null>(null);

    const { data: models, isPending, isError, error } = useSegmentationModels();

    const defaultModelId = models?.find(model => model.isDefault)?.id ?? null;
    const selectedModelId = chosenModelId ?? defaultModelId;

    const createAnalysisMutation = useCreateAnalysis();
    const navigate = useNavigate();

    const canSubmit = uploadedFile !== null && selectedModelId !== null && !createAnalysisMutation.isPending;

    const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if (!uploadedFile || !selectedModelId) {
            return;
        }
        createAnalysisMutation.mutate(
            { file: uploadedFile, modelId: selectedModelId },
            { onSuccess: analysis => navigate(`/analyses/${analysis.id}`) },
        );
    }

    return (
        <div className={styles.page}>
            <header className={styles.header}>
                <h1 className={styles.title}>Nowa analiza</h1>
                <p className={styles.subtitle}>
                    Prześlij klatkę angiografii w formacie PNG, wybierz model i uruchom segmentację naczyń.
                </p>
            </header>

            <form className={styles.form} onSubmit={handleSubmit}>
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
                        {createAnalysisMutation.isPending ? 'Wysyłanie…' : 'Rozpocznij analizę'}
                    </button>
                    {createAnalysisMutation.isError && (
                        <p className={`${styles.modelsStatus} ${styles.modelsError}`} role="alert">
                            Nie udało się utworzyć analizy: {createAnalysisMutation.error.message}
                        </p>
                    )}
                    <p className={styles.hint}>
                        Analiza trafi do kolejki, a wynik pojawi się automatycznie.
                    </p>
                </div>
            </form>
        </div>
    )
}

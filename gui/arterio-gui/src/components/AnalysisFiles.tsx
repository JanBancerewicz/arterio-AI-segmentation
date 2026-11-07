import {useEffect, useRef, useState} from 'react';
import {useMutation, useQuery} from '@tanstack/react-query';
import type {Analysis} from '../api/Analysis.ts';
import {type AnalysisFileKind, fetchAnalysisFile} from '../api/analyses.ts';
import {analysisKeys} from '../hooks/analysisKeys.ts';
import styles from './AnalysisFiles.module.css';

const labels: Record<AnalysisFileKind, string> = {
    original: 'Oryginał', mask: 'Maska', overlay: 'Nałożenie maski',
};

export const AnalysisFiles = ({analysis}: {analysis: Analysis}) => {
    const [kind, setKind] = useState<AnalysisFileKind>('original');
    const available = (['original', 'mask', 'overlay'] as const)
        .filter(item => item === 'original' || (analysis.status === 'completed' && analysis.files[item]));
    const selected = available.includes(kind) ? kind : 'original';

    return <section className={styles.card} aria-labelledby="analysis-image-heading">
        <h2 id="analysis-image-heading">Obraz i wynik</h2>
        <div className={styles.controls} role="group" aria-label="Wybór obrazu">
            {available.map(item => <button key={item} type="button"
                className={styles.button} aria-pressed={selected === item}
                onClick={() => setKind(item)}>{labels[item]}</button>)}
        </div>
        {(analysis.status === 'queued' || analysis.status === 'processing') &&
            <p>Maska i nałożenie będą dostępne po zakończeniu analizy.</p>}
        {(analysis.status === 'failed' || analysis.status === 'cancelled') &&
            <p>Ta analiza nie ma wyniku segmentacji.</p>}
        <FilePreview key={selected} analysis={analysis} kind={selected} />
    </section>;
};

const FilePreview = ({analysis, kind}: {analysis: Analysis; kind: AnalysisFileKind}) => {
    const imageRef = useRef<HTMLImageElement>(null);
    const [decodeError, setDecodeError] = useState(false);
    const image = useQuery({
        queryKey: [...analysisKeys.detail(analysis.id), 'file', kind],
        queryFn: ({signal}) => fetchAnalysisFile(analysis.id, kind, signal),
        staleTime: Infinity,
        gcTime: 30_000,
    });
    useEffect(() => {
        if (!image.data || !imageRef.current) return;
        const element = imageRef.current;
        const url = URL.createObjectURL(image.data);
        element.src = url;
        return () => {
            element.removeAttribute('src');
            URL.revokeObjectURL(url);
        };
    }, [image.data]);

    const download = useMutation({
        mutationFn: () => fetchAnalysisFile(analysis.id, kind, undefined, true),
        onSuccess: blob => {
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${analysis.filename.replace(/\.png$/i, '')}-${kind}.png`;
            document.body.append(link);
            link.click();
            link.remove();
            setTimeout(() => URL.revokeObjectURL(url), 60_000);
        },
    });

    return <div>
        {image.isPending && <p role="status">Ładowanie obrazu…</p>}
        {image.isError && <div role="alert">
            <p>{image.error.message}</p>
            <button className={styles.button} disabled={image.isFetching}
                onClick={() => void image.refetch()}>Spróbuj ponownie</button>
        </div>}
        {decodeError && <p role="alert">Nie można wyświetlić obrazu. Możesz spróbować pobrać plik.</p>}
        <img ref={imageRef} className={styles.image} hidden={!image.data || decodeError}
            alt={`${labels[kind]} — ${analysis.filename}`} onError={() => setDecodeError(true)}
            onLoad={() => setDecodeError(false)} />
        <button className={styles.button} disabled={download.isPending}
            onClick={() => download.mutate()}>
            {download.isPending ? 'Pobieranie…' : `Pobierz PNG: ${labels[kind]}`}
        </button>
        {download.isError && <p role="alert">{download.error.message}</p>}
    </div>;
};

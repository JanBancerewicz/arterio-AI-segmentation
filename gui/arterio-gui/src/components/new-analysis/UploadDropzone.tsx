import styles from './UploadDropzone.module.css';
import {type DragEvent, useEffect, useRef, useState} from "react";
import {formatFileSize, MAX_FILE_SIZE_BYTES, validateFile} from "./validateFile.ts";

type UploadDropzoneProps = {
    uploadedFile: File | null;
    onFileChange: (file: File | null) => void;
}

type ImageSize = {
    width: number;
    height: number;
}

const FilePreview = ({file}: { file: File }) => {
    const imageRef = useRef<HTMLImageElement>(null);

    useEffect(() => {
        const url = URL.createObjectURL(file);
        if (imageRef.current) {
            imageRef.current.src = url;
        }
        return () => URL.revokeObjectURL(url);
    }, [file]);

    return <img ref={imageRef} alt={`Podgląd pliku ${file.name}`} className={styles.preview} />;
}

export const UploadDropzone = (props: UploadDropzoneProps) => {
    const [isDragOver, setIsDragOver] = useState(false);
    const [isValidating, setIsValidating] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [imageSize, setImageSize] = useState<ImageSize | null>(null);

    const reject = (message: string) => {
        setError(message);
        setImageSize(null);
        props.onFileChange(null);
    }

    const handleFiles = async (files: FileList | null) => {
        if (!files || files.length === 0) {
            return;
        }
        if (files.length > 1) {
            reject('Przeciągnij tylko jeden plik naraz.');
            return;
        }

        const file = files[0];
        setIsValidating(true);
        const result = await validateFile(file);
        setIsValidating(false);

        if (!result.ok) {
            reject(result.error);
            return;
        }

        setError(null);
        setImageSize({width: result.width, height: result.height});
        props.onFileChange(file);
    }

    const handleRemove = () => {
        setError(null);
        setImageSize(null);
        props.onFileChange(null);
    }

    const handleDragOver = (e: DragEvent<HTMLLabelElement>) => {
        e.preventDefault();
        setIsDragOver(true);
    }

    const handleDragLeave = (e: DragEvent<HTMLLabelElement>) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
            setIsDragOver(false);
        }
    }

    const handleDrop = (e: DragEvent<HTMLLabelElement>) => {
        e.preventDefault();
        setIsDragOver(false);
        void handleFiles(e.dataTransfer.files);
    }

    const dropzoneClassName = [
        styles.dropzone,
        props.uploadedFile && styles.selected,
        error && styles.invalid,
        isDragOver && styles.dragOver,
    ].filter(Boolean).join(' ');

    return (
        <div className={styles.wrapper}>
            <label
                className={dropzoneClassName}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <input type="file" accept="image/png,.png" className={styles.input} onChange={e => {
                    void handleFiles(e.target.files);
                    e.target.value = '';
                }}/>

                {props.uploadedFile ? (
                    <>
                        <FilePreview file={props.uploadedFile} />
                        <span className={styles.previewHint}>Kliknij lub przeciągnij inny plik, aby go zmienić</span>
                    </>
                ) : (
                    <>
                        <span className={styles.icon} aria-hidden="true">{isValidating ? '…' : '⬆'}</span>
                        <span className={styles.title}>
                            {isValidating ? 'Sprawdzanie pliku…' : 'Przeciągnij plik PNG tutaj'}
                        </span>
                        <span className={styles.subtitle}>
                            lub <span className={styles.link}>kliknij, aby wybrać z dysku</span>
                        </span>
                        <span className={styles.requirements}>
                            PNG · pojedyncza klatka · maks. {formatFileSize(MAX_FILE_SIZE_BYTES)}
                        </span>
                    </>
                )}
            </label>

            {error && (
                <p className={styles.errorMessage} role="alert">{error}</p>
            )}

            {props.uploadedFile && (
                <div className={styles.fileBar}>
                    <span className={styles.fileIcon} aria-hidden="true">✓</span>
                    <span className={styles.fileInfo}>
                        <span className={styles.fileName}>{props.uploadedFile.name}</span>
                        <span className={styles.fileMeta}>
                            {formatFileSize(props.uploadedFile.size)}
                            {imageSize && ` · ${imageSize.width} × ${imageSize.height} px`}
                        </span>
                    </span>
                    <button type="button" className={styles.removeButton} onClick={handleRemove}>
                        Usuń
                    </button>
                </div>
            )}
        </div>
    )
}

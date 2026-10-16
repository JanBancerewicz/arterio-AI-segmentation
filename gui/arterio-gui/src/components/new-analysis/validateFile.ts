export const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

const PNG_SIGNATURE = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

export type FileValidationResult =
    | { ok: true; width: number; height: number }
    | { ok: false; error: string };

export const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(0)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const hasPngSignature = async (file: File) => {
    const header = new Uint8Array(await file.slice(0, PNG_SIGNATURE.length).arrayBuffer());
    return PNG_SIGNATURE.every((byte, i) => header[i] === byte);
}

export const validateFile = async (file: File): Promise<FileValidationResult> => {
    const looksLikePng = file.type === 'image/png' || file.name.toLowerCase().endsWith('.png');
    if (!looksLikePng) {
        return { ok: false, error: `Plik „${file.name}” nie jest obrazem PNG. Wybierz klatkę zapisaną w formacie PNG.` };
    }

    if (file.size === 0) {
        return { ok: false, error: `Plik „${file.name}” jest pusty.` };
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
        return {
            ok: false,
            error: `Plik ma ${formatFileSize(file.size)}, maksymalnie ${formatFileSize(MAX_FILE_SIZE_BYTES)}.`,
        };
    }

    if (!(await hasPngSignature(file))) {
        return { ok: false, error: `Plik „${file.name}” ma rozszerzenie .png, ale jego zawartość nie jest obrazem PNG.` };
    }

    try {
        const bitmap = await createImageBitmap(file);
        const { width, height } = bitmap;
        bitmap.close();
        return { ok: true, width, height };
    } catch {
        return { ok: false, error: `Nie udało się odczytać obrazu „${file.name}”. Plik może być uszkodzony.` };
    }
}

import {fetchAnalysisFile} from '../../api/analyses.ts';

const decode = async (blob: Blob): Promise<HTMLImageElement> => {
    const url = URL.createObjectURL(blob);
    try {
        const image = new Image();
        image.src = url;
        await image.decode();
        return image;
    } finally { URL.revokeObjectURL(url); }
};

export const loadMaskSource = async (id: string, width: number, height: number, signal: AbortSignal) => {
    const blobs = await Promise.all([
        fetchAnalysisFile(id, 'original', signal), fetchAnalysisFile(id, 'mask', signal),
    ]);
    const [original, mask] = await Promise.all(blobs.map(decode));
    if ([original, mask].some(image => image.naturalWidth !== width || image.naturalHeight !== height)) {
        throw new Error('Rozmiar maski nie zgadza się z rozmiarem oryginalnego obrazu.');
    }
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext('2d', {willReadFrequently: true});
    if (!context) throw new Error('Przeglądarka nie obsługuje edycji obrazu.');
    context.drawImage(mask, 0, 0);
    const rgba = context.getImageData(0, 0, width, height).data;
    const pixels = new Uint8Array(width * height);
    for (let i = 0; i < pixels.length; i++) {
        const offset = i * 4, value = rgba[offset];
        if ((value !== 0 && value !== 255) || rgba[offset + 1] !== value
            || rgba[offset + 2] !== value || rgba[offset + 3] !== 255) {
            throw new Error('Edytor wymaga nieprzezroczystej maski binarnej: 0 dla tła, 255 dla naczyń.');
        }
        pixels[i] = value;
    }
    return {original, pixels, width, height};
};

export type MaskSource = Awaited<ReturnType<typeof loadMaskSource>>;

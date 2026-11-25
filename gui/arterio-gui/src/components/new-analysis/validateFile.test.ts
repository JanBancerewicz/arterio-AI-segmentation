import {afterEach, describe, expect, it, vi} from 'vitest';
import {formatFileSize, MAX_FILE_SIZE_BYTES, validateFile} from './validateFile.ts';

const PNG_SIGNATURE = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

const pngFile = (name = 'frame.png', extraBytes = 16) =>
    new File([new Uint8Array([...PNG_SIGNATURE, ...new Array(extraBytes).fill(0)])], name, {type: 'image/png'});

const decodedImage = (width: number, height: number) =>
    vi.fn().mockResolvedValue({width, height, close: vi.fn()});

afterEach(() => {
    vi.unstubAllGlobals();
});

describe('validateFile', () => {
    it('accepts a decodable PNG and reports its size', async () => {
        vi.stubGlobal('createImageBitmap', decodedImage(512, 768));
        expect(await validateFile(pngFile())).toEqual({ok: true, width: 512, height: 768});
    });

    it('accepts a PNG recognized only by its extension', async () => {
        vi.stubGlobal('createImageBitmap', decodedImage(256, 256));
        const file = new File([new Uint8Array(PNG_SIGNATURE)], 'FRAME.PNG', {type: ''});
        expect((await validateFile(file)).ok).toBe(true);
    });

    it('rejects other formats', async () => {
        const result = await validateFile(new File(['GIF89a'], 'frame.gif', {type: 'image/gif'}));
        expect(result).toEqual({ok: false, error: expect.stringContaining('nie jest obrazem PNG')});
    });

    it('rejects an empty file', async () => {
        const result = await validateFile(new File([], 'empty.png', {type: 'image/png'}));
        expect(result).toEqual({ok: false, error: expect.stringContaining('jest pusty')});
    });

    it('rejects a file over the size limit', async () => {
        const file = pngFile();
        Object.defineProperty(file, 'size', {value: MAX_FILE_SIZE_BYTES + 1});
        const result = await validateFile(file);
        expect(result).toEqual({ok: false, error: expect.stringContaining('maksymalnie 10.0 MB')});
    });

    it('rejects a renamed file without the PNG signature', async () => {
        const file = new File(['<?php echo 1; ?>'], 'shell.png', {type: 'image/png'});
        const result = await validateFile(file);
        expect(result).toEqual({ok: false, error: expect.stringContaining('zawartość nie jest obrazem PNG')});
    });

    it('rejects a corrupted PNG that cannot be decoded', async () => {
        vi.stubGlobal('createImageBitmap', vi.fn().mockRejectedValue(new Error('decode failed')));
        const result = await validateFile(pngFile());
        expect(result).toEqual({ok: false, error: expect.stringContaining('może być uszkodzony')});
    });
});

describe('formatFileSize', () => {
    it.each([
        [512, '1 KB'],
        [200 * 1024, '200 KB'],
        [1024 * 1024, '1.0 MB'],
        [MAX_FILE_SIZE_BYTES, '10.0 MB'],
    ])('formats %i bytes as %s', (bytes, expected) => {
        expect(formatFileSize(bytes)).toBe(expected);
    });
});

import {fireEvent, render, screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {UploadDropzone} from './UploadDropzone.tsx';

const png = (name = 'frame.png') =>
    new File([new Uint8Array([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0])], name, {type: 'image/png'});

const fileInput = (container: HTMLElement) => container.querySelector('input[type="file"]') as HTMLInputElement;

beforeEach(() => {
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue({width: 512, height: 512, close: vi.fn()}));
    URL.createObjectURL = vi.fn(() => 'blob:preview');
    URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
    vi.unstubAllGlobals();
});

describe('UploadDropzone', () => {
    it('accepts a valid PNG chosen from disk', async () => {
        const onFileChange = vi.fn();
        const {container} = render(<UploadDropzone uploadedFile={null} onFileChange={onFileChange} />);
        await userEvent.upload(fileInput(container), png());
        expect(onFileChange).toHaveBeenCalledWith(expect.objectContaining({name: 'frame.png'}));
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });

    it('rejects an invalid file and explains why', async () => {
        const onFileChange = vi.fn();
        const {container} = render(<UploadDropzone uploadedFile={null} onFileChange={onFileChange} />);
        await userEvent.upload(fileInput(container), new File(['<?php ?>'], 'shell.png', {type: 'image/png'}));
        expect(await screen.findByText(/zawartość nie jest obrazem PNG/)).toBeInTheDocument();
        expect(onFileChange).toHaveBeenCalledWith(null);
    });

    it('rejects dropping several files at once', async () => {
        const onFileChange = vi.fn();
        const {container} = render(<UploadDropzone uploadedFile={null} onFileChange={onFileChange} />);
        fireEvent.drop(container.querySelector('label')!, {dataTransfer: {files: [png('a.png'), png('b.png')]}});
        expect(await screen.findByText('Przeciągnij tylko jeden plik naraz.')).toBeInTheDocument();
        expect(onFileChange).toHaveBeenCalledWith(null);
    });

    it('accepts a dropped file', async () => {
        const onFileChange = vi.fn();
        const {container} = render(<UploadDropzone uploadedFile={null} onFileChange={onFileChange} />);
        fireEvent.drop(container.querySelector('label')!, {dataTransfer: {files: [png()]}});
        await vi.waitFor(() => expect(onFileChange).toHaveBeenCalledWith(expect.objectContaining({name: 'frame.png'})));
    });

    it('shows a preview of the selected file', () => {
        render(<UploadDropzone uploadedFile={png()} onFileChange={vi.fn()} />);
        expect(screen.getByAltText('Podgląd pliku frame.png')).toHaveAttribute('src', 'blob:preview');
    });
});

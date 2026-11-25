import {describe, expect, it} from 'vitest';
import {masksEqual, packMask, paintSegment, restoreMask} from './maskEditing.ts';

const blank = (width: number, height: number) => new Uint8Array(width * height);

const painted = (pixels: Uint8Array, width: number) => {
    const points: string[] = [];
    pixels.forEach((value, index) => {
        if (value) points.push(`${index % width},${Math.floor(index / width)}`);
    });
    return points;
};

describe('paintSegment', () => {
    it('paints a single pixel with a 1 px brush', () => {
        const pixels = blank(5, 5);
        const bounds = paintSegment(pixels, 5, 5, {x: 2, y: 2}, {x: 2, y: 2}, 1, 255);
        expect(painted(pixels, 5)).toEqual(['2,2']);
        expect(bounds).toEqual({left: 2, top: 2, right: 2, bottom: 2});
    });

    it('paints a filled circle for a single click', () => {
        const pixels = blank(11, 11);
        paintSegment(pixels, 11, 11, {x: 5, y: 5}, {x: 5, y: 5}, 5, 255);
        expect(pixels[5 * 11 + 5]).toBe(255);
        expect(pixels[5 * 11 + 7]).toBe(255);
        expect(pixels[5 * 11 + 8]).toBe(0);
        expect(pixels[3 * 11 + 3]).toBe(0);
        expect(painted(pixels, 11)).toHaveLength(21);
    });

    it('covers the whole segment without gaps', () => {
        const pixels = blank(20, 5);
        paintSegment(pixels, 20, 5, {x: 1, y: 2}, {x: 18, y: 2}, 1, 255);
        for (let x = 1; x <= 18; x++) expect(pixels[2 * 20 + x]).toBe(255);
        expect(painted(pixels, 20)).toHaveLength(18);
    });

    it('covers a diagonal stroke continuously', () => {
        const pixels = blank(10, 10);
        paintSegment(pixels, 10, 10, {x: 0, y: 0}, {x: 9, y: 9}, 1, 255);
        for (let i = 0; i < 10; i++) expect(pixels[i * 10 + i]).toBe(255);
    });

    it('clips the brush at image edges', () => {
        const pixels = blank(4, 4);
        const bounds = paintSegment(pixels, 4, 4, {x: 0, y: 0}, {x: 0, y: 0}, 9, 255);
        expect(bounds).toEqual({left: 0, top: 0, right: 3, bottom: 3});
        expect(pixels.length).toBe(16);
    });

    it('returns null when nothing changes', () => {
        const pixels = blank(5, 5);
        expect(paintSegment(pixels, 5, 5, {x: 2, y: 2}, {x: 2, y: 2}, 3, 0)).toBeNull();
        paintSegment(pixels, 5, 5, {x: 2, y: 2}, {x: 2, y: 2}, 3, 255);
        expect(paintSegment(pixels, 5, 5, {x: 2, y: 2}, {x: 2, y: 2}, 3, 255)).toBeNull();
    });

    it('returns null for a stroke outside the image', () => {
        const pixels = blank(5, 5);
        expect(paintSegment(pixels, 5, 5, {x: 50, y: 50}, {x: 60, y: 60}, 3, 255)).toBeNull();
        expect(painted(pixels, 5)).toEqual([]);
    });

    it('erases previously painted pixels', () => {
        const pixels = new Uint8Array(25).fill(255);
        paintSegment(pixels, 5, 5, {x: 2, y: 2}, {x: 2, y: 2}, 1, 0);
        expect(pixels[12]).toBe(0);
        expect(pixels.filter(value => value === 255)).toHaveLength(24);
    });

    it('keeps even brush sizes symmetric', () => {
        const pixels = blank(10, 10);
        paintSegment(pixels, 10, 10, {x: 4, y: 4}, {x: 4, y: 4}, 2, 255);
        expect(painted(pixels, 10)).toEqual(['4,4', '5,4', '4,5', '5,5']);
    });
});

describe('packMask and restoreMask', () => {
    it('round-trips any mask', () => {
        const pixels = new Uint8Array(37).map((_, index) => (index * 7) % 3 === 0 ? 255 : 0);
        const restored = new Uint8Array(pixels.length);
        restoreMask(packMask(pixels), restored);
        expect(masksEqual(restored, pixels)).toBe(true);
    });

    it('stores eight pixels per byte', () => {
        expect(packMask(new Uint8Array(512 * 512))).toHaveLength(512 * 512 / 8);
        expect(packMask(new Uint8Array(9))).toHaveLength(2);
    });

    it('uses least significant bits first', () => {
        const pixels = new Uint8Array(8);
        pixels[0] = 255;
        pixels[7] = 255;
        expect(Array.from(packMask(pixels))).toEqual([0b10000001]);
    });

    it('restores only binary values', () => {
        const restored = new Uint8Array(8).fill(17);
        restoreMask(new Uint8Array([0b00000101]), restored);
        expect(Array.from(restored)).toEqual([255, 0, 255, 0, 0, 0, 0, 0]);
    });
});

describe('masksEqual', () => {
    it('compares length and content', () => {
        expect(masksEqual(new Uint8Array([0, 255]), new Uint8Array([0, 255]))).toBe(true);
        expect(masksEqual(new Uint8Array([0, 255]), new Uint8Array([255, 0]))).toBe(false);
        expect(masksEqual(new Uint8Array([0]), new Uint8Array([0, 0]))).toBe(false);
    });
});

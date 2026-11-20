export type Point = {x: number; y: number};
export type Bounds = {left: number; top: number; right: number; bottom: number};

// Paint pixel centres within a circular brush swept along the entire segment.
export const paintSegment = (pixels: Uint8Array, width: number, height: number,
    from: Point, to: Point, diameter: number, value: 0 | 255): Bounds | null => {
    const radius = diameter / 2;
    const offset = diameter % 2 === 0 ? 0.5 : 0;
    from = {x: from.x + offset, y: from.y + offset};
    to = {x: to.x + offset, y: to.y + offset};
    const left = Math.max(0, Math.ceil(Math.min(from.x, to.x) - radius));
    const right = Math.min(width - 1, Math.floor(Math.max(from.x, to.x) + radius));
    const top = Math.max(0, Math.ceil(Math.min(from.y, to.y) - radius));
    const bottom = Math.min(height - 1, Math.floor(Math.max(from.y, to.y) + radius));
    const dx = to.x - from.x, dy = to.y - from.y;
    const lengthSquared = dx * dx + dy * dy;
    let changed = false;
    for (let y = top; y <= bottom; y++) {
        for (let x = left; x <= right; x++) {
            const t = lengthSquared ? Math.max(0, Math.min(1, ((x - from.x) * dx + (y - from.y) * dy) / lengthSquared)) : 0;
            if ((x - from.x - t * dx) ** 2 + (y - from.y - t * dy) ** 2 <= radius * radius) {
                const index = y * width + x;
                if (pixels[index] !== value) { pixels[index] = value; changed = true; }
            }
        }
    }
    return changed ? {left, top, right, bottom} : null;
};

export const packMask = (pixels: Uint8Array): Uint8Array => {
    const packed = new Uint8Array(Math.ceil(pixels.length / 8));
    for (let i = 0; i < pixels.length; i++) if (pixels[i]) packed[i >> 3] |= 1 << (i & 7);
    return packed;
};

export const restoreMask = (packed: Uint8Array, pixels: Uint8Array) => {
    for (let i = 0; i < pixels.length; i++) pixels[i] = packed[i >> 3] & (1 << (i & 7)) ? 255 : 0;
};

export const masksEqual = (a: Uint8Array, b: Uint8Array) =>
    a.length === b.length && a.every((value, index) => value === b[index]);

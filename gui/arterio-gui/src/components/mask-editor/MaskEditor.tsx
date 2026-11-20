import {useCallback, useContext, useEffect, useRef, useState, type PointerEvent} from 'react';
import {useQuery} from '@tanstack/react-query';
import {useBeforeUnload, useBlocker} from 'react-router';
import type {Analysis} from '../../api/Analysis.ts';
import {analysisKeys} from '../../hooks/analysisKeys.ts';
import {masksEqual, packMask, paintSegment, restoreMask, type Bounds, type Point} from '../../utils/maskEditing.ts';
import {loadMaskSource, type MaskSource} from './maskSource.ts';
import styles from './MaskEditor.module.css';
import {MaskEditContext} from '../../context/MaskEditContext.ts';

type Props = {analysis: Analysis; onClose: () => void};
type Tool = 'brush' | 'eraser' | 'pan';

export const MaskEditor = ({analysis, onClose}: Props) => {
    const source = useQuery({
        queryKey: [...analysisKeys.detail(analysis.id), 'editor-source'],
        queryFn: ({signal}) => loadMaskSource(analysis.id, analysis.width, analysis.height, signal),
        // The original prediction is immutable; SSE must not replace a local editing session.
        staleTime: 'static',
        gcTime: 0,
    });
    if (!source.data) return <div>
        <p role={source.isError ? 'alert' : 'status'}>
            {source.isError ? source.error.message : 'Przygotowanie maski do edycji…'}
        </p>
        {source.isError && <button onClick={() => void source.refetch()}>Spróbuj ponownie</button>}
        <button onClick={onClose}>Wróć do podglądu</button>
    </div>;
    return <EditorCanvas source={source.data} analysis={analysis} onClose={onClose} />;
};

const createWorking = (source: MaskSource) => ({
        pixels: source.pixels.slice(),
        initial: packMask(source.pixels),
        undo: [] as Uint8Array[],
        redo: [] as Uint8Array[],
        before: null as Uint8Array | null,
        pointer: null as {id: number; point: Point; pan: boolean; clientX: number; clientY: number} | null,
        layer: null as HTMLCanvasElement | null,
        rgba: null as ImageData | null,
});

const EditorCanvas = ({source, analysis, onClose}: Props & {source: MaskSource}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const viewportRef = useRef<HTMLDivElement>(null);
    const confirmationRef = useRef<HTMLDivElement>(null);
    const {setDirty} = useContext(MaskEditContext);
    useEffect(() => () => setDirty(false), [setDirty]);
    const workingRef = useRef<ReturnType<typeof createWorking> | null>(null);
    if (workingRef.current === null) workingRef.current = createWorking(source);
    const [tool, setTool] = useState<Tool>('brush');
    const [diameter, setDiameter] = useState(12);
    const [zoom, setZoom] = useState(1);
    const [fit, setFit] = useState(1);
    const [opacity, setOpacity] = useState(0.5);
    const [showMask, setShowMask] = useState(true);
    const [history, setHistory] = useState({undo: 0, redo: 0, dirty: false});
    const [exitRequested, setExitRequested] = useState(false);
    const [exportStatus, setExportStatus] = useState('');
    const [exporting, setExporting] = useState(false);
    const historyLimit = Math.max(1, Math.min(50, Math.floor(32 * 1024 * 1024 / Math.ceil(source.pixels.length / 8))));
    const blocker = useBlocker(history.dirty);
    useBeforeUnload(useCallback((event: BeforeUnloadEvent) => {
        if (history.dirty) { event.preventDefault(); event.returnValue = ''; }
    }, [history.dirty]));

    const draw = useCallback((bounds?: Bounds) => {
        const working = workingRef.current!;
        const canvas = canvasRef.current;
        const context = canvas?.getContext('2d');
        if (!canvas || !context) return;
        if (!working.layer) {
            working.layer = document.createElement('canvas');
            working.layer.width = source.width; working.layer.height = source.height;
            working.rgba = new ImageData(source.width, source.height);
        }
        const rgba = working.rgba!;
        const area = bounds ?? {left: 0, top: 0, right: source.width - 1, bottom: source.height - 1};
        for (let y = area.top; y <= area.bottom; y++) {
            for (let x = area.left; x <= area.right; x++) {
                const index = y * source.width + x, offset = index * 4;
                rgba.data[offset] = 255; rgba.data[offset + 1] = 74; rgba.data[offset + 2] = 88;
                rgba.data[offset + 3] = working.pixels[index];
            }
        }
        working.layer.getContext('2d')!.putImageData(rgba, 0, 0, area.left, area.top,
            area.right - area.left + 1, area.bottom - area.top + 1);
        context.clearRect(0, 0, source.width, source.height);
        context.drawImage(source.original, 0, 0);
        if (showMask) {
            context.globalAlpha = opacity;
            context.drawImage(working.layer, 0, 0);
            context.globalAlpha = 1;
        }
    }, [source, opacity, showMask]);
    useEffect(() => { draw(); }, [draw]);

    useEffect(() => {
        const viewport = viewportRef.current;
        if (!viewport) return;
        const observer = new ResizeObserver(() => {
            setFit(Math.min((viewport.clientWidth - 32) / source.width, (viewport.clientHeight - 32) / source.height));
        });
        observer.observe(viewport);
        return () => observer.disconnect();
    }, [source.width, source.height]);

    const updateHistory = () => {
        const working = workingRef.current!;
        const dirty = !masksEqual(packMask(working.pixels), working.initial);
        setHistory({
            undo: working.undo.length, redo: working.redo.length, dirty,
        });
        setDirty(dirty);
    };
    const commitStroke = () => {
        const working = workingRef.current!;
        if (!working.before) return;
        if (!masksEqual(working.before, packMask(working.pixels))) {
            working.undo.push(working.before);
            if (working.undo.length > historyLimit) working.undo.shift();
            working.redo = [];
        }
        working.before = null;
        updateHistory();
        setExportStatus('');
    };
    const restore = (direction: 'undo' | 'redo') => {
        const working = workingRef.current!;
        if (working.pointer) return;
        const stack = working[direction], snapshot = stack.pop();
        if (!snapshot) return;
        working[direction === 'undo' ? 'redo' : 'undo'].push(packMask(working.pixels));
        restoreMask(snapshot, working.pixels);
        draw(); updateHistory(); setExportStatus('');
    };
    const pointAt = (event: PointerEvent): Point => {
        const rect = canvasRef.current!.getBoundingClientRect();
        return {x: Math.floor((event.clientX - rect.left) * source.width / rect.width),
            y: Math.floor((event.clientY - rect.top) * source.height / rect.height)};
    };
    const paint = (from: Point, to: Point) => {
        const working = workingRef.current!;
        const bounds = paintSegment(working.pixels, source.width, source.height, from, to, diameter, tool === 'eraser' ? 0 : 255);
        if (bounds) draw(bounds);
    };
    const pointerDown = (event: PointerEvent<HTMLDivElement>) => {
        const working = workingRef.current!;
        if (working.pointer || (event.button !== 0 && event.button !== 1)) return;
        const pan = tool === 'pan' || event.button === 1;
        if (!pan && (event.target !== canvasRef.current || !showMask || opacity === 0)) return;
        event.preventDefault();
        event.currentTarget.focus({preventScroll: true});
        event.currentTarget.setPointerCapture(event.pointerId);
        const point = pointAt(event);
        working.pointer = {id: event.pointerId, point, pan, clientX: event.clientX, clientY: event.clientY};
        if (!pan) { working.before = packMask(working.pixels); paint(point, point); }
    };
    const pointerMove = (event: PointerEvent<HTMLDivElement>) => {
        const working = workingRef.current!;
        const pointer = working.pointer;
        if (!pointer || pointer.id !== event.pointerId) return;
        if (pointer.pan) {
            event.currentTarget.scrollLeft += pointer.clientX - event.clientX;
            event.currentTarget.scrollTop += pointer.clientY - event.clientY;
        } else {
            const next = pointAt(event);
            paint(pointer.point, next); pointer.point = next;
        }
        pointer.clientX = event.clientX; pointer.clientY = event.clientY;
    };
    const pointerEnd = (event: PointerEvent<HTMLDivElement>) => {
        const working = workingRef.current!;
        if (working.pointer?.id !== event.pointerId) return;
        working.pointer = null;
        commitStroke();
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    };
    const exportMask = () => {
        const working = workingRef.current!;
        const canvas = document.createElement('canvas');
        canvas.width = source.width; canvas.height = source.height;
        const context = canvas.getContext('2d')!;
        const rgba = context.createImageData(source.width, source.height);
        for (let i = 0; i < working.pixels.length; i++) {
            const value = working.pixels[i], offset = i * 4;
            rgba.data[offset] = rgba.data[offset + 1] = rgba.data[offset + 2] = value;
            rgba.data[offset + 3] = 255;
        }
        context.putImageData(rgba, 0, 0);
        setExporting(true);
        canvas.toBlob(blob => {
            setExporting(false);
            if (!blob) { setExportStatus('Nie udało się przygotować PNG. Spróbuj ponownie.'); return; }
            const url = URL.createObjectURL(blob), link = document.createElement('a');
            link.href = url; link.download = `${analysis.filename.replace(/\.png$/i, '')}-corrected-mask.png`;
            document.body.append(link); link.click(); link.remove();
            setTimeout(() => URL.revokeObjectURL(url), 60_000);
            setExportStatus('Przygotowano pobranie PNG. Poprawki nadal nie są zapisane na serwerze.');
        }, 'image/png');
    };
    const leave = () => {
        if (blocker.state === 'blocked') blocker.proceed();
        else onClose();
    };
    const confirmExit = exitRequested || blocker.state === 'blocked';
    useEffect(() => {
        if (confirmExit) {
            confirmationRef.current?.focus();
            confirmationRef.current?.scrollIntoView({block: 'nearest'});
        }
    }, [confirmExit]);

    return <div className={styles.editor}>
        <p className={styles.notice}>Edycja lokalna. Predykcja modelu pozostaje bez zmian.
            Poprawioną maskę możesz pobrać jako PNG; zapis na serwerze pojawi się w kolejnym kroku.</p>
        <div className={styles.toolbar} role="group" aria-label="Narzędzia edycji maski">
            {(['brush', 'eraser', 'pan'] as const).map(value => <button key={value} type="button"
                aria-pressed={tool === value} onClick={() => setTool(value)}>
                {{brush: 'Pędzel', eraser: 'Gumka', pan: 'Przesuwanie'}[value]}
            </button>)}
            <label>Rozmiar: {diameter} px
                <input aria-label="Rozmiar pędzla w pikselach obrazu" type="range" min="1" max="128"
                    value={diameter} onChange={event => setDiameter(Number(event.target.value))} />
            </label>
            <button disabled={!history.undo} onClick={() => restore('undo')}>Cofnij</button>
            <button disabled={!history.redo} onClick={() => restore('redo')}>Ponów</button>
        </div>
        <div className={styles.toolbar} role="group" aria-label="Widok maski">
            <label>Powiększenie: {Math.round(zoom * 100)}%
                <input aria-label="Powiększenie względem dopasowania" type="range" min="1" max="12" step="0.25"
                    value={zoom} onChange={event => setZoom(Number(event.target.value))} />
            </label>
            <button onClick={() => { setZoom(1); viewportRef.current?.scrollTo(0, 0); }}>Dopasuj obraz</button>
            <label>Przezroczystość maski: {Math.round((1 - opacity) * 100)}%
                <input aria-label="Przezroczystość maski" type="range" min="0" max="100"
                    value={Math.round((1 - opacity) * 100)} onChange={event => setOpacity(1 - Number(event.target.value) / 100)} />
            </label>
            <label className={styles.checkbox}><input type="checkbox" checked={showMask}
                onChange={event => setShowMask(event.target.checked)} />Pokaż maskę</label>
        </div>
        <p className={styles.hint}>Pędzel i gumka zmieniają piksele obrazu. Przesuwaj narzędziem „Przesuwanie”
            lub środkowym przyciskiem myszy. Cofanie obejmuje ostatnie {historyLimit} ruchów.</p>
        {(!showMask || opacity === 0) && <p role="status">Włącz widoczność maski, aby ją edytować.</p>}
        <div ref={viewportRef} className={styles.viewport} tabIndex={0} role="region"
            aria-label="Obszar edycji maski" style={{cursor: tool === 'pan' ? 'grab' : 'crosshair'}}
            onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerEnd}
            onPointerCancel={pointerEnd} onLostPointerCapture={pointerEnd}
            onKeyDown={event => {
                if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
                    event.preventDefault(); restore(event.shiftKey ? 'redo' : 'undo');
                }
            }}>
            <div className={styles.surface} style={{width: source.width * fit * zoom + 32, height: source.height * fit * zoom + 32}}>
                <canvas ref={canvasRef} width={source.width} height={source.height}
                    style={{width: source.width * fit * zoom, height: source.height * fit * zoom}}
                    aria-label="Obraz z edytowaną maską">Do edycji potrzebna jest obsługa canvas.</canvas>
            </div>
        </div>
        <div className={styles.toolbar}>
            <button disabled={exporting} onClick={exportMask}>{exporting ? 'Przygotowanie PNG…' : 'Pobierz poprawioną maskę PNG'}</button>
            <button onClick={() => history.dirty ? setExitRequested(true) : onClose()}>Zakończ edycję</button>
            <span role="status">{history.dirty ? 'Masz lokalne poprawki.' : 'Maska zgodna z predykcją.'}</span>
        </div>
        {exportStatus && <p role="status">{exportStatus}</p>}
        {confirmExit && <div ref={confirmationRef} tabIndex={-1} className={styles.confirmation} role="alertdialog" aria-modal="false"
            aria-labelledby="discard-mask-heading">
            <h3 id="discard-mask-heading">Odrzucić lokalne poprawki?</h3>
            <p>Poprawki nie są zapisane na serwerze. Możesz najpierw pobrać PNG.</p>
            <div className={styles.toolbar}>
                <button onClick={() => { setExitRequested(false); if (blocker.state === 'blocked') blocker.reset(); }}>Kontynuuj edycję</button>
                <button onClick={leave}>Odrzuć poprawki i wyjdź</button>
            </div>
        </div>}
    </div>;
};

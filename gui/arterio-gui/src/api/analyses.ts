import type {CreateAnalysisInput} from "./CreateAnalysisInput.ts";
import type {Analysis} from "./Analysis.ts";
import {analysesMock} from "../mocks/Analyses.mock.ts";

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const fetchAnalyses = async (): Promise<Analysis[]> => {
    await delay(400);
    return [...analysesMock].sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
}

export const fetchAnalysis = async (id: string): Promise<Analysis> => {
    await delay(300);
    const analysis = analysesMock.find(analysis => analysis.id === id);
    if (!analysis) {
        throw new Error('Nie znaleziono analizy.');
    }
    return {...analysis};
}

export const createAnalysis = async (input: CreateAnalysisInput): Promise<Analysis> => {
    await delay(600);
    console.log('Tworzenie analizy:', input.file.name, input.modelId);

    const bitmap = await createImageBitmap(input.file);
    const analysis: Analysis = {
        id: crypto.randomUUID(),
        filename: input.file.name,
        modelId: input.modelId,
        status: 'queued',
        createdAt: new Date().toISOString(),
        width: bitmap.width,
        height: bitmap.height,
        errorMessage: null,
    };
    bitmap.close();

    analysesMock.push(analysis);
    return analysis;
}

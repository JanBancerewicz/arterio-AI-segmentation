import type {SegmentationModel} from "../api/SegmentationModel.ts";

export const segmentationModelsMock: SegmentationModel[] = [
    {
        id: 'unet',
        name: 'U-Net (ResNet-34)',
        version: '1.0',
        description: 'Najlepszy wynik (Dice 0.795) i najszybszy. Działa również bez GPU.',
        isDefault: true,
        available: true,
    },
    {
        id: 'qwen-lora',
        name: 'Qwen3-VL + LoRA',
        version: '1.0',
        description: 'Model badawczy (Dice 0.758). Wymaga GPU, dłuższy czas przetwarzania.',
        isDefault: false,
        available: true,
    },
    {
        id: 'qwen-unet',
        name: 'Qwen3-VL → U-Net',
        version: '1.0',
        description: 'Eksperymentalna hybryda (Dice 0.739), słabsza od samego Qwena.',
        isDefault: false,
        available: true,
    },
    {
        id: 'gemma',
        name: 'Gemma 3 + LoRA',
        version: '0.1',
        description: 'W przygotowaniu, brak wytrenowanego modelu.',
        isDefault: false,
        available: false,
    },
];

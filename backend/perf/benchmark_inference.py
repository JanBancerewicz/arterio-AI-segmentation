"""Inference and password hashing cost: docker compose run --rm -v ./backend/perf:/app/perf worker python -m perf.benchmark_inference"""
import argparse
from pathlib import Path
import statistics
import tempfile
from time import perf_counter

import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch

from app.unet import UnetSegmenter
from app.users import password_hasher


def measure(function, repeats, warmup=2):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        samples.append((perf_counter() - started) * 1000)
    samples.sort()
    return {"mean": statistics.fmean(samples), "p50": samples[len(samples) // 2],
            "p95": samples[min(len(samples) - 1, round(len(samples) * 0.95))], "min": samples[0], "max": samples[-1]}


def row(name, result):
    print(f"| {name:<34} | {result['mean']:9.2f} | {result['p50']:9.2f} | {result['p95']:9.2f} | {result['min']:9.2f} | {result['max']:9.2f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        checkpoint = Path(folder) / "unet.pth"
        network = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
        torch.save({"state_dict": network.state_dict()}, checkpoint)
        segmenter = UnetSegmenter(checkpoint, device=args.device)
        print(f"Urządzenie: {args.device}, wątki CPU: {torch.get_num_threads()}, powtórzenia: {args.repeats}\n")
        print(f"| {'Operacja':<34} | {'średnia':>9} | {'mediana':>9} | {'p95':>9} | {'min':>9} | {'max':>9} |")
        print(f"|{'-' * 36}|{'-' * 11}|{'-' * 11}|{'-' * 11}|{'-' * 11}|{'-' * 11}|")
        tensor = torch.randn(1, 3, 512, 512)
        row("predykcja U-Net (wejście 512×512)", measure(lambda: segmenter.predict(tensor), args.repeats))
        for size in (512, 1024, 2048):
            image_path = Path(folder) / f"frame-{size}.png"
            Image.fromarray(np.random.default_rng(size).integers(0, 255, (size, size), dtype=np.uint8)).save(image_path)
            original_size = (size, size)
            mask = segmenter.predict(segmenter.preprocess(image_path)[0])
            row(f"przetwarzanie wstępne {size}×{size}", measure(lambda: segmenter.preprocess(image_path), args.repeats))
            row(f"przywrócenie maski {size}×{size}", measure(lambda: segmenter.restore_mask(mask, original_size), args.repeats))
            row(f"cały potok {size}×{size}", measure(
                lambda: segmenter.restore_mask(segmenter.predict(segmenter.preprocess(image_path)[0]), original_size),
                args.repeats))
        encoded = password_hasher.hash("haslo-testowe-wydajnosci")
        row("Argon2id: haszowanie hasła", measure(lambda: password_hasher.hash("haslo-testowe-wydajnosci"), args.repeats))
        row("Argon2id: weryfikacja hasła", measure(lambda: password_hasher.verify(encoded, "haslo-testowe-wydajnosci"), args.repeats))


if __name__ == "__main__":
    main()

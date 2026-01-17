# main.py
import os
import argparse
import torch
from transformers import AutoProcessor
from data_loader import load_multimodal_dataset

def _resolve_mid(arg_mid: str | None) -> str:
    # kolejność: argument → zmienna środowiskowa → sensowna domyślna (mniejszy model)
    return arg_mid or os.environ.get("MODEL_ID") or "Qwen/Qwen2-VL-2B-Instruct"

def _import_qwen_class(model_id: str):
    # dopasuj klasę do rodziny modelu
    if "Qwen3" in model_id:
        from transformers import Qwen3VLForConditionalGeneration as QwenVL
    else:
        from transformers import Qwen2VLForConditionalGeneration as QwenVL
    return QwenVL

def load_model(model_id: str, offload_dir: str):
    QwenVL = _import_qwen_class(model_id)
    kwargs = dict(
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        max_memory={"cuda:0": "3GiB", "cpu": "32GiB"},
        offload_folder=offload_dir,
    )
    model = QwenVL.from_pretrained(model_id, **kwargs)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default=None, help="HF repo ID lub lokalna ścieżka snapshotu")
    ap.add_argument("--offline", action="store_true", help="Nie łącz się z HF (użyj lokalnego cache)")
    ap.add_argument("--offload-dir", type=str, default="./offload", help="Folder offloadu wag na dysk")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--no-model", action="store_true", help="Nie ładuj modelu, tylko dane")
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    model_id = _resolve_mid(args.model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    # dane pod Qwen-VL
    dataset = load_multimodal_dataset(processor, args.max_seq_length)
    print("Dataset size:", len(dataset))

    model = None
    if not args.no_model:
        os.makedirs(args.offload_dir, exist_ok=True)
        model = load_model(model_id, args.offload_dir)
        # krótka informacja o mapowaniu urządzeń
        dm = getattr(model, "hf_device_map", None)
        if dm:
            print("Device map:", dm)

    # TODO: tu dopnij inference/trening gdy będziesz gotowy

if __name__ == "__main__":
    main()


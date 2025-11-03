import json
import torch
from matplotlib import pyplot as plt
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from data_loader import group_annotations,IMAGE_FOLDER, JSON_FILE


print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
tokenizer = processor.tokenizer 
max_seq_length = 2048



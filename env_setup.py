import torch
from unsloth import FastVisionModel

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    load_in_4bit=True,
)

print("✅ Qwen3-VL loaded correctly")
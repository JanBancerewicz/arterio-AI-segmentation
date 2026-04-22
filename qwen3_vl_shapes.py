"""
qwen3_vl_shapes.py
===================
Mini follow-up after qwen3_vl_diagnostic.py — only prints the shapes of
the tuple returned by Qwen3-VL vision tower on a 512x512 image.

Run:
    python qwen3_vl_shapes.py 2>&1 | tee logs/qwen3_shapes.log
"""

import numpy as np
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

print("Loading model (cached — should be fast)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Loaded.\n")

rng = np.random.default_rng(0)
img = Image.fromarray((rng.random((512, 512, 3)) * 255).astype(np.uint8))
proc = processor.image_processor(images=[img], return_tensors="pt")

device = next(model.parameters()).device
pv  = proc["pixel_values"].to(device=device, dtype=torch.bfloat16)
thw = proc["image_grid_thw"].to(device=device)

print(f"pixel_values   shape: {tuple(pv.shape)}  dtype: {pv.dtype}")
print(f"image_grid_thw shape: {tuple(thw.shape)}  value: {thw.tolist()}\n")

with torch.no_grad():
    out = model.visual(hidden_states=pv, grid_thw=thw)

print(f"Output type: {type(out).__name__}")

def describe(label, val, indent=0):
    pad = "  " * indent
    if val is None:
        print(f"{pad}{label}: None")
    elif isinstance(val, torch.Tensor):
        print(f"{pad}{label}: Tensor  shape={tuple(val.shape)}  dtype={val.dtype}")
    elif isinstance(val, (list, tuple)):
        print(f"{pad}{label}: {type(val).__name__} of length {len(val)}")
        for i, x in enumerate(val):
            describe(f"[{i}]", x, indent + 1)
    else:
        print(f"{pad}{label}: type={type(val).__name__}")

# Inspect as tuple
if isinstance(out, (tuple, list)):
    print(f"\n>>> Output is {type(out).__name__} of length {len(out)}:")
    for i, item in enumerate(out):
        describe(f"out[{i}]", item, indent=1)

# Also try named attrs in case some are exposed
print("\n>>> Named attributes on output (if any):")
for attr in ("last_hidden_state", "pooler_output", "deepstack_features",
             "hidden_states", "image_features"):
    if hasattr(out, attr):
        describe(attr, getattr(out, attr), indent=1)

# Try as dict
if hasattr(out, "keys"):
    try:
        keys = list(out.keys())
        if keys:
            print(f"\n>>> Dict-like keys: {keys}")
            for k in keys:
                describe(k, out[k], indent=1)
    except Exception:
        pass

# Also: call via the wrapper 'get_image_features' if it exists on the model
print("\n>>> Does model have get_image_features?",
      hasattr(model, "get_image_features"))
print(">>> Does model.model have get_image_features?",
      hasattr(getattr(model, "model", None), "get_image_features"))
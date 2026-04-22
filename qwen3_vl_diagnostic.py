"""
qwen3_vl_diagnostic.py
=======================
Diagnostic script for Qwen3-VL BEFORE rewriting train_qwen_seg.py.

Why this exists:
    The previous training run used Qwen2_5_VLForConditionalGeneration to load
    Qwen3-VL, which silently initialized the entire visual.blocks.*.mlp and
    visual.merger with RANDOM weights. Val Dice plateaued at ~0.42 because the
    SegDecoder was training on features from a broken encoder.

What this script verifies:
    1. Qwen3VLForConditionalGeneration loads WITHOUT any "newly initialized" warnings
    2. The real dimensions: vision_config.hidden_size, patch_size, spatial_merge_size
    3. The real Linear module names in the vision tower (qkv vs q_proj/k_proj/v_proj)
       — this decides LoRA target_modules
    4. What the vision tower output actually looks like:
       - last_hidden_state shape (after PatchMerger → LLM dim)
       - deepstack_features (list from multiple ViT layers, BEFORE merger)
    5. VRAM footprint after loading

Run:
    python qwen3_vl_diagnostic.py 2>&1 | tee logs/qwen3_diagnostic.log

Then paste the whole log back into the conversation.
"""

import sys
import traceback
from collections import Counter

import numpy as np
import torch
from PIL import Image

print("=" * 72)
print("Qwen3-VL diagnostic")
print("=" * 72)

# ── Environment ───────────────────────────────────────────────────────────
import transformers
print(f"\ntransformers version : {transformers.__version__}")
print(f"torch version        : {torch.__version__}")
print(f"CUDA available       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name}  ({vram:.2f} GB)")

# ── Import the right class ────────────────────────────────────────────────
print("\n" + "─" * 72)
print("Importing Qwen3VLForConditionalGeneration (this is the CORRECT class)")
print("─" * 72)
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print("Import OK.")
except ImportError as e:
    print(f"IMPORT FAILED: {e}")
    print("transformers >= 4.57 is required. Try: pip install -U 'transformers>=4.57'")
    sys.exit(1)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# ── Load ──────────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_ID} (bf16, device_map='auto')...")
print("Watch for 'newly initialized' warnings — there should be NONE.\n")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,          # new 'dtype' arg; 'torch_dtype' is deprecated
    device_map="auto",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("\nModel + processor loaded.")

# ── VRAM after load ───────────────────────────────────────────────────────
if torch.cuda.is_available():
    print("\nVRAM after load:")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        res   = torch.cuda.memory_reserved(i)  / 1e9
        print(f"  GPU {i}: allocated {alloc:.2f} GB  reserved {res:.2f} GB")

# ── Config introspection ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("CONFIG")
print("=" * 72)

print(f"\nmodel.config type: {type(model.config).__name__}")
print(f"top-level attrs (first 25): {list(vars(model.config).keys())[:25]}")

if hasattr(model.config, "vision_config"):
    vc = model.config.vision_config
    print(f"\nvision_config type: {type(vc).__name__}")
    print(f"  hidden_size              : {getattr(vc, 'hidden_size', 'N/A')}")
    print(f"  intermediate_size        : {getattr(vc, 'intermediate_size', 'N/A')}")
    print(f"  num_hidden_layers        : {getattr(vc, 'num_hidden_layers', 'N/A')}")
    print(f"  num_attention_heads      : {getattr(vc, 'num_attention_heads', 'N/A')}")
    print(f"  patch_size               : {getattr(vc, 'patch_size', 'N/A')}")
    print(f"  spatial_merge_size       : {getattr(vc, 'spatial_merge_size', 'N/A')}")
    print(f"  temporal_patch_size      : {getattr(vc, 'temporal_patch_size', 'N/A')}")
    print(f"  deepstack_visual_indexes : {getattr(vc, 'deepstack_visual_indexes', 'N/A')}")
    print(f"  out_hidden_size          : {getattr(vc, 'out_hidden_size', 'N/A')}")
else:
    print("\nNo vision_config found — inspect config manually.")

# ── Locate vision tower ───────────────────────────────────────────────────
print("\n" + "=" * 72)
print("VISION TOWER LOCATION")
print("=" * 72)

vis = None
vis_path = None
for path in ("visual", "model.visual", "model.model.visual"):
    obj = model
    ok = True
    for part in path.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            ok = False
            break
    if ok:
        vis = obj
        vis_path = path
        break

if vis is None:
    print("Could not find vision tower at any known path.")
    print("Top-level children of model:")
    for name, _ in model.named_children():
        print(f"  {name}")
    sys.exit(1)

print(f"\nvision tower at: model.{vis_path}  (type: {type(vis).__name__})")
n_vis_params = sum(p.numel() for p in vis.parameters()) / 1e6
print(f"vision tower parameters: {n_vis_params:.1f}M")

# ── Top-level structure of vision tower ───────────────────────────────────
print("\nTop-level children of vision tower:")
for name, child in vis.named_children():
    n_p = sum(p.numel() for p in child.parameters()) / 1e6
    print(f"  {name:20s}  {type(child).__name__:30s}  ({n_p:.1f}M)")

# ── Inventory of Linear leaf names (for LoRA target_modules) ──────────────
print("\n" + "=" * 72)
print("LINEAR LAYERS IN VISION TOWER — leaf names (for LoRA targeting)")
print("=" * 72)
names = Counter()
for name, module in vis.named_modules():
    if isinstance(module, torch.nn.Linear):
        leaf = name.rsplit(".", 1)[-1]
        names[leaf] += 1

for leaf, count in sorted(names.items(), key=lambda x: -x[1]):
    print(f"  {leaf:25s}  ×{count}")

print("\nFirst occurrence full paths (for reference):")
seen = set()
for name, module in vis.named_modules():
    if isinstance(module, torch.nn.Linear):
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in seen:
            seen.add(leaf)
            print(f"  {name}")

# ── Dummy forward pass ────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("DUMMY FORWARD PASS (random 512x512 RGB image)")
print("=" * 72)

rng = np.random.default_rng(0)
img_np = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
pil_img = Image.fromarray(img_np)

proc_out = processor.image_processor(images=[pil_img], return_tensors="pt")
print(f"\nprocessor output keys : {list(proc_out.keys())}")
for k, v in proc_out.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k:20s} shape={tuple(v.shape)} dtype={v.dtype}")

device = next(model.parameters()).device
pv  = proc_out["pixel_values"].to(device=device, dtype=torch.bfloat16)
thw = proc_out["image_grid_thw"].to(device=device)

print(f"\nCalling vision tower: visual(hidden_states=pv, grid_thw=thw)")
try:
    with torch.no_grad():
        out = vis(hidden_states=pv, grid_thw=thw)
except Exception as e:
    print(f"Vision tower call failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print(f"\nOutput type: {type(out).__name__}")

def _describe(label, val, indent=2):
    pad = " " * indent
    if val is None:
        print(f"{pad}{label}: None")
    elif isinstance(val, torch.Tensor):
        print(f"{pad}{label}: Tensor  shape={tuple(val.shape)}  dtype={val.dtype}")
    elif isinstance(val, (list, tuple)):
        print(f"{pad}{label}: {type(val).__name__}  len={len(val)}")
        for i, x in enumerate(val):
            _describe(f"[{i}]", x, indent + 4)
    else:
        print(f"{pad}{label}: type={type(val).__name__}")

# 1. If it's a tuple/list
if isinstance(out, (tuple, list)):
    print(f"Output is {type(out).__name__} of length {len(out)}")
    for i, item in enumerate(out):
        _describe(f"out[{i}]", item)

# 2. If it has attributes (dataclass-like)
for attr in ("last_hidden_state", "pooler_output", "deepstack_features", "hidden_states"):
    if hasattr(out, attr):
        _describe(attr, getattr(out, attr))

# 3. If it's a dict-like
if hasattr(out, "keys"):
    try:
        keys = list(out.keys())
        print(f"\nKeys: {keys}")
        for k in keys:
            _describe(k, out[k])
    except Exception:
        pass

# ── Sanity check: expected shapes for 512x512 image ───────────────────────
print("\nExpected shapes for 512x512 input:")
print(f"  patches before merge : (H/patch × W/patch) = ({512//16} × {512//16}) = {(512//16)**2}")
print(f"  tokens before merger : {(512//16)**2} × temporal ({getattr(vc, 'temporal_patch_size', 2)}) = {(512//16)**2 * getattr(vc, 'temporal_patch_size', 2)}")
sm = getattr(vc, "spatial_merge_size", 2)
tp = getattr(vc, "temporal_patch_size", 2)
merged = ((512 // 16) // sm) ** 2
print(f"  tokens after merger  : ({(512//16)//sm} × {(512//16)//sm}) = {merged}  in dim={getattr(vc,'out_hidden_size','?')}")
print(f"  deepstack layers     : {getattr(vc, 'deepstack_visual_indexes', 'N/A')}")
print(f"  (each deepstack feature also goes through its own merger → same shape as above)")

# ── VRAM after forward ────────────────────────────────────────────────────
if torch.cuda.is_available():
    print("\nVRAM after forward pass:")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        peak  = torch.cuda.max_memory_allocated(i) / 1e9
        print(f"  GPU {i}: allocated {alloc:.2f} GB  peak {peak:.2f} GB")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("NEXT STEPS")
print("=" * 72)
print("""
Sprawdź powyższy output pod kątem:
  1. Czy były WARNINGI "Some weights ... were not initialized"?
     - jeśli TAK → nadal mamy problem z ładowaniem, zatrzymaj się i napisz
     - jeśli NIE → encoder jest prawidłowy, możemy iść dalej

  2. vision_config.hidden_size  → wejście SegDecoder (deepstack_features dim)
     config.hidden_size (LLM)    → wejście jeśli bierzemy last_hidden_state

  3. patch_size × spatial_merge_size → efektywny stride, rozmiar gridu features

  4. deepstack_visual_indexes → z których warstw ViT DeepStack zbiera features
     (liczba powinna odpowiadać len(deepstack_features))

  5. Linear leaf names        → LoRA target_modules
     Oczekiwane: 'qkv' dla attention, 'linear_fc1'/'linear_fc2' dla MLP
     (a NIE 'q_proj'/'k_proj'/'v_proj' jak w starym kodzie)

  6. last_hidden_state.shape[-1] vs vision_config.hidden_size
     - jeśli różne → output jest po PatchMerger (LLM dim)
     - shape powinien być (1, N_tokens_merged, LLM_dim)

  7. VRAM: 8B bf16 ≈ 16 GB. Z device_map='auto' rozrzuca na 2 GPU.
""")
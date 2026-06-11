import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# CONFIGURATION:
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

data_files = {
    "UNet Baseline (50ep)": load_json(os.path.join(RESULTS_DIR, "unet_baseline_full_50ep.json")),
    "Qwen+UNet Refinement": load_json(os.path.join(RESULTS_DIR, "qwen_unet_metrics.json")),
    "Qwen LoRA (Deepstack)": load_json(os.path.join(RESULTS_DIR, "qwen_metrics_LoRA_deepstack.json")),
    "Qwen Frozen (Deepstack)": load_json(os.path.join(RESULTS_DIR, "qwen_metrics_frozen_deepstack.json")),
    "Qwen Frozen (Last Only)": load_json(os.path.join(RESULTS_DIR, "qwen_metrics_frozen_last_only.json")),
}

# removing of missing files
data_files = {k: v for k, v in data_files.items() if v is not None}

# Plot 1: Overall Metrics

metrics_to_plot = ["dice", "iou", "precision", "recall"]
records = []
for model_name, data in data_files.items():
    if "test_metrics" in data:
        for metric in metrics_to_plot:
            records.append({
                "Model": model_name,
                "Metric": metric.upper(),
                "Score": data["test_metrics"][metric]
            })

df_metrics = pd.DataFrame(records)

plt.figure(figsize=(14, 7))
ax = sns.barplot(data=df_metrics, x="Metric", y="Score", hue="Model")
plt.title("Porównanie metryk testowych między modelami", fontsize=16)
plt.ylim(0, 1.05)
plt.ylabel("Wynik", fontsize=12)
plt.xlabel("Metryka", fontsize=12)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "1_metrics_comparison_bar.png"), dpi=300)
plt.close()

# Plot 2: Training and Validation Curves

print("Generating 2. Learning Curves...")
fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=False)
for model_name, data in data_files.items():
    if "history" in data:
        epochs = [entry["epoch"] for entry in data["history"]]
        val_dice = [entry["val"]["dice"] for entry in data["history"]]
        val_loss = [entry["val"]["loss"] for entry in data["history"]]
        
        axes[0].plot(epochs, val_dice, label=model_name, linewidth=2, marker='o', markersize=4)
        axes[1].plot(epochs, val_loss, label=model_name, linewidth=2, marker='o', markersize=4)


axes[1].set_title("Zmiana funkcji straty na zbiorze walidacyjnym", fontsize=14)
axes[1].set_ylabel("Wartość funkcji straty")
axes[1].set_xlabel("Numer epoki")
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend()

axes[1].set_title("Zmiana funkcji straty na zbiorze walidacyjnym", fontsize=14)
axes[1].set_ylabel("Wartość funkcji straty")
axes[1].set_xlabel("Numer epoki")
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "2_learning_curves.png"), dpi=300)
plt.close()

# Plot 3: Qwen vs Qwen-UNet Inference Comparison

infer_data = load_json(os.path.join(RESULTS_DIR, "infer/metrics.json"))
if infer_data:
    infer_records = []
    for mode in ["qwen_only", "qwen_unet"]:
        for metric in metrics_to_plot:
            infer_records.append({
                "Method": "Qwen" if mode == "qwen_only" else "Qwen + U-Net",
                "Metric": metric.upper(),
                "Score": infer_data[mode][metric]
            })
    
    df_infer = pd.DataFrame(infer_records)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_infer, x="Metric", y="Score", hue="Method", palette="Set2")
    plt.title("Wpływ wzmocnienia U-Net na Inferencje", fontsize=16)
    plt.ylim(0.5, 1.0)
    plt.ylabel("Wynik")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "3_inference_refinement.png"), dpi=300)
    plt.close()

# Plot 4: Difference between Baseline

baseline_name = "UNet Baseline (50ep)"

if baseline_name in df_metrics["Model"].values:
    baseline_scores = df_metrics[df_metrics["Model"] == baseline_name].set_index("Metric")["Score"].to_dict()

    df_delta = df_metrics.copy()
    df_delta["Delta"] = df_delta.apply(lambda row: row["Score"] - baseline_scores[row["Metric"]], axis=1)

    df_delta = df_delta[df_delta["Model"] != baseline_name]

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_delta, x="Metric", y="Delta", hue="Model", palette="coolwarm")
    plt.title(f"Poprawa/Pogorszenie wyników względem: {baseline_name}", fontsize=14)
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')  # Linia zera
    plt.ylabel("Różnica w wyniku (Delta)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    for container in ax.containers:
        ax.bar_label(container, fmt='%+.3f', padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "4_delta_baseline_plot.png"), dpi=300)
    plt.close()
else:
    print(f"Pominięto wykres 6: Brak modelu '{baseline_name}' w danych do porównania.")

# Plot 6: Models' Metrics

plt.figure(figsize=(10, 6))
sns.pointplot(data=df_metrics, x="Metric", y="Score", hue="Model", markers=["o", "s", "D", "X", "^"], linestyles="-",
              dodge=0.1)

plt.title("Profil wydajności modeli we wszystkich metrykach", fontsize=15)
plt.ylabel("Score", fontsize=12)
plt.ylim(0.2, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "6_metrics_profile_pointplot.png"), dpi=300)
plt.close()

# Plot 7: Training & Validation Curves (For each model)

import re

def clean_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name).strip('_')

for model_name, data in data_files.items():
    if "history" in data:
        epochs = [entry["epoch"] for entry in data["history"]]
        val_dice = [entry["val"]["dice"] for entry in data["history"]]
        val_loss = [entry["val"]["loss"] for entry in data["history"]]

        train_dice = [entry.get("train", {}).get("dice", entry.get("dice", np.nan)) for entry in data["history"]]
        train_loss = [entry.get("train", {}).get("loss", entry.get("loss", np.nan)) for entry in data["history"]]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(epochs, val_dice, label="Walidacja", linewidth=2.5, color="#1f77b4", linestyle='-')
        if not np.isnan(train_dice).all():
            axes[0].plot(epochs, train_dice, label="Trening", linewidth=2, color="#ff7f0e", linestyle='--', alpha=0.8)

        axes[0].set_title(f"Metryka Dice - {model_name}", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Wynik Dice", fontsize=12)
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].legend(loc="lower right")

        axes[1].plot(epochs, val_loss, label="Walidacja", linewidth=2.5, color="#1f77b4", linestyle='-')
        if not np.isnan(train_loss).all():
            axes[1].plot(epochs, train_loss, label="Trening", linewidth=2, color="#ff7f0e", linestyle='--', alpha=0.8)

        axes[1].set_title(f"Funkcja Straty (Loss) - {model_name}", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].set_xlabel("Epoka", fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].legend(loc="upper right")

        plt.tight_layout()

        safe_model_name = clean_filename(model_name)
        filename = f"7_learning_curves_{safe_model_name}.png"
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
        plt.close()

print("Indywidualne krzywe uczenia zostały wygenerowane.")

# Plot 8: Generalization Gap (Train vs Val vs Test Bar Chart)

gap_records = []

for model_name, data in data_files.items():
    test_dice = data.get("test_metrics", {}).get("dice", None)
    history = data.get("history", [])

    if history and test_dice is not None:
        last_epoch = history[-1]
        val_dice = last_epoch.get("val", {}).get("dice", None)

        train_dice = last_epoch.get("train", {}).get("dice", last_epoch.get("dice", None))

        if train_dice and val_dice:
            gap_records.append(
                {"Model": model_name, "Dataset Split": "1. Trening (Ostatnia Epoka)", "Dice": train_dice})
            gap_records.append(
                {"Model": model_name, "Dataset Split": "2. Walidacja (Ostatnia Epoka)", "Dice": val_dice})
            gap_records.append({"Model": model_name, "Dataset Split": "3. Test (Ostateczny)", "Dice": test_dice})

if gap_records:
    df_gap = pd.DataFrame(gap_records)

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=df_gap, x="Model", y="Dice", hue="Dataset Split", palette="crest")

    plt.title("Analiza Overfittingu: Trening vs Walidacja vs Test (Dice Score)", fontsize=16)
    plt.ylim(0, 1.05)
    plt.ylabel("Wynik Dice", fontsize=12)
    plt.legend(loc='lower right')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "8_generalization_gap.png"), dpi=300)
    plt.close()
else:
    print("Pominięto wykres 8: Brak danych o wynikach treningowych wewnątrz klucza 'history'.")
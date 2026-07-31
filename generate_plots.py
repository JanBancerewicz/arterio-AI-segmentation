"""
generate_plots.py
=================
Plots from results/*.json.

Default: generate BOTH language variants
  results/plots/     — Polish labels (local / ignored by git)
  results/plots_en/  — English labels (tracked)

Usage:
    python generate_plots.py              # both
    python generate_plots.py --lang pl
    python generate_plots.py --lang en
"""

from __future__ import annotations

import argparse
import json
import os
import re
from math import nan

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = "results"

# Display names for the legend (match results JSON keys).
DATA_SOURCES = {
    "UNet Baseline (50ep)": "unet_baseline_full_50ep.json",
    "Qwen + U-Net Refinement": "qwen_unet_metrics.json",
    "Qwen + U-Net Refinement (curriculum)": "qwen_unet_v4_metrics.json",
    "Qwen LoRA (DeepStack)": "qwen_metrics_LoRA_deepstack.json",
    "Qwen Frozen (DeepStack)": "qwen_metrics_frozen_deepstack.json",
    "Qwen Frozen (Last Only)": "qwen_metrics_frozen_last_only.json",
}

METRICS = ["dice", "iou", "precision", "recall"]

LABELS = {
    "pl": {
        "out_dir": os.path.join(RESULTS_DIR, "plots"),
        "metrics_title": "Porównanie metryk testowych między modelami",
        "score": "Wynik",
        "metric": "Metryka",
        "val_dice_title": "Zmiana Dice na zbiorze walidacyjnym",
        "val_loss_title": "Zmiana funkcji straty na zbiorze walidacyjnym",
        "epoch": "Numer epoki",
        "loss": "Wartość funkcji straty",
        "infer_title": "Wpływ wzmocnienia U-Net na inferencję",
        "delta_title": "Poprawa/pogorszenie względem: {baseline}",
        "delta_ylabel": "Różnica w wyniku (Delta)",
        "profile_title": "Profil wydajności modeli we wszystkich metrykach",
        "dice_title": "Metryka Dice — {model}",
        "loss_title": "Funkcja straty (Loss) — {model}",
        "dice_ylabel": "Wynik Dice",
        "train": "Trening",
        "val": "Walidacja",
        "gap_title": "Analiza overfittingu: trening vs walidacja vs test (Dice)",
        "gap_train": "1. Trening (ostatnia epoka)",
        "gap_val": "2. Walidacja (ostatnia epoka)",
        "gap_test": "3. Test (ostateczny)",
        "split": "Podział zbioru",
        "model": "Model",
    },
    "en": {
        "out_dir": os.path.join(RESULTS_DIR, "plots_en"),
        "metrics_title": "Test-set metrics by model",
        "score": "Score",
        "metric": "Metric",
        "val_dice_title": "Validation Dice over epochs",
        "val_loss_title": "Validation loss over epochs",
        "epoch": "Epoch",
        "loss": "Loss",
        "infer_title": "Effect of U-Net refinement at inference",
        "delta_title": "Change vs {baseline}",
        "delta_ylabel": "Score delta",
        "profile_title": "Model performance profile across metrics",
        "dice_title": "Dice — {model}",
        "loss_title": "Loss — {model}",
        "dice_ylabel": "Dice",
        "train": "Train",
        "val": "Validation",
        "gap_title": "Overfitting check: train vs val vs test (Dice)",
        "gap_train": "1. Train (last epoch)",
        "gap_val": "2. Val (last epoch)",
        "gap_test": "3. Test (final)",
        "split": "Split",
        "model": "Model",
    },
}


def load_json(filepath: str):
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    return None


def clean_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name).strip("_")


def load_data_files() -> dict:
    out = {}
    for name, filename in DATA_SOURCES.items():
        data = load_json(os.path.join(RESULTS_DIR, filename))
        if data is not None:
            out[name] = data
    return out


def build_metrics_df(data_files: dict) -> pd.DataFrame:
    records = []
    for model_name, data in data_files.items():
        if "test_metrics" not in data:
            continue
        for metric in METRICS:
            records.append({
                "Model": model_name,
                "Metric": metric.upper(),
                "Score": data["test_metrics"][metric],
            })
    return pd.DataFrame(records)


def generate_for_lang(lang: str, data_files: dict, df_metrics: pd.DataFrame) -> None:
    L = LABELS[lang]
    plots_dir = L["out_dir"]
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\n=== {lang.upper()} → {plots_dir}/ ===")

    # 1. Metrics bar
    print("1. Metrics comparison bar…")
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=df_metrics, x="Metric", y="Score", hue="Model")
    plt.title(L["metrics_title"], fontsize=16)
    plt.ylim(0, 1.05)
    plt.ylabel(L["score"], fontsize=12)
    plt.xlabel(L["metric"], fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "1_metrics_comparison_bar.png"), dpi=300)
    plt.close()

    # 2. Learning curves (all models, val only)
    print("2. Learning curves (all models)…")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=False)
    has_history = False
    for model_name, data in data_files.items():
        if "history" not in data:
            continue
        has_history = True
        epochs = [e["epoch"] for e in data["history"]]
        axes[0].plot(
            epochs, [e["val"]["dice"] for e in data["history"]],
            label=model_name, linewidth=2, marker="o", markersize=4,
        )
        axes[1].plot(
            epochs, [e["val"]["loss"] for e in data["history"]],
            label=model_name, linewidth=2, marker="o", markersize=4,
        )
    if has_history:
        axes[0].set_title(L["val_dice_title"], fontsize=14)
        axes[0].set_ylabel(L["dice_ylabel"])
        axes[0].set_xlabel(L["epoch"])
        axes[0].grid(True, linestyle="--", alpha=0.7)
        axes[0].legend()
        axes[1].set_title(L["val_loss_title"], fontsize=14)
        axes[1].set_ylabel(L["loss"])
        axes[1].set_xlabel(L["epoch"])
        axes[1].grid(True, linestyle="--", alpha=0.7)
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "2_learning_curves.png"), dpi=300)
        plt.close()
    else:
        plt.close()
        print("  skipped (no history)")

    # 3. Inference refinement
    print("3. Inference refinement…")
    infer_data = load_json(os.path.join(RESULTS_DIR, "infer", "metrics.json"))
    if infer_data:
        infer_records = []
        for mode in ["qwen_only", "qwen_unet"]:
            for metric in METRICS:
                infer_records.append({
                    "Method": "Qwen" if mode == "qwen_only" else "Qwen + U-Net",
                    "Metric": metric.upper(),
                    "Score": infer_data[mode][metric],
                })
        df_infer = pd.DataFrame(infer_records)
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_infer, x="Metric", y="Score", hue="Method", palette="Set2")
        plt.title(L["infer_title"], fontsize=16)
        plt.ylim(0.5, 1.0)
        plt.ylabel(L["score"])
        plt.xlabel(L["metric"])
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=10)
        plt.legend(loc="lower center")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "3_inference_refinement.png"), dpi=300)
        plt.close()
    else:
        print("  skipped (no results/infer/metrics.json)")

    # 4. Delta vs baseline
    print("4. Delta vs baseline…")
    baseline_name = "UNet Baseline (50ep)"
    if baseline_name in df_metrics["Model"].values:
        baseline_scores = (
            df_metrics[df_metrics["Model"] == baseline_name]
            .set_index("Metric")["Score"]
            .to_dict()
        )
        df_delta = df_metrics.copy()
        df_delta["Delta"] = df_delta.apply(
            lambda row: row["Score"] - baseline_scores[row["Metric"]], axis=1
        )
        df_delta = df_delta[df_delta["Model"] != baseline_name]

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df_delta, x="Metric", y="Delta", hue="Model", palette="coolwarm")
        plt.title(L["delta_title"].format(baseline=baseline_name), fontsize=14)
        plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
        plt.ylabel(L["delta_ylabel"], fontsize=12)
        plt.xlabel(L["metric"])
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        for container in ax.containers:
            ax.bar_label(container, fmt="%+.3f", padding=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "4_delta_baseline_plot.png"), dpi=300)
        plt.close()
    else:
        print(f"  skipped (missing '{baseline_name}')")

    # 6. Metrics profile
    print("6. Metrics profile…")
    n_models = max(1, df_metrics["Model"].nunique())
    markers = ["o", "s", "D", "X", "^", "v"][:n_models]
    plt.figure(figsize=(10, 6))
    sns.pointplot(
        data=df_metrics, x="Metric", y="Score", hue="Model",
        markers=markers, linestyles="-", dodge=0.1,
    )
    plt.title(L["profile_title"], fontsize=15)
    plt.ylabel(L["score"], fontsize=12)
    plt.xlabel(L["metric"])
    plt.ylim(0.2, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "6_metrics_profile_pointplot.png"), dpi=300)
    plt.close()

    # 7. Per-model curves
    print("7. Per-model learning curves…")
    for model_name, data in data_files.items():
        if "history" not in data:
            continue
        epochs = [e["epoch"] for e in data["history"]]
        val_dice = [e["val"]["dice"] for e in data["history"]]
        val_loss = [e["val"]["loss"] for e in data["history"]]
        train_dice = [
            e.get("train", {}).get("dice", e.get("dice", nan)) for e in data["history"]
        ]
        train_loss = [
            e.get("train", {}).get("loss", e.get("loss", nan)) for e in data["history"]
        ]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(epochs, val_dice, label=L["val"], linewidth=2.5, color="#1f77b4")
        if not np.isnan(train_dice).all():
            axes[0].plot(
                epochs, train_dice, label=L["train"], linewidth=2,
                color="#ff7f0e", linestyle="--", alpha=0.8,
            )
        axes[0].set_title(L["dice_title"].format(model=model_name), fontsize=14, fontweight="bold")
        axes[0].set_ylabel(L["dice_ylabel"], fontsize=12)
        axes[0].grid(True, linestyle="--", alpha=0.7)
        axes[0].legend(loc="lower right")

        axes[1].plot(epochs, val_loss, label=L["val"], linewidth=2.5, color="#1f77b4")
        if not np.isnan(train_loss).all():
            axes[1].plot(
                epochs, train_loss, label=L["train"], linewidth=2,
                color="#ff7f0e", linestyle="--", alpha=0.8,
            )
        axes[1].set_title(L["loss_title"].format(model=model_name), fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].set_xlabel(L["epoch"], fontsize=12)
        axes[1].grid(True, linestyle="--", alpha=0.7)
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        fname = f"7_learning_curves_{clean_filename(model_name)}.png"
        plt.savefig(os.path.join(plots_dir, fname), dpi=300)
        plt.close()

    # 8. Generalization gap
    print("8. Generalization gap…")
    gap_records = []
    for model_name, data in data_files.items():
        test_dice = data.get("test_metrics", {}).get("dice")
        history = data.get("history", [])
        if not history or test_dice is None:
            continue
        last = history[-1]
        val_dice = last.get("val", {}).get("dice")
        train_dice = last.get("train", {}).get("dice", last.get("dice"))
        if train_dice is None or val_dice is None:
            continue
        gap_records.extend([
            {"Model": model_name, L["split"]: L["gap_train"], "Dice": train_dice},
            {"Model": model_name, L["split"]: L["gap_val"], "Dice": val_dice},
            {"Model": model_name, L["split"]: L["gap_test"], "Dice": test_dice},
        ])

    if gap_records:
        df_gap = pd.DataFrame(gap_records)
        plt.figure(figsize=(14, 7))
        ax = sns.barplot(
            data=df_gap, x="Model", y="Dice", hue=L["split"], palette="crest",
        )
        plt.title(L["gap_title"], fontsize=16)
        plt.ylim(0, 1.05)
        plt.ylabel(L["dice_ylabel"], fontsize=12)
        plt.xlabel(L["model"])
        plt.legend(loc="lower right")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "8_generalization_gap.png"), dpi=300)
        plt.close()
    else:
        print("  skipped (no train history for gap plot)")

    print(f"Done → {plots_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate segmentation result plots")
    parser.add_argument(
        "--lang", choices=["pl", "en", "both"], default="both",
        help="pl → results/plots/, en → results/plots_en/, both → both (default)",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", palette="muted")
    data_files = load_data_files()
    if not data_files:
        raise SystemExit("No metrics JSON found in results/")
    df_metrics = build_metrics_df(data_files)
    if df_metrics.empty:
        raise SystemExit("No test_metrics in loaded JSON files")

    langs = ["pl", "en"] if args.lang == "both" else [args.lang]
    for lang in langs:
        generate_for_lang(lang, data_files, df_metrics)


if __name__ == "__main__":
    main()

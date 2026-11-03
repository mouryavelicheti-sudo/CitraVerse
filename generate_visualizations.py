#!/usr/bin/env python3
"""
Visualization stage: turns the numbers/logs already produced by the rest of
the pipeline (harmonize_datasets, train_teacher, train_student, quantize_model,
run_robustness, evaluate, generate_gradcam) into publication-quality PNG
figures for the paper.

This script trains NOTHING and computes NO new predictions. Every number it
plots is either read directly from an existing CSV/JSON output, or is a
straightforward, fully-attributable derived statistic computed from one of
those files (e.g. per-class F1 from outputs/predictions/test_predictions.csv,
or prediction-consistency from outputs/metrics/robustness_per_image.csv). If
a required input file is missing, the corresponding figure is skipped with a
logged warning -- it is never filled in with placeholder/guessed numbers.

Deliberately does NOT import torch/torchvision: every input here is already
on disk as CSV/JSON/PNG, so this stage runs fast and needs no GPU, no model
checkpoints loaded into memory, and no dataset re-scan.

Saves (all at 300 DPI, bbox_inches="tight"):
    outputs/figures/dataset_class_distribution.png
    outputs/figures/model_performance_comparison.png
    outputs/figures/model_efficiency_comparison.png
    outputs/figures/teacher_training_curves.png
    outputs/figures/student_distillation_curves.png
    outputs/figures/confusion_matrix_teacher.png
    outputs/figures/confusion_matrix_student_fp32.png
    outputs/figures/confusion_matrix_student_int8.png
    outputs/figures/per_class_f1_comparison.png
    outputs/figures/reliability_distribution.png
    outputs/figures/robustness_analysis.png
    outputs/figures/size_vs_f1.png
    outputs/figures/gradcam_comparison.png

Usage:
    python run_pipeline.py --visualize
    python scripts/generate_visualizations.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; never opens an interactive window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io_utils import load_json, load_yaml, resolve_path
from src.utils.logging_utils import get_logger

logger = get_logger("generate_visualizations")

sns.set_theme(style="whitegrid")
# IEEE-paper-friendly sizing: readable at column width, no oversized titles.
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
})

MODEL_VARIANT_ORDER = ["teacher", "student_fp32", "student_int8"]
MODEL_DISPLAY_NAME = {
    "teacher": "Teacher: ResNet18 FP32 (accuracy reference)",
    "student_fp32": "Student: MobileNetV3-Small FP32 (distilled)",
    "student_int8": "Student: MobileNetV3-Small INT8 (edge-deployed)",
}
# Multi-line version for x-axis tick labels (bar charts)
MODEL_TICK_NAME = {
    "teacher": "Teacher\nResNet18 FP32\n(reference)",
    "student_fp32": "Student\nMobileNetV3-S FP32\n(distilled)",
    "student_int8": "Student\nMobileNetV3-S INT8\n(edge-deployed)",
}
VARIANT_COLOR = {"teacher": "#1f77b4", "student_fp32": "#ff7f0e", "student_int8": "#2ca02c"}


def _save_fig(fig, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _first_present(df: pd.DataFrame, candidates: list):
    """Returns the first column name from `candidates` that actually exists
    in `df`, or None. Used so we never assume a column name without
    checking it's really there (e.g. the model-variant identifier column in
    outputs/metrics/evaluation_summary.csv is named `ablation_id` by
    src/evaluation/ablation.py, not `model`/`model_variant`)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _require_columns(df: pd.DataFrame, required: list, csv_path: str) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(
            f"{csv_path}: missing expected column(s) {missing} (found: {list(df.columns)}). "
            "Skipping the figure(s) that depend on them."
        )
        return False
    return True


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Robustly parses a column that SHOULD be boolean but may have round-tripped
    through CSV as the strings 'True'/'False' rather than a real bool dtype."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


# --- 1. Dataset class distribution --------------------------------------

def fig_dataset_class_distribution(metadata_path: str, canonical_classes: list, out_path: str):
    if not os.path.exists(metadata_path):
        logger.warning(f"{metadata_path} not found -- skipping dataset_class_distribution.png. "
                        "Run scripts/harmonize_datasets.py first.")
        return None
    df = pd.read_csv(metadata_path)
    if not _require_columns(df, ["canonical_class"], metadata_path):
        return None

    if "dataset" in df.columns:
        pivot = df.groupby(["canonical_class", "dataset"]).size().unstack(fill_value=0)
        pivot = pivot.reindex(canonical_classes, fill_value=0)
    else:
        counts = df["canonical_class"].value_counts().reindex(canonical_classes, fill_value=0)
        pivot = counts.to_frame(name="all_datasets")

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(pivot))
    palette = sns.color_palette("tab10", len(pivot.columns))
    for i, col in enumerate(pivot.columns):
        ax.bar(pivot.index, pivot[col].values, bottom=bottom, label=col, color=palette[i])
        bottom += pivot[col].values
    for i, total in enumerate(bottom):
        ax.text(i, total, f"{int(total)}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Canonical disease class")
    ax.set_ylabel("Number of images")
    ax.set_title("Harmonized Dataset Class Distribution (Actual Counts)")
    ax.legend(title="Source dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save_fig(fig, out_path)
    logger.info(f"Saved {out_path}")
    return out_path


# --- 2 & 3. Model performance / efficiency comparison ---------------------

def _load_evaluation_summary(eval_summary_path: str):
    if not os.path.exists(eval_summary_path):
        logger.warning(f"{eval_summary_path} not found -- skipping figures that depend on it. "
                        "Run scripts/evaluate.py first.")
        return None, None
    df = pd.read_csv(eval_summary_path)
    id_col = _first_present(df, ["ablation_id", "model_variant", "model"])
    if id_col is None:
        logger.warning(f"{eval_summary_path}: could not find a model-identifier column among "
                        f"['ablation_id', 'model_variant', 'model'] (found: {list(df.columns)}). "
                        "Skipping figures that depend on it.")
        return None, None
    df = df[df[id_col].isin(MODEL_VARIANT_ORDER)].copy()
    if df.empty:
        logger.warning(f"{eval_summary_path}: no rows matched known model variants {MODEL_VARIANT_ORDER} "
                        f"in column '{id_col}'. Skipping figures that depend on it.")
        return None, None
    df["_order"] = df[id_col].apply(MODEL_VARIANT_ORDER.index)
    df = df.sort_values("_order")
    return df, id_col


def fig_model_performance_comparison(eval_summary_path: str, out_path: str):
    df, id_col = _load_evaluation_summary(eval_summary_path)
    if df is None:
        return None
    metric_cols = [m for m in ["accuracy", "precision", "recall", "f1"] if m in df.columns]
    if not metric_cols:
        logger.warning(f"{eval_summary_path}: none of accuracy/precision/recall/f1 present "
                        f"(found: {list(df.columns)}). Skipping model_performance_comparison.png.")
        return None

    variants = df[id_col].tolist()
    display_names = [MODEL_TICK_NAME.get(v, v) for v in variants]
    x = np.arange(len(variants))
    n = len(metric_cols)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, metric in enumerate(metric_cols):
        values = df[metric].values.astype(float)
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize())
        for xi, v in zip(x + i * width, values):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(display_names)
    ax.set_ylabel("Score")
    ax.set_ylim(0, min(1.15, max(1.0, float(df[metric_cols].values.max()) * 1.15)))
    ax.set_title("Model Performance Comparison (Test Split)")
    ax.legend(title="Metric")
    _save_fig(fig, out_path)
    logger.info(f"Saved {out_path}")
    return out_path


def fig_model_efficiency_comparison(eval_summary_path: str, out_path: str):
    df, id_col = _load_evaluation_summary(eval_summary_path)
    if df is None:
        return None

    panels = [("model_size_mb", "Model size (MB)\n(lower is better)"),
              ("latency_ms", "Latency (ms)\n(lower is better)"),
              ("fps", "FPS\n(higher is better)")]
    panels = [(col, label) for col, label in panels if col in df.columns]
    if not panels:
        logger.warning(f"{eval_summary_path}: none of model_size_mb/latency_ms/fps present "
                        f"(found: {list(df.columns)}). Skipping model_efficiency_comparison.png.")
        return None

    variants = df[id_col].tolist()
    display_names = [MODEL_TICK_NAME.get(v, v) for v in variants]
    colors = [VARIANT_COLOR.get(v, "#7f7f7f") for v in variants]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (col, label) in zip(axes, panels):
        values = df[col].astype(float).values
        bars = ax.bar(display_names, values, color=colors)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}" if not np.isnan(v) else "N/A",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(label)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=8)
    fig.suptitle("Model Efficiency Comparison (measured, not projected)")
    _save_fig(fig, out_path)
    logger.info(f"Saved {out_path}")
    return out_path


# --- 4 & 5. Training / distillation curves --------------------------------

def fig_teacher_training_curves(history_path: str, out_path: str):
    full_history_path = resolve_path(history_path)
    if not os.path.exists(full_history_path):
        logger.warning(f"{history_path} not found -- skipping teacher_training_curves.png. "
                        "Run scripts/train_teacher.py first.")
        return None
    history = load_json(history_path)
    if not history:
        logger.warning(f"{history_path} is empty -- skipping teacher_training_curves.png.")
        return None
    df = pd.DataFrame(history)
    if not _require_columns(df, ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"], history_path):
        return None

    n_epochs = len(df)
    if n_epochs != 15:
        logger.warning(f"{history_path}: found {n_epochs} recorded epoch(s), not the 15 configured in "
                        "configs/training.yaml (teacher.epochs). Plotting the actual recorded history "
                        "rather than assuming 15 -- this may indicate an interrupted or reconfigured run.")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_loss.plot(df["epoch"], df["train_loss"], marker="o", label="Train loss")
    ax_loss.plot(df["epoch"], df["val_loss"], marker="o", label="Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Teacher Loss (learns from ground-truth labels)")
    ax_loss.legend()

    ax_acc.plot(df["epoch"], df["train_acc"], marker="o", label="Train accuracy")
    ax_acc.plot(df["epoch"], df["val_acc"], marker="o", label="Validation accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Teacher Accuracy (ResNet18 FP32)")
    ax_acc.legend()

    fig.suptitle(f"Stage 1 - Teacher Training: ResNet18 FP32 learns disease classes ({n_epochs} epochs)")
    _save_fig(fig, out_path)
    logger.info(f"Saved {out_path}")
    return out_path


def fig_student_distillation_curves(history_path: str, out_path: str):
    full_history_path = resolve_path(history_path)
    if not os.path.exists(full_history_path):
        logger.warning(f"{history_path} not found -- skipping student_distillation_curves.png. "
                        "Run scripts/train_student.py first.")
        return None
    history = load_json(history_path)
    if not history:
        logger.warning(f"{history_path} is empty -- skipping student_distillation_curves.png.")
        return None
    df = pd.DataFrame(history)
    if not _require_columns(df, ["epoch"], history_path):
        return None

    n_epochs = len(df)
    if n_epochs != 15:
        logger.warning(f"{history_path}: found {n_epochs} recorded epoch(s), not the 15 configured in "
                        "configs/training.yaml (student.epochs). Plotting the actual recorded history.")

    loss_cols = [("train_total_loss", "Total"), ("train_L_CE", "CE"),
                 ("train_L_KD", "KD"), ("train_L_signature", "Signature")]
    loss_cols = [(c, lbl) for c, lbl in loss_cols if c in df.columns]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax_loss = axes[0]
    for col, lbl in loss_cols:
        ax_loss.plot(df["epoch"], df[col], marker="o", label=lbl)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Student Loss = CE (labels) + KD (Teacher) + Signature (Z(x))")
    if loss_cols:
        ax_loss.legend()

    ax_acc = axes[1]
    if "train_acc" in df.columns:
        ax_acc.plot(df["epoch"], df["train_acc"], marker="o", label="Train accuracy")
    if "val_acc" in df.columns:
        ax_acc.plot(df["epoch"], df["val_acc"], marker="o", label="Validation accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Student Accuracy (MobileNetV3-Small FP32)")
    ax_acc.legend()

    ax_sig = axes[2]
    if "val_signature_mse" in df.columns:
        ax_sig.plot(df["epoch"], df["val_signature_mse"], marker="o", color="#d62728",
                    label="Validation signature MSE")
        ax_sig.legend()
    ax_sig.set_xlabel("Epoch")
    ax_sig.set_ylabel("MSE(Z_S(x), Z_T(x))")
    ax_sig.set_title("Evidence-Signature Preservation (Student vs. Z(x))")

    fig.suptitle(f"Stage 2 - Student Distillation: MobileNetV3-Small learns from Teacher + preserves Z(x) ({n_epochs} epochs)")
    _save_fig(fig, out_path)
    logger.info(f"Saved {out_path}")
    return out_path


# --- 6, 7, 8. Confusion matrices -----------------------------------------

def fig_confusion_matrices(test_predictions_path: str, canonical_classes: list, out_dir: str):
    if not os.path.exists(test_predictions_path):
        logger.warning(f"{test_predictions_path} not found -- skipping confusion matrix figures. "
                        "Run scripts/evaluate.py first.")
        return []
    df = pd.read_csv(test_predictions_path)
    if not _require_columns(df, ["model_variant", "ground_truth_class", "predicted_class"], test_predictions_path):
        return []

    created = []
    for variant in MODEL_VARIANT_ORDER:
        sub = df[df["model_variant"] == variant]
        if sub.empty:
            logger.warning(f"{test_predictions_path}: no rows for model_variant='{variant}' -- "
                            f"skipping confusion_matrix_{variant}.png.")
            continue
        cm = pd.crosstab(sub["ground_truth_class"], sub["predicted_class"])
        cm = cm.reindex(index=canonical_classes, columns=canonical_classes, fill_value=0)

        fig, ax = plt.subplots(figsize=(max(6, len(canonical_classes) * 0.9), max(5, len(canonical_classes) * 0.8)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                    xticklabels=canonical_classes, yticklabels=canonical_classes, ax=ax)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"Confusion Matrix (test split, counts)\n{MODEL_DISPLAY_NAME.get(variant, variant)}")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        out_path = os.path.join(out_dir, f"confusion_matrix_{variant}.png")
        _save_fig(fig, out_path)
        logger.info(f"Saved {out_path}")
        created.append(out_path)
    return created


# --- 9. Per-class F1 -------------------------------------------------------

def fig_per_class_f1_comparison(test_predictions_path: str, canonical_classes: list, out_path: str):
    if not os.path.exists(test_predictions_path):
        logger.warning(f"{test_predictions_path} not found -- skipping per_class_f1_comparison.png.")
        return None
    df = pd.read_csv(test_predictions_path)
    if not _require_columns(df, ["model_variant", "ground_truth_class", "predicted_class"], test_predictions_path):
        return None

    from sklearn.metrics import f1_score

    per_variant_f1 = {}
    for variant in MODEL_VARIANT_ORDER:
        sub = df[df["model_variant"] == variant]
        if sub.empty:
            continue
        f1s = f1_score(sub["ground_truth_class"], sub["predicted_class"],
                        labels=canonical_classes, average=None, zero_division=0)
        per_variant_f1[variant] = f1s

    if not per_variant_f1:
        logger.warning(f"{test_predictions_path}: no known model variants found -- "
                        "skipping per_class_f1_comparison.png.")
        return None

    x = np.arange(len(canonical_classes))
    n = len(per_variant_f1)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (variant, f1s) in enumerate(per_variant_f1.items()):
        ax.bar(x + i * width, f1s, width, label=MODEL_DISPLAY_NAME.get(variant, variant),
               color=VARIANT_COLOR.get(variant, None))
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(canonical_classes, rotation=20, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1-score (computed from actual test predictions)")
    ax.legend()
    _save_fig(fig, out_path)
    logger.info(f"Saved {out_path}")
    return out_path


# --- 10. Reliability distribution -----------------------------------------

def fig_reliability_distribution(test_predictions_path: str, eval_full_path: str,
                                  reliability_config_path: str, out_path: str):
    if not os.path.exists(test_predictions_path):
        logger.warning(f"{test_predictions_path} not found -- skipping reliability_distribution.png.")
        return None
    df = pd.read_csv(test_predictions_path)
    if not _require_columns(df, ["model_variant", "reliability_score"], test_predictions_path):
        return None

    tau = None
    full_eval_full_path = resolve_path(eval_full_path)
    if os.path.exists(full_eval_full_path):
        try:
            tau = load_json(eval_full_path).get("tau")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not read tau from {eval_full_path}: {e}")
    if tau is None:
        try:
            tau = load_yaml(reliability_config_path).get("tau", 0.6)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not read tau from {reliability_config_path}: {e}. Using tau=0.6.")
            tau = 0.6

    fig, ax = plt.subplots(figsize=(8, 5))
    present_variants = [v for v in MODEL_VARIANT_ORDER if not df[df["model_variant"] == v].empty]
    for variant in present_variants:
        sub = df[df["model_variant"] == variant]
        sns.kdeplot(sub["reliability_score"], ax=ax, label=MODEL_DISPLAY_NAME.get(variant, variant),
                    color=VARIANT_COLOR.get(variant, None), fill=True, alpha=0.15, clip=(0, 1))
    ax.axvline(tau, color="red", linestyle="--", label=f"tau = {tau:.2f} (decision threshold)")
    ax.set_xlabel("Reliability Score R(x)")
    ax.set_ylabel("Density")
    ax.set_title("Reliability Score Distribution")
    ax.legend()
    fig.subplots_adjust(bottom=0.2)
    # R(x) is a documented WEIGHTED HEURISTIC (w1*confidence + w2*prototype
    # agreement + w3*perturbation consistency; see src/reliability/
    # reliability_score.py), never fit/calibrated against ground-truth
    # correctness probabilities -- do not let a reader mistake this axis for
    # a calibrated probability of correctness.
    fig.text(0.5, 0.02,
              "R(x) = w1*confidence + w2*prototype agreement + w3*perturbation consistency "
              "(a weighted heuristic score in [0, 1]; NOT a calibrated probability of correctness).",
              ha="center", fontsize=8, style="italic")
    _save_fig(fig, out_path)
    logger.info(f"Saved {out_path}")
    return out_path


# --- 11. Robustness analysis -----------------------------------------------

def fig_robustness_analysis(robustness_csv_path: str, out_path: str):
    if not os.path.exists(robustness_csv_path):
        logger.warning(f"{robustness_csv_path} not found -- skipping robustness_analysis.png. "
                        "Run scripts/run_robustness.py first.")
        return None
    df = pd.read_csv(robustness_csv_path)
    if not _require_columns(df, ["perturbation", "prediction_flipped", "reliability_delta"], robustness_csv_path):
        return None

    flipped = _to_bool_series(df["prediction_flipped"])
    consistency = 1.0 - flipped.groupby(df["perturbation"]).mean()
    mean_delta = df.groupby("perturbation")["reliability_delta"].mean()
    perturbations = sorted(set(consistency.index) | set(mean_delta.index))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(perturbations, [consistency.get(p, np.nan) for p in perturbations], color="#1f77b4")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Prediction consistency (1 - flip rate)")
    ax1.set_xlabel("Perturbation type")
    ax1.set_title("Prediction Consistency Under Perturbation")
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    delta_values = [mean_delta.get(p, np.nan) for p in perturbations]
    colors = ["#d62728" if v is not None and not np.isnan(v) and v < 0 else "#2ca02c" for v in delta_values]
    ax2.bar(perturbations, delta_values, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Mean change in reliability score (perturbed - original)")
    ax2.set_xlabel("Perturbation type")
    ax2.set_title("Reliability Change Under Perturbation")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Perturbation Robustness Analysis - Student: MobileNetV3-Small FP32 (distilled)")
    fig.subplots_adjust(bottom=0.32, top=0.82)
    fig.text(0.5, 0.01,
              "Measured on the Student FP32 checkpoint only (scripts/run_robustness.py evaluates a single "
              "model variant); Teacher/INT8 robustness was not separately measured in this run. "
              "\"Prediction consistency\" is 1 minus the observed prediction flip rate, not an accuracy metric.",
              ha="center", fontsize=8, style="italic", wrap=True)
    _save_fig(fig, out_path)
    logger.info(f"Saved {out_path}")
    return out_path


# --- 12. Model size vs. F1 -------------------------------------------------

def fig_size_vs_f1(eval_summary_path: str, out_path: str):
    df, id_col = _load_evaluation_summary(eval_summary_path)
    if df is None:
        return None
    if not _require_columns(df, ["model_size_mb", "f1"], eval_summary_path):
        return None

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for _, row in df.iterrows():
        variant = row[id_col]
        ax.scatter(row["model_size_mb"], row["f1"], s=90, color=VARIANT_COLOR.get(variant, "#7f7f7f"),
                   label=MODEL_DISPLAY_NAME.get(variant, variant), zorder=3)
        ax.annotate(MODEL_DISPLAY_NAME.get(variant, variant),
                    (row["model_size_mb"], row["f1"]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Test F1-score (macro)")
    ax.set_title("Model Size vs. Performance")
    _save_fig(fig, out_path)
    logger.info(f"Saved {out_path}")
    return out_path


# --- 13. Grad-CAM contact sheet (reads existing images only) --------------

def fig_gradcam_comparison(gradcam_manifest_path: str, canonical_classes: list, out_path: str):
    if not os.path.exists(gradcam_manifest_path):
        logger.warning(f"{gradcam_manifest_path} not found -- skipping gradcam_comparison.png. "
                        "Run scripts/generate_gradcam.py first.")
        return None
    manifest = load_json(gradcam_manifest_path)
    if not manifest:
        logger.warning(f"{gradcam_manifest_path} is empty -- skipping gradcam_comparison.png.")
        return None

    # One representative entry per canonical class (the first one found, in
    # canonical class order), so the contact sheet stays a readable grid
    # regardless of how many samples-per-class generate_gradcam.py produced.
    by_class = {}
    for entry in manifest:
        cls = entry.get("ground_truth_class")
        if cls in canonical_classes and cls not in by_class:
            by_class[cls] = entry
    rows = [(cls, by_class[cls]) for cls in canonical_classes if cls in by_class]
    if not rows:
        logger.warning(f"{gradcam_manifest_path}: no entries matched the configured canonical classes. "
                        "Skipping gradcam_comparison.png.")
        return None

    missing_files = []
    for cls, entry in rows:
        for key in ("teacher_gradcam", "student_fp32_gradcam"):
            p = entry.get(key)
            if not p or not os.path.exists(p):
                missing_files.append((cls, key, p))
    if missing_files:
        logger.warning(f"gradcam_comparison.png: {len(missing_files)} referenced Grad-CAM image file(s) "
                        f"listed in the manifest are missing on disk (e.g. {missing_files[0]}). "
                        "These original files are NOT regenerated by this script; re-run "
                        "scripts/generate_gradcam.py if you need them.")
        rows = [(cls, entry) for cls, entry in rows
                if os.path.exists(entry.get("teacher_gradcam", "")) and os.path.exists(entry.get("student_fp32_gradcam", ""))]
    if not rows:
        logger.warning("gradcam_comparison.png: no rows had both Grad-CAM images present on disk. Skipping.")
        return None

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(6, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    for i, (cls, entry) in enumerate(rows):
        teacher_img = plt.imread(entry["teacher_gradcam"])
        student_img = plt.imread(entry["student_fp32_gradcam"])

        axes[i, 0].imshow(teacher_img)
        axes[i, 1].imshow(student_img)
        for j in (0, 1):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(cls, fontsize=9)

        teacher_pred = entry.get("teacher_prediction", "?")
        student_pred = entry.get("student_fp32_prediction", "?")
        axes[i, 0].set_xlabel(f"pred: {teacher_pred}", fontsize=8)
        axes[i, 1].set_xlabel(f"pred: {student_pred}", fontsize=8)

        if i == 0:
            axes[i, 0].set_title("Teacher\nResNet18 FP32 (reference)", fontsize=9)
            axes[i, 1].set_title("Student\nMobileNetV3-S FP32 (distilled)", fontsize=9)

    fig.suptitle("Grad-CAM: where Teacher (reference) and Student (distilled) look (test images, per class)")
    _save_fig(fig, out_path)
    logger.info(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="outputs/dataset_profiles/harmonized_metadata.csv")
    parser.add_argument("--class-mapping-config", default="configs/class_mapping.yaml")
    parser.add_argument("--reliability-config", default="configs/reliability.yaml")
    parser.add_argument("--evaluation-summary", default="outputs/metrics/evaluation_summary.csv")
    parser.add_argument("--evaluation-full", default="outputs/metrics/evaluation_full.json")
    parser.add_argument("--test-predictions", default="outputs/predictions/test_predictions.csv")
    parser.add_argument("--teacher-history", default="outputs/metrics/teacher_training_history.json")
    parser.add_argument("--student-history", default="outputs/metrics/student_training_history.json")
    parser.add_argument("--robustness-per-image", default="outputs/metrics/robustness_per_image.csv")
    parser.add_argument("--gradcam-manifest", default="outputs/gradcam/gradcam_manifest.json")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    canonical_classes = load_yaml(args.class_mapping_config).get("canonical_classes", [])
    if not canonical_classes:
        logger.error(f"{args.class_mapping_config} has no canonical_classes -- cannot generate figures.")
        return

    out_dir = resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    def p(name):
        return os.path.join(out_dir, name)

    generated = []

    result = fig_dataset_class_distribution(resolve_path(args.metadata), canonical_classes,
                                             p("dataset_class_distribution.png"))
    if result:
        generated.append(result)

    result = fig_model_performance_comparison(resolve_path(args.evaluation_summary),
                                               p("model_performance_comparison.png"))
    if result:
        generated.append(result)

    result = fig_model_efficiency_comparison(resolve_path(args.evaluation_summary),
                                              p("model_efficiency_comparison.png"))
    if result:
        generated.append(result)

    result = fig_teacher_training_curves(args.teacher_history, p("teacher_training_curves.png"))
    if result:
        generated.append(result)

    result = fig_student_distillation_curves(args.student_history, p("student_distillation_curves.png"))
    if result:
        generated.append(result)

    generated.extend(fig_confusion_matrices(resolve_path(args.test_predictions), canonical_classes, out_dir))

    result = fig_per_class_f1_comparison(resolve_path(args.test_predictions), canonical_classes,
                                          p("per_class_f1_comparison.png"))
    if result:
        generated.append(result)

    result = fig_reliability_distribution(resolve_path(args.test_predictions), args.evaluation_full,
                                           args.reliability_config, p("reliability_distribution.png"))
    if result:
        generated.append(result)

    result = fig_robustness_analysis(resolve_path(args.robustness_per_image), p("robustness_analysis.png"))
    if result:
        generated.append(result)

    result = fig_size_vs_f1(resolve_path(args.evaluation_summary), p("size_vs_f1.png"))
    if result:
        generated.append(result)

    result = fig_gradcam_comparison(resolve_path(args.gradcam_manifest), canonical_classes,
                                     p("gradcam_comparison.png"))
    if result:
        generated.append(result)

    print("\n" + "=" * 70)
    print("VISUALIZATION SUMMARY")
    print("=" * 70)
    print(f"Generated figures: {len(generated)}")
    print(f"Output directory: {args.out_dir}")
    print("=" * 70)
    if len(generated) < 13:
        print(f"\nNote: {13 - len(generated)} figure(s) were skipped because their required input "
              "file(s) were not found or lacked the expected columns -- see the warnings above for "
              "exactly which pipeline stage to run first. No figure is ever filled in with placeholder "
              "or fabricated data.")


if __name__ == "__main__":
    main()

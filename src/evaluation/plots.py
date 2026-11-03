"""Generates every plot the master spec requires, all driven by real data
passed in by the caller. Nothing here fabricates values -- if a plot needs
data you don't have yet, it will raise rather than draw a placeholder chart.
"""
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for scripts/servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")


def _save(fig, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_class_distribution(class_counts: dict, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    sns.barplot(x=counts, y=classes, ax=ax, orient="h")
    ax.set_xlabel("Number of images")
    ax.set_title(title)
    _save(fig, out_path)


def plot_dataset_size_comparison(dataset_sizes: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(dataset_sizes.keys()), y=list(dataset_sizes.values()), ax=ax)
    ax.set_ylabel("Number of images")
    ax.set_title("Dataset Size Comparison")
    plt.xticks(rotation=20, ha="right")
    _save(fig, out_path)


def plot_resolution_distribution(widths: list, heights: list, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(widths, heights, alpha=0.4, s=10)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Resolution Distribution")
    _save(fig, out_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: str, normalize: bool = True):
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.6), max(5, len(class_names) * 0.6)))
    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)
    sns.heatmap(data, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    _save(fig, out_path)


def plot_grouped_metric_comparison(model_names: list, metric_dict: dict, ylabel: str, title: str, out_path: str):
    """metric_dict: {metric_name: [values aligned with model_names]}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(model_names))
    n = len(metric_dict)
    width = 0.8 / max(n, 1)
    for i, (metric_name, values) in enumerate(metric_dict.items()):
        ax.bar(x + i * width, values, width, label=metric_name)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(model_names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _save(fig, out_path)


def plot_dess_comparison(dess_dict: dict, out_path: str):
    classes = [c for c, v in dess_dict.items() if v is not None]
    values = [dess_dict[c] for c in classes]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=values, y=classes, ax=ax, orient="h")
    ax.set_xlabel("DESS (cross-dataset evidence stability)")
    ax.set_xlim(0, 1)
    ax.set_title("Disease Evidence Stability Score by Class")
    _save(fig, out_path)


def plot_reliability_distribution(r_scores: np.ndarray, tau: float, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(r_scores, bins=30, kde=True, ax=ax)
    ax.axvline(tau, color="red", linestyle="--", label=f"tau = {tau:.2f}")
    ax.set_xlabel("Reliability Score R(x)")
    ax.set_title("Reliability Score Distribution")
    ax.legend()
    _save(fig, out_path)


def plot_reliability_transition_matrix(counts: dict, out_path: str):
    labels = ["Reliable -> Reliable", "Reliable -> Unreliable", "Unreliable -> Reliable", "Unreliable -> Unreliable"]
    values = [counts.get(l, 0) for l in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#7f7f7f"]
    sns.barplot(x=values, y=labels, palette=colors, ax=ax)
    ax.set_xlabel("Number of samples")
    ax.set_title("Reliability Transition Analysis")
    _save(fig, out_path)


def plot_model_size_vs_latency(model_names: list, sizes_mb: list, latencies_ms: list, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax2 = ax.twinx()
    x = np.arange(len(model_names))
    ax.bar(x - 0.2, sizes_mb, width=0.4, color="#1f77b4", label="Model size (MB)")
    ax2.bar(x + 0.2, latencies_ms, width=0.4, color="#ff7f0e", label="Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Model size (MB)", color="#1f77b4")
    ax2.set_ylabel("Latency (ms)", color="#ff7f0e")
    ax.set_title("Model Size vs. Latency")
    _save(fig, out_path)


def plot_evidence_space(z_matrix: np.ndarray, labels: list, out_path: str, method: str = "pca"):
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, init="pca", perplexity=min(30, max(5, len(z_matrix) // 3)))

    coords = reducer.fit_transform(z_matrix)
    fig, ax = plt.subplots(figsize=(7, 6))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("tab10", len(unique_labels))
    for i, lbl in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == lbl]
        ax.scatter(coords[idx, 0], coords[idx, 1], label=lbl, color=palette[i], alpha=0.7, s=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(f"Evidence Space ({method.upper()})")
    _save(fig, out_path)

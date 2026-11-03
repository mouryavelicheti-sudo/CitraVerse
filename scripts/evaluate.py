#!/usr/bin/env python3
"""
MODULE IV/V: full evaluation pipeline, run on the TEST split only.

    Test Image
        -> Classifier -> P(y|x) -> C(x)
        -> Z(x) (precomputed, Module II)
        -> Prototype Agreement A(x)
        -> Perturbation Consistency E(x)   (from scripts/run_robustness.py output, if present)
        -> R(x) -> tau -> RELIABLE / UNCERTAIN

Computed for all available model variants (teacher, student_fp32,
student_int8), so the required Teacher-vs-Student and FP32-vs-INT8
comparisons are direct outputs of this script, not something assembled by
hand afterwards.

Saves:
    outputs/metrics/evaluation_full.json           (everything, machine-readable)
    outputs/metrics/evaluation_summary.csv          (one row per model variant)
    outputs/predictions/test_predictions.csv        (one row per test image x variant)
    outputs/plots/confusion_matrix_<variant>.png
    outputs/plots/reliability_distribution_<variant>.png
    outputs/plots/model_size_vs_latency.png
    outputs/plots/reliability_transition_<a>_vs_<b>.png

Usage:
    python run_pipeline.py --evaluate
    python scripts/evaluate.py
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.ablation import build_ablation_table
from src.evaluation.metrics import classification_metrics, get_confusion_matrix
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_model_size_vs_latency,
    plot_reliability_distribution,
    plot_reliability_transition_matrix,
)
from src.prototypes.prototype_builder import load_prototypes
from src.prototypes.similarity import compute_dess, prototype_agreement
from src.quantization.quantize import benchmark_latency, model_size_mb
from src.reliability.reliability_score import (
    ReliabilityWeights,
    reliability_decision,
    reliability_score,
    reliability_transition_analysis,
)
from src.utils.io_utils import load_yaml, resolve_path, save_json
from src.utils.logging_utils import get_logger
from src.utils.seed import get_device, set_seed

logger = get_logger("evaluate")


def _predict_all(model, loader, device, is_quantized=False):
    import torch
    model.eval()
    all_probs, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for images, labels, meta in loader:
            imgs = images if is_quantized else images.to(device)
            if is_quantized:
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
            else:
                logits, probs, _feat = model(imgs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(meta["image_path"])
    return np.concatenate(all_probs), np.concatenate(all_labels), all_paths


def main():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error("PyTorch is required for evaluation. Install with: pip install torch torchvision")
        return

    from src.data.image_dataset import AgriVerseImageDataset, agriverse_collate, build_transforms
    from src.models.classifier import DiseaseClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="outputs/dataset_profiles/harmonized_metadata.csv")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--reliability-config", default="configs/reliability.yaml")
    parser.add_argument("--class-mapping-config", default="configs/class_mapping.yaml")
    parser.add_argument("--evidence-index", default="outputs/features/evidence_index.csv")
    parser.add_argument("--z-signatures", default="outputs/features/Z_signatures.npz")
    parser.add_argument("--prototypes", default="outputs/prototypes/prototypes.pkl")
    parser.add_argument("--robustness-per-image", default="outputs/metrics/robustness_per_image.csv")
    parser.add_argument("--teacher-checkpoint", default="models/teacher_best.pt")
    parser.add_argument("--student-checkpoint", default="models/student_best.pt")
    parser.add_argument("--student-int8-checkpoint", default="models/student_int8.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.training_config)
    rel_cfg = load_yaml(args.reliability_config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    metadata_path = resolve_path(args.metadata)
    if not os.path.exists(metadata_path):
        logger.error(f"{metadata_path} not found. Run scripts/harmonize_datasets.py first.")
        return

    canonical_classes = load_yaml(args.class_mapping_config).get("canonical_classes", [])
    class_to_idx = {c: i for i, c in enumerate(canonical_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    image_size = cfg.get("image_size", 224)

    test_ds = AgriVerseImageDataset(metadata_path, "test", class_to_idx, build_transforms(image_size, train=False))
    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 32), shuffle=False,
                              num_workers=cfg.get("num_workers", 2), collate_fn=agriverse_collate)

    # --- Z(x) lookup (Module II, precomputed) for prototype agreement. ---
    evidence_index_path = resolve_path(args.evidence_index)
    z_signatures_path = resolve_path(args.z_signatures)
    z_lookup = {}
    if os.path.exists(evidence_index_path) and os.path.exists(z_signatures_path):
        idx_df = pd.read_csv(evidence_index_path)
        z_arr = np.load(z_signatures_path)["Z"]
        z_lookup = {row["image_path"]: z_arr[i] for i, row in idx_df.iterrows()}
    else:
        logger.warning("Z(x) evidence signatures not found -- prototype agreement A(x) will be omitted. "
                        "Run scripts/extract_evidence.py first for full reliability scoring.")

    prototypes_path = resolve_path(args.prototypes)
    prototypes = load_prototypes(prototypes_path) if os.path.exists(prototypes_path) else None
    if prototypes is None:
        logger.warning(f"{prototypes_path} not found -- prototype agreement A(x) will be omitted. "
                        f"Run scripts/build_prototypes.py first.")

    dess_metric = rel_cfg.get("dess", {}).get("similarity_metric", "cosine")
    dess_agg = rel_cfg.get("dess", {}).get("aggregate_across_domains", "mean")
    dess_by_class = compute_dess(prototypes, metric=dess_metric, aggregate=dess_agg) if prototypes else {}

    # --- Perturbation consistency E(x), from scripts/run_robustness.py output (if run). ---
    robustness_path = resolve_path(args.robustness_per_image)
    pcs_lookup = {}
    if os.path.exists(robustness_path):
        rob_df = pd.read_csv(robustness_path)
        pcs_lookup = rob_df.groupby("image_path")["PCS"].mean().to_dict()
    else:
        logger.warning(f"{robustness_path} not found -- E(x) will be treated as neutral (1.0) with weights "
                        f"renormalized over C(x)/A(x) only. Run scripts/run_robustness.py first for the real "
                        f"perturbation-consistency term.")

    weights_cfg = rel_cfg["weights"]
    tau = rel_cfg.get("tau", 0.6)

    # --- Load whichever model variants have been trained so far. ---
    teacher_backbone = cfg.get("teacher", {}).get("backbone", "resnet18")
    student_backbone = cfg.get("student", {}).get("backbone", "mobilenet_v3_small")

    variants = {}
    teacher_ckpt = resolve_path(args.teacher_checkpoint)
    if os.path.exists(teacher_ckpt):
        m = DiseaseClassifier.build(teacher_backbone, len(canonical_classes), pretrained=False).to(device)
        m.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=False))
        variants["teacher"] = {"model": m, "is_quantized": False, "net_for_bench": m.net}
    else:
        logger.warning(f"{teacher_ckpt} not found -- skipping teacher evaluation.")

    student_ckpt = resolve_path(args.student_checkpoint)
    if os.path.exists(student_ckpt):
        m = DiseaseClassifier.build(student_backbone, len(canonical_classes), pretrained=False).to(device)
        m.load_state_dict(torch.load(student_ckpt, map_location=device, weights_only=False))
        variants["student_fp32"] = {"model": m, "is_quantized": False, "net_for_bench": m.net}
    else:
        logger.warning(f"{student_ckpt} not found -- skipping student_fp32 evaluation.")

    int8_ckpt = resolve_path(args.student_int8_checkpoint)
    if os.path.exists(int8_ckpt):
        m_int8 = torch.load(int8_ckpt, map_location="cpu", weights_only=False)
        variants["student_int8"] = {"model": m_int8, "is_quantized": True, "net_for_bench": m_int8}
    else:
        logger.warning(f"{int8_ckpt} not found -- skipping student_int8 evaluation.")

    if not variants:
        logger.error("No trained model checkpoints found at all. Run train_teacher.py / train_student.py / "
                      "quantize_model.py first.")
        return

    all_results = {}
    predictions_rows = []
    r_scores_by_variant = {}
    reliable_by_variant = {}

    for variant_name, info in variants.items():
        probs, labels, paths = _predict_all(info["model"], test_loader, device, is_quantized=info["is_quantized"])
        preds = probs.argmax(axis=1)
        confidences = probs.max(axis=1)

        cls_metrics = classification_metrics(labels, preds, average="macro")
        cm = get_confusion_matrix(labels, preds, labels=list(range(len(canonical_classes))))
        plot_confusion_matrix(cm, canonical_classes, resolve_path(f"outputs/plots/confusion_matrix_{variant_name}.png"))

        r_scores = np.zeros(len(paths))
        reliable_flags = np.zeros(len(paths), dtype=bool)

        for i, path in enumerate(paths):
            pred_class = idx_to_class[int(preds[i])]
            conf = float(confidences[i])

            agreement = 0.0
            if prototypes is not None and path in z_lookup:
                agreement = prototype_agreement(z_lookup[path], pred_class, prototypes, metric=dess_metric)

            if path in pcs_lookup:
                pcs = float(pcs_lookup[path])
                weights = ReliabilityWeights(weights_cfg["w1_confidence"], weights_cfg["w2_prototype_agreement"],
                                              weights_cfg["w3_perturbation_consistency"])
                r = reliability_score(conf, agreement, pcs, weights)
            else:
                # Renormalize over C(x) and A(x) only when E(x) hasn't been measured for this image.
                w_sum = weights_cfg["w1_confidence"] + weights_cfg["w2_prototype_agreement"]
                w1 = weights_cfg["w1_confidence"] / w_sum
                w2 = weights_cfg["w2_prototype_agreement"] / w_sum
                r = w1 * conf + w2 * agreement

            r_scores[i] = r
            decision = reliability_decision(r, tau)
            reliable_flags[i] = decision == "RELIABLE"

            predictions_rows.append({
                "image_path": path, "model_variant": variant_name,
                "ground_truth_class": idx_to_class[int(labels[i])], "predicted_class": pred_class,
                "confidence": conf, "prototype_agreement": agreement,
                "perturbation_consistency": pcs_lookup.get(path), "reliability_score": r,
                "reliability_decision": decision,
            })

        r_scores_by_variant[variant_name] = r_scores
        reliable_by_variant[variant_name] = reliable_flags
        plot_reliability_distribution(r_scores, tau, resolve_path(f"outputs/plots/reliability_distribution_{variant_name}.png"))

        net_for_bench = info["net_for_bench"]
        dummy_input = torch.randn(1, 3, image_size, image_size)
        try:
            latency = benchmark_latency(net_for_bench.cpu() if not info["is_quantized"] else net_for_bench, dummy_input)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Latency benchmark failed for {variant_name}: {e}")
            latency = {"mean_latency_ms": None, "fps": None}
        size_mb = model_size_mb(net_for_bench)

        all_results[variant_name] = {
            **cls_metrics,
            "mean_confidence": float(confidences.mean()),
            "mean_prototype_agreement": float(np.mean([
                r["prototype_agreement"] for r in predictions_rows if r["model_variant"] == variant_name
            ])),
            "mean_reliability": float(r_scores.mean()),
            "reliable_fraction": float(reliable_flags.mean()),
            "model_size_mb": size_mb,
            "latency_ms": latency["mean_latency_ms"],
            "fps": latency.get("fps"),
            "confusion_matrix": cm.tolist(),
        }
        logger.info(f"[{variant_name}] acc={cls_metrics['accuracy']:.4f} f1={cls_metrics['f1']:.4f} "
                    f"mean_R={r_scores.mean():.4f} reliable_frac={reliable_flags.mean():.4f} "
                    f"size={size_mb:.2f}MB latency={latency['mean_latency_ms']}ms")

    # --- Teacher vs Student, FP32 vs INT8: reliability transition analysis. ---
    transitions = {}
    comparison_pairs = [("student_fp32", "student_int8"), ("teacher", "student_fp32")]
    for a, b in comparison_pairs:
        if a in reliable_by_variant and b in reliable_by_variant:
            # Align by image path since both variants iterate the same test loader deterministically.
            paths_a = [r["image_path"] for r in predictions_rows if r["model_variant"] == a]
            paths_b = [r["image_path"] for r in predictions_rows if r["model_variant"] == b]
            if paths_a == paths_b:
                key = f"{a}_vs_{b}"
                result = reliability_transition_analysis(reliable_by_variant[a].tolist(), reliable_by_variant[b].tolist())
                transitions[key] = result
                plot_reliability_transition_matrix(result["counts"], resolve_path(f"outputs/plots/reliability_transition_{key}.png"))
            else:
                logger.warning(f"Could not align {a} and {b} predictions by image order for transition analysis.")

    if len(variants) >= 2:
        names = list(variants.keys())
        sizes = [all_results[n]["model_size_mb"] for n in names]
        latencies = [all_results[n]["latency_ms"] or 0.0 for n in names]
        plot_model_size_vs_latency(names, sizes, latencies, resolve_path("outputs/plots/model_size_vs_latency.png"))

    ablation_table = build_ablation_table({
        name: {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        for name, metrics in all_results.items()
    })

    save_json({"per_model": all_results, "dess_by_class": dess_by_class, "reliability_transitions": transitions,
               "tau": tau, "num_test_images": len(test_ds)}, "outputs/metrics/evaluation_full.json")
    ablation_table.to_csv(resolve_path("outputs/metrics/evaluation_summary.csv"), index=False)
    pd.DataFrame(predictions_rows).to_csv(resolve_path("outputs/predictions/test_predictions.csv"), index=False)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY (test split)")
    print("=" * 70)
    print(ablation_table.to_string(index=False))
    print("\nSaved: outputs/metrics/evaluation_full.json, outputs/metrics/evaluation_summary.csv, "
          "outputs/predictions/test_predictions.csv")
    print("Next step:\n  python run_pipeline.py --export-dataset")


if __name__ == "__main__":
    main()

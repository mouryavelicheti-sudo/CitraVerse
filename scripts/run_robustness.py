#!/usr/bin/env python3
"""
MODULE III/IV robustness experiment:

    Original image x
        |-> T_e(x): brightness / shadow / noise / blur / contrast
        |-> Evidence extraction (Module II) on both x and T_e(x)
        |-> PCS (perturbation consistency score, src/stability/pcs.py)
        |-> Prediction consistency (does the class prediction survive T_e(x)?)
        |-> Reliability change R(x) vs R(T_e(x))

Runs on a sample of the TEST split (never train/val, to avoid reusing data
the model or its prototypes were fit on). Uses the already-fitted Module II
fusion statistics (outputs/features/fusion_stats.pkl) so perturbed images
are normalized identically to how the original evidence was normalized.

Saves:
    outputs/metrics/robustness_report.json
    outputs/metrics/robustness_per_image.csv
    outputs/plots/robustness_pcs_by_perturbation.png

Run this BEFORE scripts/evaluate.py: evaluate.py looks for this script's
outputs/metrics/robustness_per_image.csv and, if present, uses the real
measured PCS as E(x) in the reliability score. If you skip this step,
evaluate.py still runs, but E(x) is treated as neutral and R(x) is
renormalized over C(x)/A(x) only -- clearly logged, never silently faked.

Usage:
    python run_pipeline.py --robustness
    python scripts/run_robustness.py --num-samples 100
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evidence.color_evidence import extract_color_evidence
from src.evidence.deep_evidence import DeepEvidenceExtractor
from src.evidence.fusion import EvidenceComponents, fuse
from src.evidence.shape_evidence import extract_shape_evidence
from src.evidence.texture_evidence import extract_texture_evidence
from src.prototypes.prototype_builder import load_prototypes
from src.prototypes.similarity import prototype_agreement
from src.reliability.reliability_score import ReliabilityWeights, reliability_score
from src.stability.pcs import perturbation_consistency_score
from src.stability.perturbations import apply_all_perturbations
from src.utils.io_utils import load_yaml, resolve_path, save_json
from src.utils.logging_utils import get_logger
from src.utils.seed import get_device, set_seed

logger = get_logger("run_robustness")


def _evidence_for_image(img_bgr, ev_cfg, deep_extractor, fusion_stats, tfm, normalize):
    color_vec = extract_color_evidence(img_bgr, hist_bins=ev_cfg.get("color_bins", 32))
    texture_vec = extract_texture_evidence(
        img_bgr, lbp_points=ev_cfg.get("lbp_points", 24), lbp_radius=ev_cfg.get("lbp_radius", 3),
        glcm_distances=ev_cfg.get("glcm_distances", [1, 2]), glcm_angles_deg=ev_cfg.get("glcm_angles_deg", [0, 45, 90, 135]),
    )
    shape_vec = extract_shape_evidence(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    deep_vec = deep_extractor.extract_single(tfm(img_rgb))
    comp = EvidenceComponents(color=color_vec, texture=texture_vec, shape=shape_vec, deep=deep_vec)
    return fuse(comp, fusion_stats, normalize=normalize)


def main():
    try:
        import torch
        from torchvision import transforms
    except ImportError:
        logger.error("PyTorch is required for the robustness experiment. Install with: pip install torch torchvision")
        return

    from src.models.classifier import DiseaseClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="outputs/dataset_profiles/harmonized_metadata.csv")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--edge-config", default="configs/edge.yaml")
    parser.add_argument("--reliability-config", default="configs/reliability.yaml")
    parser.add_argument("--class-mapping-config", default="configs/class_mapping.yaml")
    parser.add_argument("--fusion-stats", default="outputs/features/fusion_stats.pkl")
    parser.add_argument("--prototypes", default="outputs/prototypes/prototypes.pkl")
    parser.add_argument("--student-checkpoint", default="models/student_best.pt")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    cfg = load_yaml(args.training_config)
    edge_cfg = load_yaml(args.edge_config)
    rel_cfg = load_yaml(args.reliability_config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    metadata_path = resolve_path(args.metadata)
    fusion_stats_path = resolve_path(args.fusion_stats)
    prototypes_path = resolve_path(args.prototypes)
    student_ckpt_path = resolve_path(args.student_checkpoint)

    for required, hint in [
        (metadata_path, "Run scripts/harmonize_datasets.py first."),
        (fusion_stats_path, "Run scripts/extract_evidence.py first."),
        (prototypes_path, "Run scripts/build_prototypes.py first."),
        (student_ckpt_path, "Run scripts/train_student.py first."),
    ]:
        if not os.path.exists(required):
            logger.error(f"{required} not found. {hint}")
            return

    df = pd.read_csv(metadata_path)
    test_df = df[df["split"] == "test"]
    if len(test_df) == 0:
        logger.error("No test-split rows found.")
        return
    sample_df = test_df.sample(n=min(args.num_samples, len(test_df)), random_state=42).reset_index(drop=True)

    canonical_classes = load_yaml(args.class_mapping_config).get("canonical_classes", [])
    class_to_idx = {c: i for i, c in enumerate(canonical_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    image_size = cfg.get("image_size", 224)
    ev_cfg = cfg.get("evidence", {})
    normalize = ev_cfg.get("fusion_normalize", "l2")

    with open(fusion_stats_path, "rb") as f:
        fusion_stats = pickle.load(f)
    prototypes = load_prototypes(prototypes_path)

    deep_extractor = DeepEvidenceExtractor(backbone=ev_cfg.get("deep_backbone", "resnet18"),
                                            device=str(device), pretrained=True)
    deep_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    classifier_tfm = deep_tfm  # same normalization used for the classifier input

    student_backbone = cfg.get("student", {}).get("backbone", "mobilenet_v3_small")
    student = DiseaseClassifier.build(student_backbone, len(canonical_classes), pretrained=False).to(device)
    student.load_state_dict(torch.load(student_ckpt_path, map_location=device))
    student.eval()

    weights = ReliabilityWeights(
        w1_confidence=rel_cfg["weights"]["w1_confidence"],
        w2_prototype_agreement=rel_cfg["weights"]["w2_prototype_agreement"],
        w3_perturbation_consistency=rel_cfg["weights"]["w3_perturbation_consistency"],
    )
    dess_metric = rel_cfg.get("dess", {}).get("similarity_metric", "cosine")
    perturb_cfg = edge_cfg.get("perturbations", {})

    per_image_rows = []
    pcs_by_perturbation = {}

    for _, row in sample_df.iterrows():
        img_bgr = cv2.imread(row["image_path"])
        if img_bgr is None:
            logger.warning(f"Could not read {row['image_path']}, skipping.")
            continue
        img_bgr = cv2.resize(img_bgr, (image_size, image_size))

        z_orig = _evidence_for_image(img_bgr, ev_cfg, deep_extractor, fusion_stats, deep_tfm, normalize)

        with torch.no_grad():
            tensor = classifier_tfm(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            logits, probs, _feat = student(tensor)
            conf_orig = float(probs.max().item())
            pred_orig = idx_to_class[int(probs.argmax(1).item())]

        agreement_orig = prototype_agreement(z_orig, pred_orig, prototypes, metric=dess_metric)

        perturbed_images = apply_all_perturbations(img_bgr, perturb_cfg, seed=42)
        z_perturbed = {}
        preds_perturbed = {}
        confs_perturbed = {}
        for name, perturbed_bgr in perturbed_images.items():
            z_p = _evidence_for_image(perturbed_bgr, ev_cfg, deep_extractor, fusion_stats, deep_tfm, normalize)
            z_perturbed[name] = z_p
            with torch.no_grad():
                tensor_p = classifier_tfm(cv2.cvtColor(perturbed_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                logits_p, probs_p, _feat_p = student(tensor_p)
                confs_perturbed[name] = float(probs_p.max().item())
                preds_perturbed[name] = idx_to_class[int(probs_p.argmax(1).item())]

        pcs_result = perturbation_consistency_score(z_orig, z_perturbed,
                                                      aggregate=rel_cfg.get("pcs", {}).get("aggregate_over_perturbations", "mean"))

        # Reliability BEFORE (on original image) using the actual PCS(x) as E(x).
        r_before = reliability_score(conf_orig, agreement_orig, pcs_result["PCS"], weights)

        for name in perturbed_images:
            agreement_p = prototype_agreement(z_perturbed[name], preds_perturbed[name], prototypes, metric=dess_metric)
            # Per-perturbation PCS uses that single perturbation's L_ES only.
            single_pcs = 1.0 / (1.0 + pcs_result["per_perturbation_L_ES"][name])
            r_after = reliability_score(confs_perturbed[name], agreement_p, single_pcs, weights)

            per_image_rows.append({
                "image_path": row["image_path"],
                "ground_truth_class": row["canonical_class"],
                "perturbation": name,
                "prediction_original": pred_orig,
                "prediction_perturbed": preds_perturbed[name],
                "prediction_flipped": pred_orig != preds_perturbed[name],
                "confidence_original": conf_orig,
                "confidence_perturbed": confs_perturbed[name],
                "L_ES": pcs_result["per_perturbation_L_ES"][name],
                "PCS": single_pcs,
                "reliability_before": r_before,
                "reliability_after": r_after,
                "reliability_delta": r_after - r_before,
            })
            pcs_by_perturbation.setdefault(name, []).append(single_pcs)

    per_image_df = pd.DataFrame(per_image_rows)
    out_csv = resolve_path("outputs/metrics/robustness_per_image.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    per_image_df.to_csv(out_csv, index=False)

    summary = {
        "num_images_tested": int(sample_df.shape[0]),
        "num_perturbation_records": int(len(per_image_df)),
        "prediction_flip_rate": float(per_image_df["prediction_flipped"].mean()) if len(per_image_df) else None,
        "mean_PCS_by_perturbation": {k: float(np.mean(v)) for k, v in pcs_by_perturbation.items()},
        "mean_reliability_delta": float(per_image_df["reliability_delta"].mean()) if len(per_image_df) else None,
        "mean_reliability_delta_by_perturbation": (
            per_image_df.groupby("perturbation")["reliability_delta"].mean().to_dict() if len(per_image_df) else {}
        ),
    }
    save_json(summary, "outputs/metrics/robustness_report.json")

    try:
        from src.evaluation.plots import plot_grouped_metric_comparison
        perturb_names = list(summary["mean_PCS_by_perturbation"].keys())
        plot_grouped_metric_comparison(
            perturb_names, {"Mean PCS": [summary["mean_PCS_by_perturbation"][n] for n in perturb_names]},
            ylabel="PCS", title="Perturbation Consistency Score by Perturbation Type",
            out_path=resolve_path("outputs/plots/robustness_pcs_by_perturbation.png"),
        )
    except Exception as e:  # pragma: no cover - plotting must never crash the experiment
        logger.warning(f"Could not generate robustness plot: {e}")

    print("\n" + "=" * 70)
    print("ROBUSTNESS EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Images tested:        {summary['num_images_tested']}")
    print(f"  Prediction flip rate: {summary['prediction_flip_rate']:.4f}" if summary["prediction_flip_rate"] is not None else "  Prediction flip rate: N/A")
    print(f"  Mean reliability shift under perturbation: {summary['mean_reliability_delta']:.4f}" if summary["mean_reliability_delta"] is not None else "")
    print(f"\nSaved: {out_csv}, outputs/metrics/robustness_report.json")
    print("Next step:\n  python run_pipeline.py --explain")


if __name__ == "__main__":
    main()

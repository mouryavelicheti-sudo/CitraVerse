"""End-to-end smoke test: Modules I -> II -> III on a tiny synthetic
dataset, exercising the same real functions the CLI scripts call, without
touching the real local datasets or writing to the project's outputs/
folder. This is the "does the whole pipeline actually connect" check
called for in BUILD_STATUS.md.

Deep semantic evidence Ed(x) is skipped here (torch/torchvision are heavy
for a smoke test and are already covered by test_distillation_losses.py /
test_quantization.py) -- color/texture/shape evidence and the full
fusion -> prototypes -> DESS -> PCS -> reliability chain are exercised with
real data end-to-end.
"""
import cv2
import numpy as np
import pandas as pd

from src.data.dataset_scanner import scan_dataset
from src.evidence.color_evidence import extract_color_evidence
from src.evidence.fusion import EvidenceComponents, fit_fusion_stats, fuse
from src.evidence.shape_evidence import extract_shape_evidence
from src.evidence.texture_evidence import extract_texture_evidence
from src.harmonization.class_mapper import harmonize
from src.preprocessing.splitter import stratified_split
from src.prototypes.prototype_builder import build_prototypes
from src.prototypes.similarity import compute_dess, prototype_agreement
from src.reliability.reliability_score import ReliabilityWeights, reliability_decision, reliability_score
from src.stability.perturbations import apply_all_perturbations
from src.stability.pcs import perturbation_consistency_score


def test_full_pipeline_modules_i_through_iv_on_synthetic_data(synthetic_dataset_root):
    # ---- Module I: scan + harmonize + split ----
    all_records = []
    records_by_dataset_class = {}
    for dataset_name in ["DatasetA", "DatasetB"]:
        _profile, records = scan_dataset(dataset_name, f"{synthetic_dataset_root}/{dataset_name}")
        all_records.extend(records)
        for r in records:
            records_by_dataset_class[(r.dataset, r.original_class)] = (
                records_by_dataset_class.get((r.dataset, r.original_class), 0) + 1
            )

    mapping_config = {
        "canonical_classes": ["healthy", "canker"],
        "mapping": {
            "DatasetA": {"leaf_healthy": "healthy", "leaf_canker": "canker"},
            "DatasetB": {"healthy_leaf": "healthy", "canker_spot": "canker"},
        },
        "excluded_classes": [],
    }
    harmonization_result = harmonize(records_by_dataset_class, mapping_config)
    assert harmonization_result.unmapped == [], "All synthetic classes should map cleanly."

    canonical_by_pair = {
        (m["dataset"], m["original_class"]): m["canonical_class"] for m in harmonization_result.mapped
    }
    rows = [{
        "image_path": r.path, "dataset": r.dataset, "original_class": r.original_class,
        "canonical_class": canonical_by_pair[(r.dataset, r.original_class)], "augmentation_type": "none",
    } for r in all_records]
    df = pd.DataFrame(rows)
    assert len(df) == 22  # 12 (DatasetA) + 10 (DatasetB), per conftest fixture

    split_df = stratified_split(df, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=0)
    assert set(split_df["split"]) <= {"train", "val", "test"}
    for split_name in ("train", "val", "test"):
        assert (split_df["split"] == split_name).sum() > 0

    # ---- Module II: extract handcrafted evidence + fuse (deep evidence mocked as zeros) ----
    deep_dim = 16
    color_list, texture_list, shape_list, deep_list = [], [], [], []
    for path in split_df["image_path"]:
        img_bgr = cv2.imread(path)
        img_bgr = cv2.resize(img_bgr, (64, 64))
        color_list.append(extract_color_evidence(img_bgr))
        texture_list.append(extract_texture_evidence(img_bgr))
        shape_list.append(extract_shape_evidence(img_bgr))
        deep_list.append(np.zeros(deep_dim, dtype=np.float32))

    train_mask = (split_df["split"] == "train").values
    train_components = [
        EvidenceComponents(color=color_list[i], texture=texture_list[i], shape=shape_list[i], deep=deep_list[i])
        for i in range(len(split_df)) if train_mask[i]
    ]
    fusion_stats = fit_fusion_stats(train_components)

    z_signatures = [
        fuse(EvidenceComponents(color=color_list[i], texture=texture_list[i], shape=shape_list[i], deep=deep_list[i]),
             fusion_stats, normalize="l2")
        for i in range(len(split_df))
    ]
    assert all(np.all(np.isfinite(z)) for z in z_signatures)

    # ---- Module III: prototypes + DESS ----
    train_z = [z_signatures[i] for i in range(len(split_df)) if train_mask[i]]
    train_classes = split_df.loc[train_mask, "canonical_class"].tolist()
    train_domains = split_df.loc[train_mask, "dataset"].tolist()
    prototypes = build_prototypes(train_z, train_classes, train_domains)
    dess = compute_dess(prototypes)
    # Both classes appear in both DatasetA and DatasetB -> DESS should be defined for both.
    assert dess["healthy"] is not None
    assert dess["canker"] is not None

    # ---- Module III continued: perturbation robustness -> PCS ----
    sample_path = split_df.iloc[0]["image_path"]
    sample_img = cv2.resize(cv2.imread(sample_path), (64, 64))
    perturb_cfg = {
        "brightness_delta": [-0.2, 0.2], "contrast_factor": [0.8, 1.2],
        "gaussian_noise_std": 0.02, "gaussian_blur_kernel": 3, "shadow_strength": 0.3,
    }
    perturbed_images = apply_all_perturbations(sample_img, perturb_cfg, seed=0)
    assert len(perturbed_images) > 0

    z_orig = fuse(EvidenceComponents(
        color=extract_color_evidence(sample_img), texture=extract_texture_evidence(sample_img),
        shape=extract_shape_evidence(sample_img), deep=np.zeros(deep_dim, dtype=np.float32),
    ), fusion_stats, normalize="l2")

    z_perturbed = {}
    for name, perturbed_bgr in perturbed_images.items():
        z_perturbed[name] = fuse(EvidenceComponents(
            color=extract_color_evidence(perturbed_bgr), texture=extract_texture_evidence(perturbed_bgr),
            shape=extract_shape_evidence(perturbed_bgr), deep=np.zeros(deep_dim, dtype=np.float32),
        ), fusion_stats, normalize="l2")

    pcs_result = perturbation_consistency_score(z_orig, z_perturbed)
    assert 0.0 < pcs_result["PCS"] <= 1.0

    # ---- Module IV: reliability score end-to-end ----
    predicted_class = split_df.iloc[0]["canonical_class"]
    agreement = prototype_agreement(z_orig, predicted_class, prototypes)
    weights = ReliabilityWeights(0.4, 0.35, 0.25)
    r = reliability_score(confidence=0.9, prototype_agreement=agreement,
                           perturbation_consistency=pcs_result["PCS"], weights=weights)
    decision = reliability_decision(r, tau=0.6)

    assert 0.0 <= r <= 1.0
    assert decision in ("RELIABLE", "UNCERTAIN_EXPERT_REVIEW")

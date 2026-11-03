#!/usr/bin/env python3
"""
MODULE II: Multi-Evidence Extraction + Fusion.

Reads outputs/dataset_profiles/harmonized_metadata.csv, extracts
Color/Texture/Shape/Deep evidence for every image, fits fusion
normalization statistics on TRAIN only, and writes:
    outputs/features/evidence_components.npz   (raw per-component vectors)
    outputs/features/fusion_stats.yaml
    outputs/features/Z_signatures.npz          (fused Z(x) for every image)

Usage:
    python run_pipeline.py --extract-evidence
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evidence.color_evidence import extract_color_evidence
from src.evidence.deep_evidence import DeepEvidenceExtractor
from src.evidence.fusion import EvidenceComponents, fit_fusion_stats, fuse
from src.evidence.shape_evidence import extract_shape_evidence
from src.evidence.texture_evidence import extract_texture_evidence
from src.utils.io_utils import load_yaml, resolve_path
from src.utils.logging_utils import get_logger
from src.utils.seed import get_device, set_seed

logger = get_logger("extract_evidence")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="outputs/dataset_profiles/harmonized_metadata.csv")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--out-dir", default="outputs/features")
    args = parser.parse_args()

    cfg = load_yaml(args.training_config)
    set_seed(cfg.get("seed", 42))

    metadata_path = resolve_path(args.metadata)
    if not os.path.exists(metadata_path):
        logger.error(f"{metadata_path} not found. Run scripts/harmonize_datasets.py first.")
        return
    df = pd.read_csv(metadata_path)

    ev_cfg = cfg.get("evidence", {})
    image_size = cfg.get("image_size", 224)

    try:
        deep_extractor = DeepEvidenceExtractor(
            backbone=ev_cfg.get("deep_backbone", "resnet18"),
            device=str(get_device(cfg.get("device", "auto"))),
            pretrained=True,
        )
        torch_available = True
    except ImportError as e:
        logger.warning(str(e))
        logger.warning(
            "Deep semantic evidence Ed(x) requires torch/torchvision, which are "
            "not installed. Color/Texture/Shape evidence will still be extracted "
            "and saved. Install torch+torchvision and re-run this script to add "
            "Ed(x) and complete the fused signature Z(x)."
        )
        deep_extractor = None
        torch_available = False

    color_list, texture_list, shape_list, deep_list = [], [], [], []
    valid_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting evidence"):
        img_bgr = cv2.imread(row["image_path"])
        if img_bgr is None:
            logger.warning(f"Could not read image, skipping: {row['image_path']}")
            continue
        img_bgr = cv2.resize(img_bgr, (image_size, image_size))

        color_vec = extract_color_evidence(img_bgr, hist_bins=ev_cfg.get("color_bins", 32))
        texture_vec = extract_texture_evidence(
            img_bgr,
            lbp_points=ev_cfg.get("lbp_points", 24),
            lbp_radius=ev_cfg.get("lbp_radius", 3),
            glcm_distances=ev_cfg.get("glcm_distances", [1, 2]),
            glcm_angles_deg=ev_cfg.get("glcm_angles_deg", [0, 45, 90, 135]),
        )
        shape_vec = extract_shape_evidence(img_bgr)

        if torch_available:
            import torch
            from torchvision import transforms

            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = tfm(img_rgb)
            deep_vec = deep_extractor.extract_single(tensor)
        else:
            deep_vec = np.zeros(ev_cfg.get("deep_feature_dim", 512), dtype=np.float32)

        color_list.append(color_vec)
        texture_list.append(texture_vec)
        shape_list.append(shape_vec)
        deep_list.append(deep_vec)
        valid_rows.append(row)

    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    out_dir = resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(out_dir, "evidence_components.npz"),
        color=np.stack(color_list), texture=np.stack(texture_list),
        shape=np.stack(shape_list), deep=np.stack(deep_list),
    )
    valid_df.to_csv(os.path.join(out_dir, "evidence_index.csv"), index=False)

    # Fit fusion stats on TRAIN split ONLY.
    train_mask = (valid_df["split"] == "train").values
    train_components = [
        EvidenceComponents(color=color_list[i], texture=texture_list[i], shape=shape_list[i], deep=deep_list[i])
        for i in range(len(valid_df)) if train_mask[i]
    ]
    if not train_components:
        logger.error("No TRAIN-split samples found -- cannot fit fusion statistics.")
        return

    fusion_stats = fit_fusion_stats(train_components)
    with open(os.path.join(out_dir, "fusion_stats.pkl"), "wb") as f:
        pickle.dump(fusion_stats, f)

    z_signatures = []
    for i in range(len(valid_df)):
        comp = EvidenceComponents(color=color_list[i], texture=texture_list[i], shape=shape_list[i], deep=deep_list[i])
        z = fuse(comp, fusion_stats, normalize=ev_cfg.get("fusion_normalize", "l2"))
        z_signatures.append(z)

    np.savez_compressed(os.path.join(out_dir, "Z_signatures.npz"), Z=np.stack(z_signatures))

    logger.info(f"Deep evidence available: {torch_available}")
    logger.info(f"Saved evidence for {len(valid_df)} images to {out_dir}")
    print("\nNext step:\n  python run_pipeline.py --build-prototypes")


if __name__ == "__main__":
    main()

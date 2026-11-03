#!/usr/bin/env python3
"""
MODULE III: build P(c,d) prototypes from TRAIN evidence and compute DESS.

Usage:
    python run_pipeline.py --build-prototypes
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.plots import plot_dess_comparison
from src.prototypes.prototype_builder import build_prototypes, save_prototypes
from src.prototypes.similarity import compute_dess
from src.utils.io_utils import load_yaml, resolve_path, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("build_prototypes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-index", default="outputs/features/evidence_index.csv")
    parser.add_argument("--z-signatures", default="outputs/features/Z_signatures.npz")
    parser.add_argument("--reliability-config", default="configs/reliability.yaml")
    parser.add_argument("--out", default="outputs/prototypes/prototypes.pkl")
    args = parser.parse_args()

    idx_path = resolve_path(args.evidence_index)
    z_path = resolve_path(args.z_signatures)
    if not (os.path.exists(idx_path) and os.path.exists(z_path)):
        logger.error("Evidence files not found. Run scripts/extract_evidence.py first.")
        return

    df = pd.read_csv(idx_path)
    z = np.load(z_path)["Z"]
    assert len(df) == len(z), "evidence_index.csv and Z_signatures.npz are out of sync"

    train_mask = (df["split"] == "train").values
    train_z = [z[i] for i in range(len(df)) if train_mask[i]]
    train_classes = df.loc[train_mask, "canonical_class"].tolist()
    train_domains = df.loc[train_mask, "dataset"].tolist()

    prototypes = build_prototypes(train_z, train_classes, train_domains)

    out_path = resolve_path(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_prototypes(prototypes, out_path)

    rel_cfg = load_yaml(args.reliability_config).get("dess", {})
    dess = compute_dess(prototypes, metric=rel_cfg.get("similarity_metric", "cosine"),
                         aggregate=rel_cfg.get("aggregate_across_domains", "mean"))

    save_json(dess, "outputs/metrics/dess_by_class.json")
    plot_dess_comparison(dess, "outputs/plots/dess_comparison.png")

    print("\n" + "=" * 70)
    print("DISEASE EVIDENCE STABILITY SCORE (DESS) BY CLASS")
    print("=" * 70)
    for c, score in dess.items():
        if score is None:
            print(f"  {c}: UNDEFINED (present in fewer than 2 domains)")
        else:
            print(f"  {c}: {score:.4f}")

    print(f"\nPrototypes saved to: {out_path}")
    print("Next step:\n  python run_pipeline.py --train-teacher")


if __name__ == "__main__":
    main()

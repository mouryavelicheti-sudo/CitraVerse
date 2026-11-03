#!/usr/bin/env python3
"""
MODULE I: Harmonization + Deduplication + Stratified Split.

Requires configs/class_mapping.yaml to already contain reviewed mappings
(see scripts/profile_datasets.py output). Produces:
    outputs/dataset_profiles/harmonized_metadata.csv
    outputs/dataset_profiles/unmapped_classes_report.json  (if any)

Usage:
    python run_pipeline.py --preprocess
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset_scanner import scan_dataset
from src.harmonization.class_mapper import harmonize
from src.preprocessing.dedup import build_hash_index, find_near_duplicate_groups
from src.preprocessing.splitter import stratified_split
from src.provenance.provenance_tracker import ProvenanceTracker
from src.utils.io_utils import load_yaml, resolve_path, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("harmonize_datasets")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--class-mapping-config", default="configs/class_mapping.yaml")
    parser.add_argument("--out", default="outputs/dataset_profiles/harmonized_metadata.csv")
    args = parser.parse_args()

    ds_config = load_yaml(args.config)
    mapping_config = load_yaml(args.class_mapping_config)

    tracker = ProvenanceTracker()
    records_by_dataset_class = {}
    all_records = []

    for name, info in ds_config["datasets"].items():
        path = resolve_path(info["path"])
        if not os.path.isdir(path):
            logger.warning(f"Skipping missing dataset: {name} ({path})")
            continue
        _profile, records = scan_dataset(
            name, path,
            tuple(ds_config.get("valid_extensions", [".jpg", ".jpeg", ".png"])),
            layout_override=info.get("layout"),
            class_root_subpath=info.get("class_root_subpath"),
        )
        for r in records:
            if r.corrupt:
                continue
            records_by_dataset_class[(r.dataset, r.original_class)] = records_by_dataset_class.get(
                (r.dataset, r.original_class), 0
            ) + 1
            all_records.append(r)

    if not all_records:
        logger.error("No usable images found across any configured dataset. Aborting.")
        return

    result = harmonize(records_by_dataset_class, mapping_config)

    if result.unmapped:
        logger.warning(
            f"{len(result.unmapped)} (dataset, class) pairs are UNMAPPED and will be "
            "EXCLUDED from the harmonized dataset. See "
            "outputs/dataset_profiles/unmapped_classes_report.json"
        )
        save_json(result.unmapped, "outputs/dataset_profiles/unmapped_classes_report.json")
        for u in result.unmapped:
            logger.warning(f"  UNMAPPED: {u['dataset']}:{u['original_class']} ({u['count']} images) - {u['reason']}")

    mapped_lookup = {(m["dataset"], m["original_class"]): m["canonical_class"] for m in result.mapped}

    kept_records = [r for r in all_records if (r.dataset, r.original_class) in mapped_lookup]
    logger.info(f"{len(kept_records)}/{len(all_records)} images have a valid canonical class mapping.")

    if not kept_records:
        logger.error(
            "Zero images were successfully mapped to a canonical class. "
            "Edit configs/class_mapping.yaml (see the suggested version in "
            "outputs/dataset_profiles/class_mapping_suggested.yaml) and re-run."
        )
        return

    for r in kept_records:
        canonical = mapped_lookup[(r.dataset, r.original_class)]
        tracker.add_real_image(
            r.path, r.dataset, r.original_class, canonical,
            group=r.group, relative_path=r.relative_path,
        )

    df = tracker.to_dataframe()

    # --- Near-duplicate detection (across the WHOLE pool, dataset-agnostic,
    # since the same physical leaf image sometimes ends up reused across
    # public datasets) ---
    dedup_cfg = ds_config.get("dedup", {})
    logger.info("Computing perceptual hashes for near-duplicate detection (this can take a while)...")
    hash_index = build_hash_index(df["image_path"].tolist(), hash_size=dedup_cfg.get("hash_size", 8))
    dup_groups = find_near_duplicate_groups(hash_index, dedup_cfg.get("near_duplicate_hamming_threshold", 5))
    logger.info(f"Found {len(dup_groups)} near-duplicate groups.")
    save_json(
        [{"group_size": len(g), "paths": g} for g in dup_groups],
        "outputs/dataset_profiles/near_duplicate_groups.json",
    )

    # --- Stratified split (near-duplicate groups kept together) ---
    split_cfg = ds_config.get("split", {})
    df_split = stratified_split(
        df,
        stratify_col=split_cfg.get("stratify_by", "canonical_class"),
        train_frac=split_cfg.get("train", 0.70),
        val_frac=split_cfg.get("val", 0.15),
        test_frac=split_cfg.get("test", 0.15),
        seed=split_cfg.get("seed", 42),
        duplicate_groups=dup_groups,
    )

    out_path = resolve_path(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_split.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("HARMONIZATION SUMMARY")
    print("=" * 70)
    print(df_split.groupby(["dataset", "canonical_class"]).size().unstack(fill_value=0))
    print("\nSplit sizes:")
    print(df_split["split"].value_counts())
    print(f"\nHarmonized metadata written to: {out_path}")
    print("Next step:\n  python run_pipeline.py --extract-evidence")


if __name__ == "__main__":
    main()

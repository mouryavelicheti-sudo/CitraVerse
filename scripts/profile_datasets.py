#!/usr/bin/env python3
"""
PHASE 1 / MODULE I: Dataset inspection & profiling.

Run this FIRST, before anything else. It never assumes dataset structure --
it inspects the folders on disk and reports exactly what it finds, including
a proposed (NOT auto-applied) canonical class mapping suggestion.

Usage:
    python run_pipeline.py --profile
    python scripts/profile_datasets.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset_scanner import (
    find_exact_duplicates,
    scan_dataset,
    summarize_resolution_stats,
)
from src.evaluation.plots import plot_class_distribution, plot_dataset_size_comparison, plot_resolution_distribution
from src.harmonization.class_mapper import suggest_mapping
from src.utils.io_utils import load_yaml, resolve_path, save_json, save_yaml
from src.utils.logging_utils import get_logger

logger = get_logger("profile_datasets")


def main():
    parser = argparse.ArgumentParser(description="Inspect and profile AgriVerse datasets.")
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--class-mapping-config", default="configs/class_mapping.yaml")
    args = parser.parse_args()

    ds_config = load_yaml(args.config)
    mapping_config = load_yaml(args.class_mapping_config)

    all_profiles = {}
    all_records = {}
    records_by_dataset_class = {}

    print("=" * 70)
    print("DATASET INSPECTION REPORT")
    print("=" * 70)

    for name, info in ds_config["datasets"].items():
        path = resolve_path(info["path"])
        if not os.path.isdir(path):
            logger.warning(f"Dataset path does not exist yet: {path} (skipping '{name}')")
            print(f"\n{name}\n  [NOT FOUND] Expected at: {path}")
            print("  -> Place your dataset here, or update configs/datasets.yaml, then re-run.")
            continue

        profile, records = scan_dataset(
            name, path,
            tuple(ds_config.get("valid_extensions", [".jpg", ".jpeg", ".png"])),
            layout_override=info.get("layout"),
            class_root_subpath=info.get("class_root_subpath"),
        )
        all_profiles[name] = profile
        all_records[name] = records

        for cls, count in profile.class_counts.items():
            records_by_dataset_class[(name, cls)] = count

        res_stats = summarize_resolution_stats(profile.resolutions)
        exact_dupes = find_exact_duplicates(records)

        print(f"\n{name}")
        print(f"  Detected layout : {profile.layout_detected}")
        print(f"  Images          : {profile.num_images}")
        print(f"  Classes         : {profile.num_classes} -> {sorted(profile.class_counts.keys())}")
        print(f"  Per-class counts: {profile.class_counts}")
        if profile.group_counts:
            print(f"  Groups/subsets  : {sorted(profile.group_counts.keys())}")
            for g, counts in profile.group_counts.items():
                print(f"    {g}: {counts}")
        print(f"  Class imbalance ratio (max/min): {profile.class_imbalance_ratio}")
        print(f"  File formats    : {profile.formats}")
        if res_stats:
            print(f"  Resolution      : {res_stats}")
        print(f"  Corrupt images  : {len(profile.corrupt_images)}")
        print(f"  Exact duplicates: {len(exact_dupes)}")

        if profile.resolutions:
            widths = [w for w, h in profile.resolutions]
            heights = [h for w, h in profile.resolutions]
            plot_resolution_distribution(widths, heights, f"outputs/plots/{name}_resolution_distribution.png")
        if profile.class_counts:
            plot_class_distribution(profile.class_counts, f"{name} Class Distribution",
                                     f"outputs/plots/{name}_class_distribution.png")

        save_json({
            "name": profile.name,
            "layout_detected": profile.layout_detected,
            "num_images": profile.num_images,
            "num_classes": profile.num_classes,
            "class_counts": profile.class_counts,
            "group_counts": profile.group_counts,
            "class_imbalance_ratio": profile.class_imbalance_ratio,
            "formats": profile.formats,
            "resolution_stats": res_stats,
            "num_corrupt": len(profile.corrupt_images),
            "corrupt_images": profile.corrupt_images[:50],  # cap for readability
            "num_exact_duplicates": len(exact_dupes),
        }, f"outputs/dataset_profiles/{name}_profile.json")

    if not all_profiles:
        print("\nNo datasets found. Nothing further to report. Update configs/datasets.yaml and re-run.")
        return

    # Cross-dataset comparison plot
    plot_dataset_size_comparison({n: p.num_images for n, p in all_profiles.items()},
                                  "outputs/plots/dataset_size_comparison.png")

    # Canonical taxonomy suggestion (SUGGESTION ONLY -- never auto-applied)
    canonical_classes = mapping_config.get("canonical_classes", [])
    suggestions = suggest_mapping(records_by_dataset_class, canonical_classes)

    suggested_yaml = {
        "canonical_classes": canonical_classes,
        "mapping": suggestions,
        "excluded_classes": [],
        "_NOTE": (
            "AUTO-GENERATED SUGGESTIONS via fuzzy string matching. Review every "
            "entry manually before copying into configs/class_mapping.yaml. "
            "`null` means no confident match was found -- these need a human "
            "decision, not a guess."
        ),
    }
    save_yaml(suggested_yaml, "outputs/dataset_profiles/class_mapping_suggested.yaml")

    print("\n" + "=" * 70)
    print("PROPOSED CANONICAL TAXONOMY (suggestions only -- review before use)")
    print("=" * 70)
    for dataset, cls_map in suggestions.items():
        print(f"\n{dataset}:")
        for original, canonical in cls_map.items():
            arrow = f"-> {canonical}" if canonical else "-> [UNMAPPED: needs manual review]"
            print(f"  {original} {arrow}")

    print(
        "\nFull suggestion file written to: "
        "outputs/dataset_profiles/class_mapping_suggested.yaml\n"
        "Edit configs/class_mapping.yaml with your reviewed decisions, then run:\n"
        "  python run_pipeline.py --harmonize\n"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export the harmonized, deduplicated, split LOCAL datasets into a single
portable directory (and optionally a .zip), with full provenance metadata.

    Exported Dataset
    |-- train/<canonical_class>/*.jpg
    |-- val/<canonical_class>/*.jpg
    |-- test/<canonical_class>/*.jpg
    `-- metadata.csv   (image_path, dataset, original_class, canonical_class,
                         split, augmentation_type, source_image)

IMPORTANT: EXPORT != DOWNLOAD. This script only ever reads from your local
dataset folders (configs/datasets.yaml -> root) and copies files into a new
local export directory -- it never fetches anything from the network, and
it never includes images that scripts/harmonize_datasets.py excluded
(unmapped classes, exact/near-duplicates removed, etc.), since the export
is meant to be exactly what the model was actually trained/evaluated on.

Usage:
    python run_pipeline.py --export-dataset
    python scripts/export_dataset.py --zip
"""
import argparse
import os
import shutil
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io_utils import resolve_path
from src.utils.logging_utils import get_logger

logger = get_logger("export_dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="outputs/dataset_profiles/harmonized_metadata.csv")
    parser.add_argument("--out-dir", default="outputs/exports/agriverse_dataset")
    parser.add_argument("--zip", action="store_true", help="Also produce a .zip of the exported directory.")
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy",
                         help="'copy' duplicates image bytes (portable); 'symlink' saves disk space "
                              "but the export is only valid on the same machine/filesystem.")
    args = parser.parse_args()

    metadata_path = resolve_path(args.metadata)
    if not os.path.exists(metadata_path):
        logger.error(f"{metadata_path} not found. Run scripts/harmonize_datasets.py first.")
        return

    df = pd.read_csv(metadata_path)
    out_dir = resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    exported_rows = []
    missing = 0
    for _, row in df.iterrows():
        split = row.get("split")
        if pd.isna(split) or split not in ("train", "val", "test"):
            logger.warning(f"Row without a valid split, skipping: {row['image_path']}")
            continue

        src_path = row["image_path"]
        if not os.path.exists(src_path):
            missing += 1
            continue

        class_dir = os.path.join(out_dir, split, str(row["canonical_class"]))
        os.makedirs(class_dir, exist_ok=True)

        # Disambiguate filenames across datasets that may share a basename.
        dest_name = f"{row['dataset']}__{os.path.basename(src_path)}"
        dest_path = os.path.join(class_dir, dest_name)

        if args.copy_mode == "symlink":
            if not os.path.exists(dest_path):
                os.symlink(os.path.abspath(src_path), dest_path)
        else:
            shutil.copy2(src_path, dest_path)

        exported_rows.append({
            "image_path": os.path.relpath(dest_path, out_dir),
            "dataset": row["dataset"],
            "original_class": row["original_class"],
            "canonical_class": row["canonical_class"],
            "split": split,
            "augmentation_type": row.get("augmentation_type", "none"),
            "source_image": row.get("source_image", src_path),
        })

    if missing:
        logger.warning(f"{missing} row(s) referenced image files that no longer exist on disk and were skipped.")

    export_df = pd.DataFrame(exported_rows)
    export_df.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)

    split_counts = export_df["split"].value_counts().to_dict() if len(export_df) else {}
    class_counts = export_df["canonical_class"].value_counts().to_dict() if len(export_df) else {}

    print("\n" + "=" * 70)
    print("DATASET EXPORT SUMMARY")
    print("=" * 70)
    print(f"  Exported {len(export_df)} images to: {out_dir}")
    print(f"  Split sizes:  {split_counts}")
    print(f"  Class counts: {class_counts}")

    if args.zip:
        zip_path = out_dir.rstrip(os.sep) + ".zip"
        logger.info(f"Zipping export directory to {zip_path} ...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(out_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    zf.write(fpath, os.path.relpath(fpath, os.path.dirname(out_dir)))
        print(f"  Zipped to: {zip_path}")

    print("\nNext step:\n  python run_pipeline.py --all   (to re-run everything end-to-end)")


if __name__ == "__main__":
    main()

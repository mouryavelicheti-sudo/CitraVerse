"""
MODULE I - Dataset Variability Mapping (part 1/3): scanning & profiling.

This module inspects the ACTUAL dataset folders on disk. It never assumes
class names, image counts, or formats -- everything here is measured, not
guessed.
"""
import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, UnidentifiedImageError


@dataclass
class ImageRecord:
    path: str
    dataset: str
    original_class: str
    width: int
    height: int
    aspect_ratio: float
    file_format: str
    file_size_bytes: int
    corrupt: bool = False
    error: str = None
    group: str = None              # provenance/subset label, e.g. "Leaves"/"Fruits"/"Converted Image"
    relative_path: str = None      # path relative to the dataset's configured root, for provenance


@dataclass
class DatasetProfile:
    name: str
    root: str
    layout_detected: str
    num_images: int = 0
    num_classes: int = 0
    class_counts: dict = field(default_factory=dict)
    formats: dict = field(default_factory=dict)
    resolutions: list = field(default_factory=list)
    aspect_ratios: list = field(default_factory=list)
    corrupt_images: list = field(default_factory=list)
    class_imbalance_ratio: float = None  # max_class_count / min_class_count
    group_counts: dict = field(default_factory=dict)  # {group_name: {class_name: count}}, only if groups exist


VALID_EXTENSIONS_DEFAULT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def detect_layout(dataset_path: str) -> str:
    """Heuristically detect how a dataset folder is organized.

    Returns one of: "class_subfolders", "nested_splits", "flat", "empty".
    """
    if not os.path.isdir(dataset_path):
        return "empty"
    entries = [e for e in os.listdir(dataset_path) if not e.startswith(".")]
    subdirs = [e for e in entries if os.path.isdir(os.path.join(dataset_path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(dataset_path, e))]

    if not subdirs and files:
        return "flat"
    if not subdirs:
        return "empty"

    lower_subdirs = {s.lower() for s in subdirs}
    if {"train", "val", "test"} & lower_subdirs or {"train", "validation", "test"} & lower_subdirs:
        return "nested_splits"
    return "class_subfolders"


def _scan_class_subfolders(dataset_path: str, dataset_name: str, valid_ext,
                            group_label: str = None, provenance_root: str = None) -> list:
    """Layout A: <dataset_path>/<class_name>/<image>.

    `provenance_root` (if given) is the outer dataset root used to compute
    relative paths for provenance, even when `dataset_path` itself is a
    deeper sub-path (see layout C / `class_root_subpath`).
    """
    provenance_root = provenance_root or dataset_path
    records = []
    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_ext:
                continue
            fpath = os.path.join(class_dir, fname)
            rel = os.path.relpath(fpath, provenance_root)
            records.append(_inspect_image(fpath, dataset_name, class_name, group=group_label, relative_path=rel))
    return records


def _scan_group_class_subfolders(dataset_path: str, dataset_name: str, valid_ext) -> list:
    """Layout B: <dataset_path>/<group_name>/<class_name>/<image>.

    `group_name` (e.g. "Leaves"/"Fruits") is NOT a disease class -- it is
    kept purely as provenance/subset metadata on each ImageRecord. Only the
    innermost folder name is treated as the (original) disease class.
    """
    records = []
    for group_name in sorted(os.listdir(dataset_path)):
        group_dir = os.path.join(dataset_path, group_name)
        if not os.path.isdir(group_dir):
            continue
        for class_name in sorted(os.listdir(group_dir)):
            class_dir = os.path.join(group_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in valid_ext:
                    continue
                fpath = os.path.join(class_dir, fname)
                rel = os.path.relpath(fpath, dataset_path)
                records.append(_inspect_image(fpath, dataset_name, class_name, group=group_name, relative_path=rel))
    return records


def _scan_nested_splits(dataset_path: str, dataset_name: str, valid_ext) -> list:
    """For datasets already split into train/val/test, merge them back into
    a single pool of (image, original_class) records -- AgriVerse performs
    its own stratified split downstream, so we treat the union as the source
    pool and record provenance of which pre-existing split it came from."""
    records = []
    for split_name in os.listdir(dataset_path):
        split_dir = os.path.join(dataset_path, split_name)
        if not os.path.isdir(split_dir):
            continue
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in valid_ext:
                    continue
                fpath = os.path.join(class_dir, fname)
                rec = _inspect_image(fpath, dataset_name, class_name)
                records.append(rec)
    return records


def _inspect_image(fpath: str, dataset_name: str, class_name: str,
                    group: str = None, relative_path: str = None) -> ImageRecord:
    file_size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
    ext = os.path.splitext(fpath)[1].lower()
    try:
        with Image.open(fpath) as img:
            img.verify()  # cheap corruption check
        with Image.open(fpath) as img:  # re-open: verify() invalidates the file pointer
            w, h = img.size
            fmt = img.format
        return ImageRecord(
            path=fpath,
            dataset=dataset_name,
            original_class=class_name,
            width=w,
            height=h,
            aspect_ratio=round(w / h, 4) if h else 0.0,
            file_format=fmt,
            file_size_bytes=file_size,
            corrupt=False,
            group=group,
            relative_path=relative_path,
        )
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return ImageRecord(
            path=fpath,
            dataset=dataset_name,
            original_class=class_name,
            width=0,
            height=0,
            aspect_ratio=0.0,
            file_format=ext,
            file_size_bytes=file_size,
            corrupt=True,
            error=str(e),
            group=group,
            relative_path=relative_path,
        )


def scan_dataset(dataset_name: str, dataset_path: str, valid_ext=VALID_EXTENSIONS_DEFAULT,
                  layout_override: str = None, class_root_subpath: str = None):
    """Scan one dataset folder and return (DatasetProfile, list[ImageRecord]).

    `dataset_path` is always treated as the dataset's canonical root (used for
    reporting and for resolving relative/provenance paths), even when the
    actual class folders live deeper inside it.

    layout_override: explicit layout, always wins over heuristic detection.
        One of: "class_subfolders" (root/class/image), "group_class_subfolders"
        (root/group/class/image, e.g. Leaves|Fruits), "nested_splits", "flat".
        If None, falls back to `detect_layout()` heuristics (legacy behavior).
    class_root_subpath: optional relative path (from `dataset_path`) to the
        folder that actually contains the class subfolders -- e.g.
        "Sweetorange/Sweetorange/Converted Image". Lets datasets.yaml point
        at a nested container (dataset/container/container/class/image)
        without physically moving any files and without scanning sibling
        containers (e.g. an "Annotation" folder) that aren't needed.
    """
    effective_path = dataset_path
    if class_root_subpath:
        effective_path = os.path.join(dataset_path, class_root_subpath)

    layout = layout_override or detect_layout(effective_path)

    if layout == "class_subfolders":
        group_label = os.path.basename(class_root_subpath.rstrip("/\\")) if class_root_subpath else None
        records = _scan_class_subfolders(
            effective_path, dataset_name, valid_ext,
            group_label=group_label, provenance_root=dataset_path,
        )
    elif layout == "group_class_subfolders":
        records = _scan_group_class_subfolders(effective_path, dataset_name, valid_ext)
    elif layout == "nested_splits":
        records = _scan_nested_splits(effective_path, dataset_name, valid_ext)
    elif layout == "flat":
        # No class info available at all -- flag every image as "UNLABELED".
        records = []
        for fname in sorted(os.listdir(effective_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_ext:
                continue
            records.append(_inspect_image(os.path.join(effective_path, fname), dataset_name, "UNLABELED"))
    else:
        records = []

    profile = DatasetProfile(name=dataset_name, root=dataset_path, layout_detected=layout)
    profile.num_images = len(records)

    class_counts = defaultdict(int)
    format_counts = defaultdict(int)
    group_counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        class_counts[r.original_class] += 1
        format_counts[r.file_format or "UNKNOWN"] += 1
        if r.group:
            group_counts[r.group][r.original_class] += 1
        if r.corrupt:
            profile.corrupt_images.append({"path": r.path, "error": r.error})
        else:
            profile.resolutions.append((r.width, r.height))
            profile.aspect_ratios.append(r.aspect_ratio)

    profile.class_counts = dict(class_counts)
    profile.num_classes = len(class_counts)
    profile.formats = dict(format_counts)
    profile.group_counts = {g: dict(c) for g, c in group_counts.items()}

    if class_counts:
        counts = list(class_counts.values())
        profile.class_imbalance_ratio = round(max(counts) / max(min(counts), 1), 3)

    return profile, records


def find_exact_duplicates(records: list) -> list:
    """Byte-identical duplicate detection via file size + fast hash bucket.

    Cheap first pass before the perceptual-hash near-duplicate check in
    src/preprocessing/dedup.py (which is more expensive and content-aware).
    """
    import hashlib

    seen = {}
    duplicates = []
    for r in records:
        if r.corrupt:
            continue
        try:
            with open(r.path, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
        except OSError:
            continue
        if h in seen:
            duplicates.append({"original": seen[h], "duplicate": r.path})
        else:
            seen[h] = r.path
    return duplicates


def summarize_resolution_stats(resolutions: list) -> dict:
    if not resolutions:
        return {}
    widths = np.array([w for w, h in resolutions])
    heights = np.array([h for w, h in resolutions])
    return {
        "width_mean": float(widths.mean()),
        "width_min": int(widths.min()),
        "width_max": int(widths.max()),
        "height_mean": float(heights.mean()),
        "height_min": int(heights.min()),
        "height_max": int(heights.max()),
        "most_common_resolution": _most_common_pair(resolutions),
    }


def _most_common_pair(pairs: list):
    from collections import Counter

    c = Counter(pairs)
    if not c:
        return None
    (w, h), count = c.most_common(1)[0]
    return {"width": w, "height": h, "count": count}

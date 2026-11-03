"""
MODULE I - Duplicate / near-duplicate detection.

Uses perceptual hashing (pHash) so that near-identical images (recompressed,
resized, lightly cropped) are caught, not just byte-identical files. This is
what prevents data leakage across train/val/test splits.
"""
from collections import defaultdict

import imagehash
from PIL import Image


def compute_phash(image_path: str, hash_size: int = 8):
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img.convert("RGB"), hash_size=hash_size)
    except Exception:
        return None


def build_hash_index(image_paths: list, hash_size: int = 8) -> dict:
    """Returns {image_path: imagehash.ImageHash or None}."""
    return {p: compute_phash(p, hash_size) for p in image_paths}


def find_near_duplicate_groups(hash_index: dict, hamming_threshold: int = 5) -> list:
    """Greedy clustering of images whose pHash Hamming distance is below the
    threshold. Returns a list of groups (each a list of image paths); groups
    of size 1 are not near-duplicates of anything and are omitted.

    NOTE: this is O(n^2) in the number of images, which is fine for
    per-dataset or a few thousand images. For very large corpora, bucket by
    hash prefix first.
    """
    items = [(p, h) for p, h in hash_index.items() if h is not None]
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if items[i][1] - items[j][1] <= hamming_threshold:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(items[i][0])

    return [g for g in groups.values() if len(g) > 1]


def select_survivors(duplicate_groups: list) -> dict:
    """For each near-duplicate group, keep the first image (deterministic,
    sorted) and mark the rest for removal.

    Returns {"keep": [...], "remove": [...]}.
    """
    keep, remove = [], []
    for group in duplicate_groups:
        sorted_group = sorted(group)
        keep.append(sorted_group[0])
        remove.extend(sorted_group[1:])
    return {"keep": keep, "remove": remove}

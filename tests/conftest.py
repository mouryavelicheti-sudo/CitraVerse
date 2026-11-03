"""Shared fixtures for the AgriVerse test suite.

None of these tests touch your real datasets in C:\\Users\\Mourya\\Downloads\\
AWS research papers\\Datasets -- everything here uses tiny synthetic images
generated on the fly, so the suite runs anywhere (including CI) without the
real data present.
"""
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def synthetic_image_bgr():
    """A small deterministic synthetic BGR image (looks vaguely leaf-like:
    green background with a reddish-brown blotch, so shape/color evidence
    extraction has something non-trivial to measure)."""
    rng = np.random.RandomState(0)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 1] = 120  # green channel base (BGR order: B,G,R)
    img += rng.randint(0, 10, size=img.shape, dtype=np.uint8)
    # A "lesion" blotch.
    img[20:40, 20:40, 2] = 160  # reddish patch
    img[20:40, 20:40, 1] = 60
    return img


def _write_dummy_images(class_dir, n, rng, size=(48, 48)):
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = rng.randint(0, 255, size=(size[0], size[1], 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(class_dir / f"img_{i:03d}.jpg")


@pytest.fixture
def pakistan_citrus_root(tmp_path):
    """Mimics the real Pakistan_Citrus layout: group (Leaves|Fruits) -> class -> image.

    Leaves and Fruits are groups, NOT disease classes.
    """
    root = tmp_path / "Pakistan_Citrus"
    rng = np.random.RandomState(1)
    leaves = {
        "healthy": 4, "Melanose": 3, "greening": 3, "Black spot": 2, "canker": 2,
    }
    fruits = {
        "healthy": 3, "Scab": 3, "Black spot": 2, "Canker": 2, "Greening": 2,
    }
    for class_name, n in leaves.items():
        _write_dummy_images(root / "Leaves" / class_name, n, rng)
    for class_name, n in fruits.items():
        _write_dummy_images(root / "Fruits" / class_name, n, rng)
    return str(root)


@pytest.fixture
def bangladesh_sweet_orange_root(tmp_path):
    """Mimics the real Bangladesh_Sweet_Orange layout:
    Sweetorange/Sweetorange/{Original Image, Converted Image, Annotation}
    with class folders only under "Converted Image".
    """
    root = tmp_path / "Bangladesh_Sweet_Orange"
    nested = root / "Sweetorange" / "Sweetorange"
    rng = np.random.RandomState(2)

    converted_classes = {
        "Citrus canker": 4, "Citrus greening": 3, "Healthy leaf": 5,
        "Citrus mealybugs": 2, "Yellow leaves": 2,  # excluded classes, should still be discoverable but not mapped
    }
    for class_name, n in converted_classes.items():
        _write_dummy_images(nested / "Converted Image" / class_name, n, rng)

    # "Original Image" mirrors the same photos but is unlabeled/not used for classes.
    orig_dir = nested / "Original Image"
    orig_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        arr = rng.randint(0, 255, size=(48, 48, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(orig_dir / f"raw_{i}.jpg")

    # "Annotation" holds non-image files that must never be counted as images.
    ann_dir = nested / "Annotation"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / "labels.txt").write_text("some annotation content, not an image")

    return str(root)


@pytest.fixture
def bangladesh_lemon_root(tmp_path):
    """Mimics the real Bangladesh_lemon layout: class folders directly under
    the dataset root, including several classes with no six-class mapping."""
    root = tmp_path / "Bangladesh_lemon"
    rng = np.random.RandomState(3)
    classes = {
        "Healthy": 4, "Citrus Canker": 3, "Greening": 3, "Black Spot": 2,
        "Melanose": 2, "Citrus_Scab": 2,
        # excluded / unmapped classes:
        "Algal_Leaf_Spot": 2, "Anthracnose": 2, "Bacterial Blight": 1,
        "Citrus Hindu Mite": 1, "Citrus Leafminer": 1, "Citrus_Pest": 1,
        "Curl Leaf": 1, "Dry Leaf": 1, "Lemon_Sooty_Mold": 1,
        "Spider Mites": 1, "Swallowtail Larval Herbivory (Deficiency)": 1,
        "Yellow_Spot": 1,
    }
    for class_name, n in classes.items():
        _write_dummy_images(root / class_name, n, rng)
    return str(root)


@pytest.fixture
def synthetic_dataset_root(tmp_path):
    """Builds a tiny multi-dataset folder tree matching configs/datasets.yaml's
    expected class_subfolders layout:

        <root>/DatasetA/classX/*.jpg
        <root>/DatasetA/classY/*.jpg
        <root>/DatasetB/classX/*.jpg
    """
    root = tmp_path / "Datasets"
    layout = {
        "DatasetA": {"leaf_healthy": 6, "leaf_canker": 6},
        "DatasetB": {"healthy_leaf": 5, "canker_spot": 5},
    }
    rng = np.random.RandomState(42)
    for dataset_name, classes in layout.items():
        for class_name, n in classes.items():
            class_dir = root / dataset_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = rng.randint(0, 255, size=(48, 48, 3), dtype=np.uint8)
                Image.fromarray(arr, mode="RGB").save(class_dir / f"img_{i:03d}.jpg")
    return str(root)

"""
AgriVerse - Representative Input Image Selection
=================================================

Purpose
-------
Select representative source images from the three AgriVerse datasets.

Selection rule:
    - Up to 10 images per harmonized class per dataset
    - Fixed random seed for reproducibility
    - Original datasets are NEVER modified

The script also creates:
    1. Selected individual images organized by dataset/class
    2. One 6-column grid per dataset
    3. One combined overview grid

Important:
---------
The source datasets do not all contain the same classes.

Actual harmonized availability:
    Bangladesh_lemon       : 6 classes
    Bangladesh_Sweet_Orange: 3 classes
    Pakistan_Citrus        : 6 classes

Therefore the expected maximum is:
    60 + 30 + 60 = 150 images

This script uses exact source-folder mappings to avoid accidentally
mapping unrelated diseases to the six canonical classes.
"""

import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\Mourya\Downloads\AWS research papers")

OUTPUT_DIR = BASE_DIR / "outputs" / "representative_images"

IMAGES_PER_CLASS = 10
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

# Canonical classes used by the AgriVerse harmonization.
CANONICAL_CLASSES = [
    "healthy",
    "citrus_canker",
    "citrus_greening_hlb",
    "black_spot",
    "melanose",
    "scab",
]

CLASS_DISPLAY_NAMES = {
    "healthy": "Healthy",
    "citrus_canker": "Citrus Canker",
    "citrus_greening_hlb": "Citrus Greening / HLB",
    "black_spot": "Black Spot",
    "melanose": "Melanose",
    "scab": "Scab",
}


# ============================================================
# EXACT DATASET MAPPINGS
# ============================================================
#
# Only classes that actually exist in each source dataset
# are included.
#
# Multiple folders are allowed for a canonical class.
# This is used for Pakistan Citrus because its data is split
# between Fruits and Leaves.
# ============================================================

DATASETS = {

    "Bangladesh_lemon": {

        "healthy": [
            BASE_DIR / "Datasets" / "Bangladesh_lemon" / "Healthy"
        ],

        "citrus_canker": [
            BASE_DIR / "Datasets" / "Bangladesh_lemon" / "Citrus Canker"
        ],

        "citrus_greening_hlb": [
            BASE_DIR / "Datasets" / "Bangladesh_lemon" / "Greening"
        ],

        "black_spot": [
            BASE_DIR / "Datasets" / "Bangladesh_lemon" / "Black Spot"
        ],

        "melanose": [
            BASE_DIR / "Datasets" / "Bangladesh_lemon" / "Melanose"
        ],

        "scab": [
            BASE_DIR / "Datasets" / "Bangladesh_lemon" / "Citrus_Scab"
        ],
    },


    "Bangladesh_Sweet_Orange": {

        "healthy": [
            BASE_DIR
            / "Datasets"
            / "Bangladesh_Sweet_Orange"
            / "Sweetorange"
            / "Sweetorange"
            / "Original Image"
            / "Healthy leaf"
        ],

        "citrus_canker": [
            BASE_DIR
            / "Datasets"
            / "Bangladesh_Sweet_Orange"
            / "Sweetorange"
            / "Sweetorange"
            / "Original Image"
            / "Citrus canker"
        ],

        "citrus_greening_hlb": [
            BASE_DIR
            / "Datasets"
            / "Bangladesh_Sweet_Orange"
            / "Sweetorange"
            / "Sweetorange"
            / "Original Image"
            / "Citrus greening"
        ],
    },


    "Pakistan_Citrus": {

        "healthy": [
            BASE_DIR
            / "Datasets"
            / "Pakistan_Citrus"
            / "Fruits"
            / "healthy",

            BASE_DIR
            / "Datasets"
            / "Pakistan_Citrus"
            / "Leaves"
            / "healthy",
        ],

        "citrus_canker": [
            BASE_DIR
            / "Datasets"
            / "Pakistan_Citrus"
            / "Fruits"
            / "Canker",

            BASE_DIR
            / "Datasets"
            / "Pakistan_Citrus"
            / "Leaves"
            / "canker",
        ],

        "citrus_greening_hlb": [
            BASE_DIR
            / "Datasets"
            / "Pakistan_Citrus"
            / "Fruits"
            / "Greening",

            BASE_DIR
            / "Datasets"
            / "Pakistan_Citrus"
            / "Leaves"
            / "greening",
        ],

        "black_spot": [
            BASE_DIR
            / "Datasets"
            / "Pakistan_Citrus"
            / "Fruits"
            / "Black spot",

            BASE_DIR
            / "Datasets"
            / "Pakistan_Citrus"
            / "Leaves"
            / "Black spot",
        ],

        "melanose": [
            BASE_DIR
            / "Datasets"
            / "Pakistan_Citrus"
            / "Leaves"
            / "Melanose",
        ],

        "scab": [
            BASE_DIR
            / "Datasets"
            / "Pakistan_Citrus"
            / "Fruits"
            / "Scab",
        ],
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_images(folder):
    """
    Recursively find all supported image files in a folder.
    """
    folder = Path(folder)

    if not folder.exists():
        return []

    if not folder.is_dir():
        return []

    images = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return sorted(images)


def collect_class_images(source_folders):
    """
    Collect images from one or more source folders.
    Duplicate paths are removed.
    """
    all_images = []
    seen = set()

    for folder in source_folders:
        folder = Path(folder)

        if not folder.exists():
            print(f"  WARNING: Folder not found: {folder}")
            continue

        images = find_images(folder)

        for image_path in images:
            resolved = str(image_path.resolve())

            if resolved not in seen:
                seen.add(resolved)
                all_images.append(image_path)

    return sorted(all_images)


def safe_filename(path):
    """
    Make a filename safe and informative when copied to output.
    """
    return path.name


def select_images(images, number, seed):
    """
    Randomly select up to `number` images reproducibly.
    """
    rng = random.Random(seed)

    if len(images) <= number:
        return list(images)

    return rng.sample(images, number)


def copy_selected_images(
    selected_images,
    dataset_name,
    class_name,
):
    """
    Copy selected images into:
        OUTPUT_DIR / dataset_name / class_name
    """
    destination = OUTPUT_DIR / dataset_name / class_name
    destination.mkdir(parents=True, exist_ok=True)

    copied_paths = []

    for index, source in enumerate(selected_images, start=1):

        # Prefix avoids filename collisions between source folders.
        destination_name = (
            f"{index:02d}_{source.name}"
        )

        destination_path = destination / destination_name

        shutil.copy2(source, destination_path)

        copied_paths.append(destination_path)

    return copied_paths


def make_placeholder(ax, text):
    """
    Draw a clean placeholder when an image cannot be opened.
    """
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=9,
    )


def display_image(ax, image_path):
    """
    Display an image safely in a matplotlib axis.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = ImageOps.exif_transpose(img)

            ax.imshow(img)

        ax.axis("off")

    except Exception as exc:
        make_placeholder(
            ax,
            f"Unable to read\n{Path(image_path).name}\n{exc}",
        )


def create_dataset_grid(
    dataset_name,
    selected_by_class,
):
    """
    Create one grid for a dataset.

    Rows:
        harmonized classes

    Columns:
        up to 10 representative images
    """

    n_columns = IMAGES_PER_CLASS
    n_rows = len(CANONICAL_CLASSES)

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(20, 13),
    )

    # Ensure axes is always 2-D.
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(
        "Representative Images - "
        f"{dataset_name}\n"
        "Selected from source dataset after class harmonization",
        fontsize=16,
        fontweight="bold",
    )

    for row_index, class_name in enumerate(CANONICAL_CLASSES):

        images = selected_by_class.get(class_name, [])

        for column_index in range(n_columns):

            ax = axes[row_index][column_index]

            if column_index < len(images):
                display_image(
                    ax,
                    images[column_index],
                )
            else:
                ax.axis("off")

            if column_index == 0:
                ax.set_ylabel(
                    CLASS_DISPLAY_NAMES[class_name],
                    fontsize=10,
                    fontweight="bold",
                )

    plt.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    output_path = (
        OUTPUT_DIR
        / f"{dataset_name}_representative_grid.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def create_combined_grid(all_selected):
    """
    Create a combined overview.

    Each dataset gets a block of rows.
    Only classes actually present in the dataset are displayed.
    """

    rows = []

    for dataset_name, selected_by_class in all_selected.items():

        for class_name in CANONICAL_CLASSES:

            images = selected_by_class.get(class_name, [])

            if images:
                rows.append(
                    (
                        dataset_name,
                        class_name,
                        images,
                    )
                )

    if not rows:
        return None

    n_rows = len(rows)
    n_columns = IMAGES_PER_CLASS

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(20, max(3, n_rows * 2.6)),
    )

    # Normalize axes shape.
    if n_rows == 1:
        axes = [axes]

    for row_index, (
        dataset_name,
        class_name,
        images,
    ) in enumerate(rows):

        for column_index in range(n_columns):

            ax = axes[row_index][column_index]

            if column_index < len(images):
                display_image(
                    ax,
                    images[column_index],
                )
            else:
                ax.axis("off")

            if column_index == 0:
                ax.set_ylabel(
                    f"{dataset_name}\n"
                    f"{CLASS_DISPLAY_NAMES[class_name]}",
                    fontsize=9,
                    fontweight="bold",
                )

    fig.suptitle(
        "Representative Images from the Harmonized Citrus Datasets",
        fontsize=17,
        fontweight="bold",
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    output_path = (
        OUTPUT_DIR
        / "combined_representative_grid.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AgriVerse Representative Image Selection")
    print("=" * 70)

    print()
    print(f"Project directory : {BASE_DIR}")
    print(f"Output directory  : {OUTPUT_DIR}")
    print(f"Images per class  : {IMAGES_PER_CLASS}")
    print(f"Random seed       : {RANDOM_SEED}")

    # --------------------------------------------------------
    # Check project directory
    # --------------------------------------------------------

    if not BASE_DIR.exists():
        raise FileNotFoundError(
            f"Project directory does not exist:\n{BASE_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Output directory created/verified.")

    # --------------------------------------------------------
    # Process every dataset
    # --------------------------------------------------------

    all_selected = {}

    total_selected = 0

    for dataset_name, class_mapping in DATASETS.items():

        print()
        print("-" * 70)
        print(f"DATASET: {dataset_name}")
        print("-" * 70)

        selected_by_class = {}

        dataset_total = 0

        for class_name in CANONICAL_CLASSES:

            source_folders = class_mapping.get(
                class_name,
                [],
            )

            # Dataset does not contain this harmonized class.
            if not source_folders:
                print(
                    f"\n{CLASS_DISPLAY_NAMES[class_name]}:"
                    f" NOT PRESENT"
                )
                continue

            print(
                f"\n{CLASS_DISPLAY_NAMES[class_name]}"
            )

            print("Source folders:")

            for folder in source_folders:
                print(f"  {folder}")

            # Collect source images.
            images = collect_class_images(
                source_folders
            )

            print(
                f"Available images: {len(images)}"
            )

            if not images:
                print(
                    "WARNING: No images found for this class."
                )

                selected_by_class[class_name] = []

                continue

            # Select reproducibly.
            selected = select_images(
                images,
                IMAGES_PER_CLASS,
                RANDOM_SEED,
            )

            print(
                f"Selected images: {len(selected)}"
            )

            # Copy selected files.
            copied = copy_selected_images(
                selected,
                dataset_name,
                class_name,
            )

            selected_by_class[class_name] = copied

            dataset_total += len(copied)
            total_selected += len(copied)

        all_selected[dataset_name] = selected_by_class

        print()
        print(
            f"Dataset total selected: {dataset_total}"
        )

        # ----------------------------------------------------
        # Dataset grid
        # ----------------------------------------------------

        grid_path = create_dataset_grid(
            dataset_name,
            selected_by_class,
        )

        print(
            f"Dataset grid saved: {grid_path}"
        )

    # --------------------------------------------------------
    # Combined grid
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("Creating combined representative-image grid")
    print("-" * 70)

    combined_path = create_combined_grid(
        all_selected
    )

    if combined_path:
        print(
            f"Combined grid saved: {combined_path}"
        )
    else:
        print(
            "WARNING: No images were available for combined grid."
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SELECTION COMPLETE")
    print("=" * 70)

    print(
        f"Total representative images selected: "
        f"{total_selected}"
    )

    print()
    print("Expected based on the current dataset structure:")
    print("  Bangladesh_lemon        : up to 60")
    print("  Bangladesh_Sweet_Orange : up to 30")
    print("  Pakistan_Citrus         : up to 60")
    print("  --------------------------------")
    print("  Maximum expected total  : 150")

    print()
    print("Results:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("Original datasets were not modified.")

    print()
    print("Suggested paper wording:")
    print(
        "Representative images selected from each source "
        "dataset after class harmonization."
    )


if __name__ == "__main__":
    main()

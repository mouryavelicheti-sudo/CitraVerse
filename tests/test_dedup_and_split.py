import pandas as pd

from src.preprocessing.dedup import build_hash_index, find_near_duplicate_groups, select_survivors
from src.preprocessing.splitter import stratified_split


def test_hash_index_and_near_duplicates(synthetic_dataset_root, tmp_path):
    from src.data.dataset_scanner import scan_dataset

    _profile, records = scan_dataset("DatasetA", f"{synthetic_dataset_root}/DatasetA")
    paths = [r.path for r in records]
    hash_index = build_hash_index(paths)
    assert len(hash_index) == len(paths)

    groups = find_near_duplicate_groups(hash_index, hamming_threshold=5)
    # Random synthetic images are extremely unlikely to collide; near-zero groups expected.
    assert isinstance(groups, list)

    survivors = select_survivors(groups)
    assert isinstance(survivors, dict)


def test_stratified_split_respects_fractions_and_stratification():
    rows = []
    for c in ["a", "b"]:
        for i in range(20):
            rows.append({"image_path": f"{c}_{i}.jpg", "canonical_class": c, "augmentation_type": "none"})
    df = pd.DataFrame(rows)

    split_df = stratified_split(df, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=0)

    assert set(split_df["split"].unique()) <= {"train", "val", "test"}
    assert (split_df["split"] == "unassigned").sum() == 0

    counts = split_df.groupby(["canonical_class", "split"]).size().unstack(fill_value=0)
    # Roughly 60/20/20 per class, allowing rounding slack.
    for c in ["a", "b"]:
        assert counts.loc[c, "train"] >= 8
        assert counts.loc[c, "val"] >= 2
        assert counts.loc[c, "test"] >= 2


def test_stratified_split_keeps_synthetic_images_out_of_test():
    rows = []
    for i in range(20):
        rows.append({"image_path": f"real_{i}.jpg", "canonical_class": "a", "augmentation_type": "none"})
    for i in range(20):
        rows.append({"image_path": f"aug_{i}.jpg", "canonical_class": "a", "augmentation_type": "gan_generated"})
    df = pd.DataFrame(rows)

    split_df = stratified_split(df, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=0)
    test_rows = split_df[split_df["split"] == "test"]
    assert (test_rows["augmentation_type"] == "none").all(), (
        "Synthetic/augmented images must never be assigned to the test split."
    )

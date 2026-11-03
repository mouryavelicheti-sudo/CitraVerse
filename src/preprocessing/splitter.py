"""
MODULE I - Stratified train/validation/test splitting with provenance rules:
  - GAN-generated / synthetic / augmented images must NEVER land in test.
  - Split is stratified by canonical_class (configurable) to preserve class
    balance across splits.
  - Near-duplicate groups (from dedup.py) are kept together in the SAME split
    to avoid leakage.
"""
import numpy as np
import pandas as pd


def stratified_split(
    df: pd.DataFrame,
    stratify_col: str = "canonical_class",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    duplicate_groups: list = None,
    real_only_col: str = "augmentation_type",
) -> pd.DataFrame:
    """Assigns a 'split' column to df in-place-equivalent (returns a copy).

    duplicate_groups: optional list of lists of image_path that must be
        assigned to the same split (near-duplicate clusters).
    real_only_col: column whose value == "none" marks a REAL (non-synthetic)
        image. Synthetic/augmented rows are excluded from the eligible pool
        for TEST and only allowed into train/val.
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, (
        "train/val/test fractions must sum to 1.0"
    )
    rng = np.random.RandomState(seed)
    df = df.copy().reset_index(drop=True)
    df["split"] = "unassigned"

    # Build a mapping image_path -> group_id so duplicate groups move together.
    group_id = {}
    if duplicate_groups:
        for gid, group in enumerate(duplicate_groups):
            for p in group:
                group_id[p] = gid
    df["_group_id"] = df["image_path"].map(lambda p: group_id.get(p, -1))

    is_real = df[real_only_col].astype(str).str.lower().isin(["none", "nan", ""])

    for cls, cls_df in df.groupby(stratify_col):
        idx_real = cls_df.index[is_real.loc[cls_df.index]].tolist()
        idx_synth = cls_df.index[~is_real.loc[cls_df.index]].tolist()

        # Collapse duplicate-group members so a group is treated as one unit
        # when shuffling/splitting real images (synthetic images typically
        # have no meaningful duplicate group and are handled separately).
        real_groups = _collapse_groups(cls_df.loc[idx_real], "_group_id")
        rng.shuffle(real_groups)

        n_real = sum(len(g) for g in real_groups)
        n_train = int(round(n_real * train_frac))
        n_val = int(round(n_real * val_frac))

        train_rows, val_rows, test_rows = [], [], []
        running = 0
        for g in real_groups:
            running_before = running
            running += len(g)
            if running_before < n_train:
                train_rows.extend(g)
            elif running_before < n_train + n_val:
                val_rows.extend(g)
            else:
                test_rows.extend(g)

        df.loc[train_rows, "split"] = "train"
        df.loc[val_rows, "split"] = "val"
        df.loc[test_rows, "split"] = "test"

        # Synthetic/augmented samples: NEVER test. Split proportionally
        # between train and val only, preserving relative train:val ratio.
        rng.shuffle(idx_synth)
        if idx_synth:
            train_val_total = train_frac + val_frac
            n_synth_train = int(round(len(idx_synth) * (train_frac / train_val_total)))
            df.loc[idx_synth[:n_synth_train], "split"] = "train"
            df.loc[idx_synth[n_synth_train:], "split"] = "val"

    df = df.drop(columns=["_group_id"])
    unassigned = (df["split"] == "unassigned").sum()
    if unassigned:
        # Fallback: any leftover row (e.g. class with zero real images) goes to train.
        df.loc[df["split"] == "unassigned", "split"] = "train"

    _assert_no_synthetic_in_test(df, real_only_col)
    return df


def _collapse_groups(sub_df: pd.DataFrame, group_col: str) -> list:
    """Returns a list of lists of row-indices, where rows sharing a real
    group id (>=0) are combined into one unit, and each ungrouped row (-1)
    is its own singleton unit."""
    groups = {}
    singles = []
    for idx, gid in sub_df[group_col].items():
        if gid == -1:
            singles.append([idx])
        else:
            groups.setdefault(gid, []).append(idx)
    return list(groups.values()) + singles


def _assert_no_synthetic_in_test(df: pd.DataFrame, real_only_col: str) -> None:
    test_df = df[df["split"] == "test"]
    is_real = test_df[real_only_col].astype(str).str.lower().isin(["none", "nan", ""])
    if not is_real.all():
        offending = test_df[~is_real]
        raise RuntimeError(
            "Integrity check failed: synthetic/augmented images were assigned "
            f"to the TEST split. Offending rows:\n{offending[['image_path', real_only_col]]}"
        )

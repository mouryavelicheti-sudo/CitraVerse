"""Tests for src/data/image_dataset.py.

Covers the DataLoader crash reported against `python run_pipeline.py
--train-teacher`:

    Caught TypeError in DataLoader worker process 0.
    ...
    torch/utils/data/_utils/collate.py
        return torch.tensor(batch, dtype=torch.float64)
    TypeError: must be real number, not str

Root cause
----------
`AgriVerseImageDataset.__getitem__` returns `(image_tensor, label_int,
meta_dict)`. `meta_dict` carries pure provenance strings (image_path,
dataset, original_class, group, split, ...) -- not model inputs. Some of
these fields don't apply to every source dataset: `group` is
"Leaves"/"Fruits" for Pakistan_Citrus and "Converted Image" for
Bangladesh_Sweet_Orange, but Bangladesh_lemon has no group/subset concept
at all, so `ProvenanceTracker` records `group=None` for those rows. Once
that goes through `harmonized_metadata.csv` and back through
`pandas.read_csv`, a missing `group` comes back as `float('nan')`, not
`None` or `""`.

When `torch.utils.data.DataLoader`'s default `collate_fn` batches a dict,
it collates each field independently and decides HOW to combine a field's
values by inspecting only the *first* sample's type for that field. If a
shuffled batch's first row is a Bangladesh_lemon row (`group` == NaN,
a float) but a later row in the same batch is a Pakistan_Citrus/BSO row
(`group` == a string), collate takes the "batch of numeric scalars" path
and calls `torch.tensor(batch, dtype=torch.float64)` on a list that also
contains strings -- exactly the reported crash.

Fix: `agriverse_collate` (see src/data/image_dataset.py) collates images
and labels with the normal numeric `default_collate`, but *never* runs
`meta` through it -- `meta` is transposed into a plain dict-of-lists
instead, so strings/None pass through untouched.
"""
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from src.data.image_dataset import AgriVerseImageDataset, agriverse_collate, build_transforms  # noqa: E402

CANONICAL_CLASSES = ["healthy", "citrus_canker", "citrus_greening_hlb", "black_spot", "melanose", "scab"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}


def _write_dummy_image(path, rng, size=(32, 32)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = rng.randint(0, 255, size=(size[0], size[1], 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _build_mixed_group_metadata_csv(tmp_path, n_per_class=3):
    """Builds a tiny harmonized_metadata.csv mirroring the real pipeline's
    output: some rows have a string `group` (like Pakistan_Citrus /
    Bangladesh_Sweet_Orange), others have `group=None` (like
    Bangladesh_lemon), reproducing the exact mixed-type condition that
    crashes the default collate_fn. Row order is NOT shuffled here so the
    very first row of the first batch is a no-group (NaN) row -- this is
    the condition that actually triggers the bug (see module docstring).
    """
    rng = np.random.RandomState(0)
    rows = []
    idx = 0
    # Bangladesh_lemon-like rows first: group=None -> NaN after CSV round-trip.
    for cls in ["healthy", "citrus_canker"]:
        for _ in range(n_per_class):
            img_path = str(tmp_path / "images" / f"lemon_{idx}.jpg")
            _write_dummy_image(img_path, rng)
            rows.append({
                "image_path": img_path,
                "original_image": img_path,
                "dataset": "Bangladesh_lemon",
                "original_class": cls if cls != "citrus_canker" else "Citrus Canker",
                "canonical_class": cls,
                "group": None,
                "original_relative_path": f"{cls}/img_{idx}.jpg",
                "augmentation_type": "none",
                "source_image": img_path,
                "split": "train",
            })
            idx += 1
    # Pakistan_Citrus-like rows: group is a real string.
    for cls, group in [("healthy", "Leaves"), ("citrus_canker", "Fruits")]:
        for _ in range(n_per_class):
            img_path = str(tmp_path / "images" / f"pk_{idx}.jpg")
            _write_dummy_image(img_path, rng)
            rows.append({
                "image_path": img_path,
                "original_image": img_path,
                "dataset": "Pakistan_Citrus",
                "original_class": cls,
                "canonical_class": cls,
                "group": group,
                "original_relative_path": f"{group}/{cls}/img_{idx}.jpg",
                "augmentation_type": "none",
                "source_image": img_path,
                "split": "train",
            })
            idx += 1

    csv_path = tmp_path / "harmonized_metadata.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(csv_path)


def test_missing_group_round_trips_as_nan_not_none_or_string(tmp_path):
    """Confirms the actual data condition that causes the bug: a `group`
    value of None, once written to CSV and re-read, comes back as a float
    NaN sitting in the same object column as ordinary strings."""
    metadata_csv = _build_mixed_group_metadata_csv(tmp_path)
    df = pd.read_csv(metadata_csv)

    lemon_group = df.loc[df["dataset"] == "Bangladesh_lemon", "group"].iloc[0]
    pk_group = df.loc[df["dataset"] == "Pakistan_Citrus", "group"].iloc[0]

    assert isinstance(pk_group, str)
    assert isinstance(lemon_group, float) and np.isnan(lemon_group), (
        "A missing `group` must round-trip through the metadata CSV as NaN "
        "(float), not None/empty-string -- this mixed-type column is the "
        "actual trigger for the reported collate crash."
    )


def test_default_collate_reproduces_the_reported_typeerror(tmp_path):
    """Reproduces the EXACT reported failure: using the vanilla DataLoader
    default collate_fn (no collate_fn override) on a batch whose first
    sample has a NaN `group` and whose later samples have string `group`
    values raises the reported TypeError. This is what used to happen
    before scripts/train_teacher.py (and train_student.py/evaluate.py/
    quantize_model.py) started passing collate_fn=agriverse_collate.
    """
    metadata_csv = _build_mixed_group_metadata_csv(tmp_path)
    ds = AgriVerseImageDataset(metadata_csv, "train", CLASS_TO_IDX, build_transforms(32, train=False))

    # shuffle=False + num_workers=0: deterministic, and isolates the bug to
    # the collate step itself rather than multiprocessing pickling.
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)

    with pytest.raises(TypeError):
        next(iter(loader))


def test_agriverse_collate_batches_mixed_metadata_successfully(tmp_path):
    """The actual fix: with collate_fn=agriverse_collate, the same dataset
    (same mixed-type `group` column, same lack of shuffling) batches
    cleanly, images/labels stay real tensors, and every metadata field
    (including the problematic `group` column) survives untouched as a
    plain Python list -- no silent string->number coercion anywhere.
    """
    metadata_csv = _build_mixed_group_metadata_csv(tmp_path)
    ds = AgriVerseImageDataset(metadata_csv, "train", CLASS_TO_IDX, build_transforms(32, train=False))

    loader = torch.utils.data.DataLoader(
        ds, batch_size=len(ds), shuffle=False, num_workers=0, collate_fn=agriverse_collate,
    )
    images, labels, meta = next(iter(loader))

    # Images: a real float tensor, correctly shaped/batched.
    assert isinstance(images, torch.Tensor)
    assert images.dtype.is_floating_point
    assert images.shape == (len(ds), 3, 32, 32)

    # Labels: integer class indices, torch.long, in-range.
    assert isinstance(labels, torch.Tensor)
    assert labels.dtype == torch.long
    assert set(labels.tolist()) <= set(range(len(CANONICAL_CLASSES)))

    # Metadata: never turned into tensors; strings/None pass through as-is.
    assert isinstance(meta, dict)
    assert set(meta.keys()) == {
        "image_path", "dataset", "original_class", "canonical_class", "group",
        "augmentation_type", "split",
    }
    assert len(meta["image_path"]) == len(ds)
    assert all(isinstance(p, str) for p in meta["image_path"])
    # The exact mixed-type field that used to crash default_collate:
    assert any(g is None for g in meta["group"]), "Bangladesh_lemon-style rows should keep group=None."
    assert any(isinstance(g, str) for g in meta["group"]), "Pakistan-style rows should keep their string group."
    assert all(isinstance(c, str) for c in meta["canonical_class"])


def test_agriverse_collate_also_works_with_worker_processes(tmp_path):
    """Same as above but with num_workers=2, matching how train_teacher.py
    actually runs -- the original bug report is specifically a
    'DataLoader worker process' crash, so this exercises the
    multiprocessing collate path too, not just the main-process path."""
    metadata_csv = _build_mixed_group_metadata_csv(tmp_path)
    ds = AgriVerseImageDataset(metadata_csv, "train", CLASS_TO_IDX, build_transforms(32, train=False))

    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=True, num_workers=2, collate_fn=agriverse_collate,
    )
    n_seen = 0
    for images, labels, meta in loader:
        assert isinstance(images, torch.Tensor)
        assert labels.dtype == torch.long
        assert isinstance(meta, dict)
        n_seen += images.shape[0]
    assert n_seen == len(ds)


@pytest.mark.parametrize("script_path,var_name", [
    ("scripts/train_teacher.py", "train_loader"),
    ("scripts/train_teacher.py", "val_loader"),
    ("scripts/train_student.py", "train_loader"),
    ("scripts/train_student.py", "val_loader"),
    ("scripts/evaluate.py", "test_loader"),
    ("scripts/quantize_model.py", "calib_loader"),
])
def test_all_agriverseimagedataset_dataloaders_use_agriverse_collate(script_path, var_name):
    """Guards against regressing back to the crash by someone constructing
    a new/edited DataLoader over AgriVerseImageDataset without
    collate_fn=agriverse_collate."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(repo_root, script_path), "r", encoding="utf-8") as f:
        source = f.read()

    # Find the DataLoader(...) call assigned to var_name (may span multiple
    # lines and contains nested parens, e.g. cfg.get("batch_size", 32)) --
    # so do a proper balanced-paren scan rather than stopping at the first ")".
    marker = f"{var_name} = DataLoader("
    open_paren_idx = source.index(marker) + len(marker) - 1
    depth = 0
    end = None
    for i in range(open_paren_idx, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end is not None, f"Could not find closing paren for {marker} in {script_path}"
    call_text = source[open_paren_idx:end]
    assert "collate_fn=agriverse_collate" in call_text, (
        f"{script_path}:{var_name} must pass collate_fn=agriverse_collate to avoid "
        "the reported 'TypeError: must be real number, not str' collate crash."
    )


def test_teacher_training_smoke_end_to_end(tmp_path):
    """Minimal end-to-end smoke test standing in for
    `python run_pipeline.py --train-teacher`: builds a tiny synthetic
    harmonized_metadata.csv (with the same mixed-type `group` column that
    triggers the reported bug), constructs the DataLoader exactly the way
    scripts/train_teacher.py does (collate_fn=agriverse_collate), and runs
    one real forward/backward/optimizer step through an untrained
    (pretrained=False, no network needed) DiseaseClassifier. This is what
    would previously crash with 'Caught TypeError in DataLoader worker
    process 0'; after the fix it should complete without error.
    """
    from src.models.classifier import DiseaseClassifier

    metadata_csv = _build_mixed_group_metadata_csv(tmp_path, n_per_class=2)
    train_ds = AgriVerseImageDataset(metadata_csv, "train", CLASS_TO_IDX, build_transforms(32, train=True))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=agriverse_collate,
    )

    model = DiseaseClassifier.build("resnet18", len(CANONICAL_CLASSES), pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    batches_run = 0
    for images, labels, meta in train_loader:
        optimizer.zero_grad()
        logits, probs, _feat = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)
        assert probs.shape == (images.shape[0], len(CANONICAL_CLASSES))
        assert len(meta["image_path"]) == images.shape[0]
        batches_run += 1

    assert batches_run > 0

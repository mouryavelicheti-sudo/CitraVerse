"""PyTorch Dataset backed by the harmonized metadata table (CSV) produced by
Module I (see scripts/harmonize_datasets.py). Every row carries full
provenance: original_image, dataset, original_class, canonical_class,
augmentation_type, source_image, split.
"""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class AgriVerseImageDataset(Dataset):
    def __init__(self, metadata_csv: str, split: str, class_to_idx: dict, transform=None):
        df = pd.read_csv(metadata_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(
                f"No rows found for split='{split}' in {metadata_csv}. "
                "Did you run scripts/harmonize_datasets.py first?"
            )
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row["canonical_class"]]
        raw_group = row.get("group") if "group" in self.df.columns else None
        meta = {
            "image_path": row["image_path"],
            "dataset": row["dataset"],
            "original_class": row["original_class"],
            "canonical_class": row["canonical_class"],
            # pandas.read_csv() reads a missing `group` (e.g. Bangladesh_lemon,
            # which has no group/subset concept) back as float NaN, not None.
            # Normalize that here with pd.isna() (NOT `raw_group != raw_group`
            # or `raw_group == float('nan')`, which are the classic broken
            # ways people try to spell "is NaN") so the dataset's metadata
            # contract is always a real Python `None` for "no group",
            # regardless of what pandas' missing-value representation is.
            "group": None if pd.isna(raw_group) else raw_group,
            "augmentation_type": row.get("augmentation_type", "none"),
            "split": row["split"],
        }
        return img, label, meta


def agriverse_collate(batch):
    """collate_fn for DataLoader(AgriVerseImageDataset(...), ...).

    Each sample from AgriVerseImageDataset.__getitem__ is a 3-tuple:
    (image_tensor, label_int, meta_dict). Images and labels are ordinary
    numeric data and are safe to batch with torch's default collation
    (`torch.utils.data._utils.collate.default_collate`), which stacks them
    into a single image tensor and a torch.long label tensor.

    `meta` must NOT go through that same default numeric collation. Its
    values are provenance strings (image_path, dataset, original_class,
    group, split, ...) needed later for evaluation/export, not model
    inputs. Some of these fields are legitimately missing for some source
    datasets -- e.g. `group` is "Leaves"/"Fruits" for Pakistan_Citrus,
    "Converted Image" for Bangladesh_Sweet_Orange, but None/NaN for
    Bangladesh_lemon (it has no group/subset). When such a field is batched
    with `default_collate`, PyTorch inspects only the *first* element's
    type to decide how to combine the whole batch. If that first element
    happens to be a float (a missing `group` re-read as NaN out of the CSV)
    while other rows in the same shuffled batch are strings, collate takes
    the "batch of numeric scalars" path and calls
    `torch.tensor(batch, dtype=torch.float64)` on a list that also contains
    strings, raising `TypeError: must be real number, not str`. This is
    exactly the reported DataLoader worker crash.

    Fix: never hand `meta` to `default_collate`. Instead, transpose the
    list of per-sample meta dicts into a dict of per-field Python lists
    (a bare Python "collate" with no type coercion at all), so strings,
    None, and any other metadata value pass through untouched while images
    and labels remain proper tensors.
    """
    images = default_collate([sample[0] for sample in batch])
    labels = default_collate([sample[1] for sample in batch])

    meta_list = [sample[2] for sample in batch]
    keys = meta_list[0].keys() if meta_list else []
    meta = {key: [m[key] for m in meta_list] for key in keys}

    return images, labels, meta


def build_transforms(image_size: int = 224, train: bool = True):
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.15, 0.15, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

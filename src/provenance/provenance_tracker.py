"""
MODULE I - Provenance Control.

Every row that ever enters the pipeline (real or augmented/synthetic) is
tracked with:
    original_image, dataset, original_class, canonical_class,
    augmentation_type, source_image, split

This is what lets us guarantee (and prove, via export) that:
  - synthetic/augmented images never leak into the test set,
  - every prediction can be traced back to its source dataset and class,
  - duplicate removal and harmonization decisions are auditable.
"""
import pandas as pd


PROVENANCE_COLUMNS = [
    "image_path",
    "original_image",
    "dataset",
    "original_class",
    "canonical_class",
    "group",
    "original_relative_path",
    "augmentation_type",
    "source_image",
    "split",
]


class ProvenanceTracker:
    def __init__(self):
        self._rows = []

    def add_real_image(self, image_path: str, dataset: str, original_class: str, canonical_class: str,
                        group: str = None, relative_path: str = None):
        self._rows.append({
            "image_path": image_path,
            "original_image": image_path,
            "dataset": dataset,
            "original_class": original_class,
            "canonical_class": canonical_class,
            "group": group,                        # e.g. "Leaves"/"Fruits"/"Converted Image", or None
            "original_relative_path": relative_path,  # path relative to the dataset's configured root
            "augmentation_type": "none",
            "source_image": image_path,
            "split": None,  # filled in later by the splitter
        })

    def add_augmented_image(
        self, image_path: str, dataset: str, original_class: str, canonical_class: str,
        augmentation_type: str, source_image: str, group: str = None, relative_path: str = None,
    ):
        self._rows.append({
            "image_path": image_path,
            "original_image": source_image,
            "dataset": dataset,
            "original_class": original_class,
            "canonical_class": canonical_class,
            "group": group,
            "original_relative_path": relative_path,
            "augmentation_type": augmentation_type,
            "source_image": source_image,
            "split": None,
        })

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._rows, columns=PROVENANCE_COLUMNS)
        return df

    def save(self, path: str) -> pd.DataFrame:
        df = self.to_dataframe()
        if path.endswith(".csv"):
            df.to_csv(path, index=False)
        elif path.endswith(".json"):
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError("Provenance export must be .csv or .json")
        return df

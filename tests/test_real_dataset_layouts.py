"""Tests for the three REAL AgriVerse dataset layouts:

    Test 1: Pakistan_Citrus       -- group (Leaves|Fruits) / class / image
    Test 2: Bangladesh_Sweet_Orange -- nested container / container / "Converted Image" / class / image
    Test 3: Bangladesh_lemon      -- root / class / image

These use tiny synthetic fixtures (see conftest.py) that mirror the *real*
folder/class names exactly, so mapping logic is exercised against the real
names rather than placeholder strings. No real dataset files are touched.
"""
import os

from src.data.dataset_scanner import scan_dataset
from src.harmonization.class_mapper import harmonize
from src.utils.io_utils import load_yaml

REAL_CLASS_MAPPING_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "class_mapping.yaml")
)


def _load_real_mapping_config():
    with open(REAL_CLASS_MAPPING_CONFIG, "r", encoding="utf-8") as f:
        import yaml
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Test 1: Pakistan_Citrus group/class/image layout
# ---------------------------------------------------------------------------

def test_pakistan_layout_discovers_all_images_across_groups(pakistan_citrus_root):
    profile, records = scan_dataset(
        "Pakistan_Citrus", pakistan_citrus_root, layout_override="group_class_subfolders",
    )
    # 4+3+3+2+2 (Leaves) + 3+3+2+2+2 (Fruits) = 14 + 12 = 26
    assert profile.num_images == 26
    assert len(records) == 26


def test_pakistan_leaves_and_fruits_are_groups_not_classes(pakistan_citrus_root):
    profile, records = scan_dataset(
        "Pakistan_Citrus", pakistan_citrus_root, layout_override="group_class_subfolders",
    )
    # "Leaves" and "Fruits" must never appear as class names.
    assert "Leaves" not in profile.class_counts
    assert "Fruits" not in profile.class_counts
    assert set(profile.group_counts.keys()) == {"Leaves", "Fruits"}
    assert all(r.group in {"Leaves", "Fruits"} for r in records)

    # Original class names (case-sensitive, exactly as on disk) are preserved.
    assert set(profile.class_counts.keys()) == {
        "healthy", "Melanose", "greening", "Black spot", "canker",
        "Scab", "Canker", "Greening",
    }


def test_pakistan_aggregates_same_class_across_both_groups(pakistan_citrus_root):
    profile, _records = scan_dataset(
        "Pakistan_Citrus", pakistan_citrus_root, layout_override="group_class_subfolders",
    )
    # "healthy" appears in both Leaves (4) and Fruits (3) -> aggregated count.
    assert profile.class_counts["healthy"] == 4 + 3
    # "Black spot" appears identically-cased in both groups (2 + 2).
    assert profile.class_counts["Black spot"] == 2 + 2
    assert profile.group_counts["Leaves"]["healthy"] == 4
    assert profile.group_counts["Fruits"]["healthy"] == 3


def test_pakistan_six_class_mapping_uses_real_config(pakistan_citrus_root):
    profile, _records = scan_dataset(
        "Pakistan_Citrus", pakistan_citrus_root, layout_override="group_class_subfolders",
    )
    mapping_config = _load_real_mapping_config()
    records_by_dataset_class = {("Pakistan_Citrus", cls): count for cls, count in profile.class_counts.items()}
    result = harmonize(records_by_dataset_class, mapping_config)

    assert result.unmapped == [], "Every real Pakistan_Citrus class should have a defensible mapping."
    mapped = {(m["original_class"]): m["canonical_class"] for m in result.mapped}
    assert mapped == {
        "healthy": "healthy",
        "Melanose": "melanose",
        "greening": "citrus_greening_hlb",
        "Black spot": "black_spot",
        "canker": "citrus_canker",
        "Canker": "citrus_canker",
        "Scab": "scab",
        "Greening": "citrus_greening_hlb",
    }
    # All six target classes are actually represented in Pakistan_Citrus.
    assert set(mapped.values()) == {
        "healthy", "citrus_canker", "citrus_greening_hlb", "black_spot", "melanose", "scab",
    }


# ---------------------------------------------------------------------------
# Test 2: Bangladesh_Sweet_Orange nested Converted Image/class/image layout
# ---------------------------------------------------------------------------

def test_bangladesh_sweet_orange_discovers_only_converted_image(bangladesh_sweet_orange_root):
    profile, records = scan_dataset(
        "Bangladesh_Sweet_Orange", bangladesh_sweet_orange_root,
        layout_override="class_subfolders",
        class_root_subpath=os.path.join("Sweetorange", "Sweetorange", "Converted Image"),
    )
    # 4 + 3 + 5 + 2 + 2 = 16 images under Converted Image.
    assert profile.num_images == 16
    assert len(records) == 16
    # "Original Image" (3 unlabeled photos) and "Annotation" (.txt) must NOT be counted.
    assert set(profile.class_counts.keys()) == {
        "Citrus canker", "Citrus greening", "Healthy leaf", "Citrus mealybugs", "Yellow leaves",
    }


def test_bangladesh_sweet_orange_ignores_annotation_text_files(bangladesh_sweet_orange_root):
    profile, records = scan_dataset(
        "Bangladesh_Sweet_Orange", bangladesh_sweet_orange_root,
        layout_override="class_subfolders",
        class_root_subpath=os.path.join("Sweetorange", "Sweetorange", "Converted Image"),
    )
    # The "Annotation" folder (sibling of "Converted Image") is never scanned at
    # all, and even if it were, its .txt file is not a supported image extension.
    assert "Annotation" not in profile.class_counts
    assert "labels" not in profile.class_counts
    assert all(r.path.lower().endswith((".jpg", ".jpeg", ".png")) for r in records)


def test_bangladesh_sweet_orange_six_class_mapping_excludes_pests_and_symptoms(bangladesh_sweet_orange_root):
    profile, _records = scan_dataset(
        "Bangladesh_Sweet_Orange", bangladesh_sweet_orange_root,
        layout_override="class_subfolders",
        class_root_subpath=os.path.join("Sweetorange", "Sweetorange", "Converted Image"),
    )
    mapping_config = _load_real_mapping_config()
    records_by_dataset_class = {
        ("Bangladesh_Sweet_Orange", cls): count for cls, count in profile.class_counts.items()
    }
    result = harmonize(records_by_dataset_class, mapping_config)

    mapped = {m["original_class"]: m["canonical_class"] for m in result.mapped}
    assert mapped == {
        "Citrus canker": "citrus_canker",
        "Citrus greening": "citrus_greening_hlb",
        "Healthy leaf": "healthy",
    }
    # Pest/symptom classes must be excluded, not force-mapped anywhere.
    excluded_originals = {e["original_class"] for e in result.excluded}
    assert excluded_originals == {"Citrus mealybugs", "Yellow leaves"}

    canonical_targets = set(mapped.values())
    assert "black_spot" not in canonical_targets
    assert "melanose" not in canonical_targets
    assert "scab" not in canonical_targets
    assert "unknown_disease" not in canonical_targets


# ---------------------------------------------------------------------------
# Test 3: Bangladesh_lemon root/class/image layout
# ---------------------------------------------------------------------------

def test_bangladesh_lemon_discovers_all_source_classes(bangladesh_lemon_root):
    profile, records = scan_dataset("Bangladesh_lemon", bangladesh_lemon_root, layout_override="class_subfolders")
    assert profile.num_images == sum([4, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert len(records) == profile.num_images
    assert profile.num_classes == 18  # all source categories, per instructions section 11


def test_bangladesh_lemon_harmonization_selects_only_six_target_classes(bangladesh_lemon_root):
    profile, _records = scan_dataset("Bangladesh_lemon", bangladesh_lemon_root, layout_override="class_subfolders")
    mapping_config = _load_real_mapping_config()
    records_by_dataset_class = {("Bangladesh_lemon", cls): count for cls, count in profile.class_counts.items()}
    result = harmonize(records_by_dataset_class, mapping_config)

    mapped = {m["original_class"]: m["canonical_class"] for m in result.mapped}
    assert mapped == {
        "Healthy": "healthy",
        "Citrus Canker": "citrus_canker",
        "Greening": "citrus_greening_hlb",
        "Black Spot": "black_spot",
        "Melanose": "melanose",
        "Citrus_Scab": "scab",
    }

    excluded_originals = {e["original_class"] for e in result.excluded}
    expected_excluded = {
        "Algal_Leaf_Spot", "Anthracnose", "Bacterial Blight", "Citrus Hindu Mite",
        "Citrus Leafminer", "Citrus_Pest", "Curl Leaf", "Dry Leaf", "Lemon_Sooty_Mold",
        "Spider Mites", "Swallowtail Larval Herbivory (Deficiency)", "Yellow_Spot",
    }
    assert excluded_originals == expected_excluded
    # Nothing excluded should have snuck into the six-class mapping.
    assert expected_excluded.isdisjoint(mapped.keys())
    # Explicitly forbidden mis-mappings (visual-similarity guesses) never happen.
    assert result.unmapped == []  # every source class is either mapped or explicitly excluded


# ---------------------------------------------------------------------------
# Cross-cutting: provenance + no file mutation
# ---------------------------------------------------------------------------

def test_scanning_does_not_modify_or_move_any_image_files(pakistan_citrus_root):
    before = {}
    for dirpath, _dirs, files in os.walk(pakistan_citrus_root):
        for fname in files:
            fpath = os.path.join(dirpath, fname)
            before[fpath] = (os.path.getsize(fpath), os.path.getmtime(fpath))

    scan_dataset("Pakistan_Citrus", pakistan_citrus_root, layout_override="group_class_subfolders")

    after = {}
    for dirpath, _dirs, files in os.walk(pakistan_citrus_root):
        for fname in files:
            fpath = os.path.join(dirpath, fname)
            after[fpath] = (os.path.getsize(fpath), os.path.getmtime(fpath))

    assert before == after
    assert set(before.keys()) == set(after.keys())


def test_provenance_fields_are_preserved_on_every_record(pakistan_citrus_root, bangladesh_sweet_orange_root):
    _profile, pk_records = scan_dataset(
        "Pakistan_Citrus", pakistan_citrus_root, layout_override="group_class_subfolders",
    )
    for r in pk_records:
        assert r.dataset == "Pakistan_Citrus"
        assert r.group in {"Leaves", "Fruits"}
        assert r.relative_path is not None
        assert not os.path.isabs(r.relative_path)

    _profile2, bso_records = scan_dataset(
        "Bangladesh_Sweet_Orange", bangladesh_sweet_orange_root,
        layout_override="class_subfolders",
        class_root_subpath=os.path.join("Sweetorange", "Sweetorange", "Converted Image"),
    )
    for r in bso_records:
        assert r.dataset == "Bangladesh_Sweet_Orange"
        assert r.group == "Converted Image"
        # relative_path is computed against the dataset ROOT, not the nested subpath.
        assert r.relative_path.replace("\\", "/").startswith("Sweetorange/Sweetorange/Converted Image")


# ---------------------------------------------------------------------------
# Config-level sanity checks (guards against regressions in the YAML itself)
# ---------------------------------------------------------------------------

def test_real_datasets_config_has_exactly_the_three_real_datasets():
    ds_config = load_yaml("configs/datasets.yaml")
    assert set(ds_config["datasets"].keys()) == {
        "Pakistan_Citrus", "Bangladesh_Sweet_Orange", "Bangladesh_lemon",
    }
    assert "PlantVillage_Citrus" not in ds_config["datasets"]
    assert ds_config["datasets"]["Pakistan_Citrus"]["layout"] == "group_class_subfolders"
    assert ds_config["datasets"]["Bangladesh_Sweet_Orange"]["class_root_subpath"]
    assert ds_config["valid_extensions"] == [".jpg", ".jpeg", ".png"]


def test_real_class_mapping_config_has_exactly_six_canonical_classes():
    mapping_config = _load_real_mapping_config()
    assert mapping_config["canonical_classes"] == [
        "healthy", "citrus_canker", "citrus_greening_hlb", "black_spot", "melanose", "scab",
    ]
    assert "unknown_disease" not in mapping_config["canonical_classes"]

    # Never map excluded/pest categories into disease classes.
    forbidden_targets = {"black_spot", "melanose", "scab", "unknown_disease"}
    bso_mapping = mapping_config["mapping"]["Bangladesh_Sweet_Orange"]
    assert set(bso_mapping.values()).isdisjoint(forbidden_targets)
    assert "Yellow leaves" not in bso_mapping
    assert "Shot hole" not in bso_mapping

    lemon_mapping = mapping_config["mapping"]["Bangladesh_lemon"]
    assert "Algal_Leaf_Spot" not in lemon_mapping
    assert "Anthracnose" not in lemon_mapping

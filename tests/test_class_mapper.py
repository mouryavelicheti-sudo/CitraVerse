from src.harmonization.class_mapper import harmonize, suggest_mapping


def test_harmonize_maps_known_classes_and_flags_unmapped():
    records_by_dataset_class = {
        ("DatasetA", "leaf_healthy"): 10,
        ("DatasetA", "leaf_canker"): 8,
        ("DatasetA", "mystery_class"): 3,
    }
    mapping_config = {
        "canonical_classes": ["healthy", "citrus_canker"],
        "mapping": {"DatasetA": {"leaf_healthy": "healthy", "leaf_canker": "citrus_canker"}},
        "excluded_classes": [],
    }

    result = harmonize(records_by_dataset_class, mapping_config)

    mapped_originals = {(m["dataset"], m["original_class"]) for m in result.mapped}
    assert ("DatasetA", "leaf_healthy") in mapped_originals
    assert ("DatasetA", "leaf_canker") in mapped_originals

    unmapped_originals = {(u["dataset"], u["original_class"]) for u in result.unmapped}
    assert ("DatasetA", "mystery_class") in unmapped_originals


def test_harmonize_respects_explicit_exclusion():
    records_by_dataset_class = {("DatasetA", "background"): 5}
    mapping_config = {
        "canonical_classes": ["healthy"],
        "mapping": {},
        "excluded_classes": ["DatasetA:background"],
    }
    result = harmonize(records_by_dataset_class, mapping_config)
    assert len(result.excluded) == 1
    assert result.mapped == []
    assert result.unmapped == []


def test_harmonize_rejects_target_not_in_canonical_list():
    records_by_dataset_class = {("DatasetA", "weird"): 4}
    mapping_config = {
        "canonical_classes": ["healthy"],
        "mapping": {"DatasetA": {"weird": "not_a_real_canonical_class"}},
        "excluded_classes": [],
    }
    result = harmonize(records_by_dataset_class, mapping_config)
    assert len(result.unmapped) == 1
    assert result.mapped == []


def test_suggest_mapping_never_auto_applies_low_confidence_matches():
    records_by_dataset_class = {("DatasetA", "totally_unrelated_xyz"): 1}
    suggestions = suggest_mapping(records_by_dataset_class, canonical_classes=["citrus_canker"], min_ratio=0.9)
    # A near-zero-similarity string should not be force-matched at a high min_ratio.
    assert suggestions["DatasetA"]["totally_unrelated_xyz"] is None


def test_suggest_mapping_matches_close_strings():
    records_by_dataset_class = {("DatasetA", "citrus canker"): 1}
    suggestions = suggest_mapping(records_by_dataset_class, canonical_classes=["citrus_canker"], min_ratio=0.5)
    assert suggestions["DatasetA"]["citrus canker"] == "citrus_canker"

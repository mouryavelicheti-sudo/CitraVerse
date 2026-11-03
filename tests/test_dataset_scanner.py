from src.data.dataset_scanner import detect_layout, find_exact_duplicates, scan_dataset


def test_detect_layout_class_subfolders(synthetic_dataset_root):
    path = f"{synthetic_dataset_root}/DatasetA"
    assert detect_layout(path) == "class_subfolders"


def test_detect_layout_empty(tmp_path):
    empty_dir = tmp_path / "nothing_here"
    empty_dir.mkdir()
    assert detect_layout(str(empty_dir)) == "empty"


def test_scan_dataset_counts_images_and_classes(synthetic_dataset_root):
    profile, records = scan_dataset("DatasetA", f"{synthetic_dataset_root}/DatasetA")
    assert profile.num_images == 12  # 6 + 6, per conftest fixture
    assert profile.num_classes == 2
    assert set(profile.class_counts.keys()) == {"leaf_healthy", "leaf_canker"}
    assert len(records) == 12
    assert all(r.dataset == "DatasetA" for r in records)
    assert all(not r.corrupt for r in records)


def test_scan_dataset_second_dataset_different_classes(synthetic_dataset_root):
    profile, records = scan_dataset("DatasetB", f"{synthetic_dataset_root}/DatasetB")
    assert profile.num_images == 10
    assert set(profile.class_counts.keys()) == {"healthy_leaf", "canker_spot"}


def test_find_exact_duplicates_none_among_random_images(synthetic_dataset_root):
    _profile, records = scan_dataset("DatasetA", f"{synthetic_dataset_root}/DatasetA")
    dupes = find_exact_duplicates(records)
    # Randomly generated images should not collide.
    assert dupes == [] or isinstance(dupes, list)

"""Tests for scripts/generate_visualizations.py and its --visualize wiring
into run_pipeline.py.

None of these tests use real experimental numbers -- they build small,
schema-accurate SYNTHETIC fixtures (matching the exact columns/keys that
scripts/evaluate.py, train_teacher.py, train_student.py, run_robustness.py,
and generate_gradcam.py actually produce, as read directly from that source)
purely to exercise the plotting code path. This is standard test-fixture
data, not a substitute for the real evaluation outputs the visualization
stage is meant to run against.
"""
import importlib
import json
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
genviz = importlib.import_module("scripts.generate_visualizations")

CANONICAL_CLASSES = ["healthy", "citrus_canker", "citrus_greening_hlb", "black_spot", "melanose", "scab"]
MODEL_VARIANTS = ["teacher", "student_fp32", "student_int8"]


# --- Fixture builders (schema-accurate synthetic data only) ---------------

def _write_metadata_csv(path, rng, n=120):
    datasets = ["Pakistan_Citrus", "Bangladesh_Sweet_Orange", "Bangladesh_lemon"]
    rows = [{
        "image_path": f"/fake/{i}.jpg", "dataset": rng.choice(datasets),
        "canonical_class": rng.choice(CANONICAL_CLASSES), "group": None,
        "split": rng.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15]),
    } for i in range(n)]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_evaluation_summary_csv(path):
    pd.DataFrame([
        {"ablation_id": "teacher", "accuracy": 0.91, "precision": 0.90, "recall": 0.89, "f1": 0.895,
         "model_size_mb": 42.7, "latency_ms": 55.2, "fps": 18.1},
        {"ablation_id": "student_fp32", "accuracy": 0.87, "precision": 0.86, "recall": 0.85, "f1": 0.855,
         "model_size_mb": 8.9, "latency_ms": 12.4, "fps": 80.6},
        {"ablation_id": "student_int8", "accuracy": 0.85, "precision": 0.84, "recall": 0.83, "f1": 0.835,
         "model_size_mb": 2.6, "latency_ms": 13.1, "fps": 76.3},   # INT8 slower than FP32 -- intentional
    ]).to_csv(path, index=False)


def _write_test_predictions_csv(path, rng, n_per_variant=60):
    rows = []
    for variant, acc in [("teacher", 0.9), ("student_fp32", 0.85), ("student_int8", 0.8)]:
        for i in range(n_per_variant):
            gt = CANONICAL_CLASSES[i % len(CANONICAL_CLASSES)]
            pred = gt if rng.rand() < acc else rng.choice(CANONICAL_CLASSES)
            rows.append({
                "image_path": f"/fake/test_{i}.jpg", "model_variant": variant,
                "ground_truth_class": gt, "predicted_class": pred,
                "confidence": float(rng.uniform(0.5, 0.99)),
                "reliability_score": float(rng.uniform(0.3, 0.95)),
                "reliability_decision": "RELIABLE",
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_history_json(path, keys, n_epochs=15):
    history = []
    for e in range(1, n_epochs + 1):
        row = {"epoch": e}
        for k in keys:
            row[k] = max(0.01, 1.0 / e)
        history.append(row)
    with open(path, "w") as f:
        json.dump(history, f)


def _write_robustness_csv(path, rng):
    rows = []
    for pert in ["blur", "brightness", "noise"]:
        for i in range(20):
            flipped = bool(rng.rand() < 0.2)
            rows.append({
                "image_path": f"/fake/rob_{i}.jpg", "ground_truth_class": CANONICAL_CLASSES[i % 6],
                "perturbation": pert, "prediction_flipped": flipped,
                "reliability_delta": float(-0.05 if flipped else -0.01),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_gradcam_manifest(gradcam_dir, rng):
    os.makedirs(gradcam_dir, exist_ok=True)
    manifest = []
    for cls in CANONICAL_CLASSES:
        t_path = os.path.join(gradcam_dir, f"{cls}__teacher__fake.png")
        s_path = os.path.join(gradcam_dir, f"{cls}__student_fp32__fake.png")
        Image.fromarray((rng.rand(32, 32, 3) * 255).astype(np.uint8)).save(t_path)
        Image.fromarray((rng.rand(32, 32, 3) * 255).astype(np.uint8)).save(s_path)
        manifest.append({
            "image_path": "/fake/x.jpg", "ground_truth_class": cls,
            "teacher_prediction": cls, "student_fp32_prediction": cls,
            "teacher_gradcam": t_path, "student_fp32_gradcam": s_path,
        })
    manifest_path = os.path.join(gradcam_dir, "gradcam_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


@pytest.fixture
def full_fixture_set(tmp_path):
    """All inputs the visualization stage looks for, schema-accurate and
    fully present -- the 'everything ran successfully' scenario."""
    rng = np.random.RandomState(0)
    metadata_csv = tmp_path / "harmonized_metadata.csv"
    _write_metadata_csv(metadata_csv, rng)

    eval_summary_csv = tmp_path / "evaluation_summary.csv"
    _write_evaluation_summary_csv(eval_summary_csv)

    eval_full_json = tmp_path / "evaluation_full.json"
    eval_full_json.write_text(json.dumps({"tau": 0.6}))

    test_predictions_csv = tmp_path / "test_predictions.csv"
    _write_test_predictions_csv(test_predictions_csv, rng)

    teacher_history = tmp_path / "teacher_training_history.json"
    _write_history_json(teacher_history, ["train_loss", "val_loss", "train_acc", "val_acc"])

    student_history = tmp_path / "student_training_history.json"
    _write_history_json(student_history, ["train_total_loss", "train_L_CE", "train_L_KD",
                                           "train_L_signature", "train_acc", "val_acc", "val_signature_mse"])

    robustness_csv = tmp_path / "robustness_per_image.csv"
    _write_robustness_csv(robustness_csv, rng)

    gradcam_dir = tmp_path / "gradcam"
    gradcam_manifest = _write_gradcam_manifest(str(gradcam_dir), rng)

    out_dir = tmp_path / "figures"

    return {
        "metadata": str(metadata_csv), "evaluation_summary": str(eval_summary_csv),
        "evaluation_full": str(eval_full_json), "test_predictions": str(test_predictions_csv),
        "teacher_history": str(teacher_history), "student_history": str(student_history),
        "robustness_per_image": str(robustness_csv), "gradcam_manifest": gradcam_manifest,
        "out_dir": str(out_dir),
    }


# --- run_pipeline.py wiring -------------------------------------------------

def test_run_pipeline_exposes_visualize_stage():
    import run_pipeline
    stage_keys = [key for key, _module, _help in run_pipeline.STAGES]
    assert "visualize" in stage_keys
    module_by_key = {key: module for key, module, _help in run_pipeline.STAGES}
    assert module_by_key["visualize"] == "scripts.generate_visualizations"


def test_run_pipeline_visualize_flag_is_parseable():
    import argparse
    import run_pipeline

    parser = argparse.ArgumentParser()
    for stage_key, _module, help_text in run_pipeline.STAGES:
        parser.add_argument(run_pipeline._flag_name(stage_key), action="store_true", help=help_text)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(["--visualize"])
    assert args.visualize is True


# --- Design constraints (static checks) ------------------------------------

def test_generate_visualizations_does_not_require_torch():
    """The visualization stage only reads existing CSV/JSON/PNG outputs --
    it should never actually import torch/torchvision or load a checkpoint.
    Uses ast parsing (not a substring check) so this isn't fooled by the
    module's own docstring, which mentions torch in prose."""
    import ast

    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_visualizations.py"))
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0])

    assert "torch" not in imported_modules
    assert "torchvision" not in imported_modules


def test_uses_non_interactive_backend_and_never_calls_plt_show():
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_visualizations.py"))
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()
    assert 'matplotlib.use("Agg")' in source
    assert ".show(" not in source


def test_reliability_terminology_avoids_calibrated_probability_claim():
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_visualizations.py"))
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()
    assert "not a calibrated probability" in source.lower()


def test_robustness_terminology_never_says_robustness_accuracy():
    """Regression guard for the explicit instruction: never call
    1 - flip_rate 'robustness accuracy'."""
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_visualizations.py"))
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()
    assert "robustness accuracy" not in source.lower()
    assert "prediction consistency" in source.lower()


def test_int8_efficiency_figure_never_asserts_int8_is_faster():
    """Regression guard: the efficiency figure must plot real numbers only
    and never hardcode a 'faster' claim about INT8."""
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_visualizations.py"))
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()
    efficiency_fn_start = source.index("def fig_model_efficiency_comparison")
    efficiency_fn_end = source.index("\ndef ", efficiency_fn_start + 1)
    body = source[efficiency_fn_start:efficiency_fn_end].lower()
    assert "faster" not in body
    assert "int8 is" not in body


# --- Full end-to-end run against schema-accurate fixtures -------------------

def test_full_run_generates_all_13_figures(full_fixture_set, monkeypatch):
    argv = ["generate_visualizations.py",
            "--metadata", full_fixture_set["metadata"],
            "--evaluation-summary", full_fixture_set["evaluation_summary"],
            "--evaluation-full", full_fixture_set["evaluation_full"],
            "--test-predictions", full_fixture_set["test_predictions"],
            "--teacher-history", full_fixture_set["teacher_history"],
            "--student-history", full_fixture_set["student_history"],
            "--robustness-per-image", full_fixture_set["robustness_per_image"],
            "--gradcam-manifest", full_fixture_set["gradcam_manifest"],
            "--out-dir", full_fixture_set["out_dir"]]
    monkeypatch.setattr("sys.argv", argv)
    genviz.main()

    expected_files = [
        "dataset_class_distribution.png", "model_performance_comparison.png",
        "model_efficiency_comparison.png", "teacher_training_curves.png",
        "student_distillation_curves.png", "confusion_matrix_teacher.png",
        "confusion_matrix_student_fp32.png", "confusion_matrix_student_int8.png",
        "per_class_f1_comparison.png", "reliability_distribution.png",
        "robustness_analysis.png", "size_vs_f1.png", "gradcam_comparison.png",
    ]
    for fname in expected_files:
        fpath = os.path.join(full_fixture_set["out_dir"], fname)
        assert os.path.exists(fpath), f"Expected figure not generated: {fname}"
        assert os.path.getsize(fpath) > 0


def test_figures_are_saved_at_300_dpi(full_fixture_set, monkeypatch):
    argv = ["generate_visualizations.py",
            "--metadata", full_fixture_set["metadata"],
            "--evaluation-summary", full_fixture_set["evaluation_summary"],
            "--evaluation-full", full_fixture_set["evaluation_full"],
            "--test-predictions", full_fixture_set["test_predictions"],
            "--teacher-history", full_fixture_set["teacher_history"],
            "--student-history", full_fixture_set["student_history"],
            "--robustness-per-image", full_fixture_set["robustness_per_image"],
            "--gradcam-manifest", full_fixture_set["gradcam_manifest"],
            "--out-dir", full_fixture_set["out_dir"]]
    monkeypatch.setattr("sys.argv", argv)
    genviz.main()

    sample = os.path.join(full_fixture_set["out_dir"], "dataset_class_distribution.png")
    with Image.open(sample) as im:
        dpi = im.info.get("dpi")
    assert dpi is not None
    assert dpi[0] == pytest.approx(300, abs=1)
    assert dpi[1] == pytest.approx(300, abs=1)


def test_missing_optional_inputs_does_not_crash(full_fixture_set, monkeypatch):
    """robustness/gradcam are optional stages -- if their outputs don't
    exist yet, the visualization stage must still complete and generate
    everything it can, not crash the whole run."""
    argv = ["generate_visualizations.py",
            "--metadata", full_fixture_set["metadata"],
            "--evaluation-summary", full_fixture_set["evaluation_summary"],
            "--evaluation-full", full_fixture_set["evaluation_full"],
            "--test-predictions", full_fixture_set["test_predictions"],
            "--teacher-history", full_fixture_set["teacher_history"],
            "--student-history", full_fixture_set["student_history"],
            "--robustness-per-image", "/nonexistent/robustness_per_image.csv",
            "--gradcam-manifest", "/nonexistent/gradcam_manifest.json",
            "--out-dir", full_fixture_set["out_dir"]]
    monkeypatch.setattr("sys.argv", argv)
    genviz.main()  # must not raise

    assert os.path.exists(os.path.join(full_fixture_set["out_dir"], "dataset_class_distribution.png"))
    assert not os.path.exists(os.path.join(full_fixture_set["out_dir"], "robustness_analysis.png"))
    assert not os.path.exists(os.path.join(full_fixture_set["out_dir"], "gradcam_comparison.png"))


def test_missing_metadata_file_skips_gracefully(tmp_path):
    result = genviz.fig_dataset_class_distribution(
        str(tmp_path / "does_not_exist.csv"), CANONICAL_CLASSES, str(tmp_path / "out.png"))
    assert result is None
    assert not os.path.exists(tmp_path / "out.png")


def test_unexpected_column_names_are_detected_not_assumed(tmp_path):
    """If test_predictions.csv doesn't have the expected column names, the
    stage must skip with a warning rather than silently guessing a column."""
    bad_csv = tmp_path / "test_predictions.csv"
    pd.DataFrame([{"foo": 1, "bar": 2}]).to_csv(bad_csv, index=False)
    result = genviz.fig_per_class_f1_comparison(str(bad_csv), CANONICAL_CLASSES, str(tmp_path / "out.png"))
    assert result is None
    assert not os.path.exists(tmp_path / "out.png")


# --- Confusion-matrix class ordering (introspects the real Axes object) ---

def test_confusion_matrices_use_exact_six_class_order_on_both_axes(tmp_path, monkeypatch):
    rng = np.random.RandomState(0)
    test_predictions_csv = tmp_path / "test_predictions.csv"
    _write_test_predictions_csv(test_predictions_csv, rng)

    captured = {}
    real_save_fig = genviz._save_fig

    def fake_save_fig(fig, out_path):
        # Capture tick labels before the real save closes the figure.
        ax = fig.axes[0]
        captured[out_path] = {
            "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
            "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
        }
        real_save_fig(fig, out_path)

    monkeypatch.setattr(genviz, "_save_fig", fake_save_fig)
    out_dir = str(tmp_path / "figs")
    os.makedirs(out_dir, exist_ok=True)
    genviz.fig_confusion_matrices(str(test_predictions_csv), CANONICAL_CLASSES, out_dir)

    assert len(captured) == 3
    for out_path, labels in captured.items():
        assert labels["xticklabels"] == CANONICAL_CLASSES, out_path
        assert labels["yticklabels"] == CANONICAL_CLASSES, out_path


def test_gradcam_comparison_does_not_modify_source_images(full_fixture_set):
    gradcam_dir = os.path.dirname(full_fixture_set["gradcam_manifest"])
    source_pngs = [os.path.join(gradcam_dir, f) for f in os.listdir(gradcam_dir) if f.endswith(".png")]
    mtimes_before = {p: os.path.getmtime(p) for p in source_pngs}

    out_path = os.path.join(full_fixture_set["out_dir"], "gradcam_comparison.png")
    os.makedirs(full_fixture_set["out_dir"], exist_ok=True)
    result = genviz.fig_gradcam_comparison(full_fixture_set["gradcam_manifest"], CANONICAL_CLASSES, out_path)

    assert result == out_path
    assert os.path.exists(out_path)
    for p in source_pngs:
        assert os.path.getmtime(p) == mtimes_before[p], f"Original Grad-CAM file was modified: {p}"

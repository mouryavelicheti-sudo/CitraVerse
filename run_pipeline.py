#!/usr/bin/env python3
"""
AgriVerse / CitraVerse - one-command pipeline entry point.

This does not reimplement any stage -- it just calls the existing
scripts/*.py `main()` functions in order, in-process, so you get one
command instead of remembering nine. Every stage can still be run directly
via `python scripts/<name>.py` with its own flags if you need more control.

Usage (run from the project root):
    python run_pipeline.py --profile
    python run_pipeline.py --preprocess
    python run_pipeline.py --extract-evidence
    python run_pipeline.py --build-prototypes
    python run_pipeline.py --train-teacher
    python run_pipeline.py --train-student
    python run_pipeline.py --quantize
    python run_pipeline.py --robustness
    python run_pipeline.py --evaluate
    python run_pipeline.py --explain
    python run_pipeline.py --visualize
    python run_pipeline.py --export-dataset
    python run_pipeline.py --all              # runs every stage above, in order

Multiple flags can be combined, e.g.:
    python run_pipeline.py --train-teacher --train-student --quantize --evaluate
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

STAGES = [
    ("profile", "scripts.profile_datasets", "Module I: dataset inspection & profiling"),
    ("preprocess", "scripts.harmonize_datasets", "Module I: harmonization + dedup + split"),
    ("extract_evidence", "scripts.extract_evidence", "Module II: multi-evidence extraction + fusion"),
    ("build_prototypes", "scripts.build_prototypes", "Module III: prototypes + DESS"),
    ("train_teacher", "scripts.train_teacher", "Module IV: train FP32 teacher"),
    ("train_student", "scripts.train_student", "Module V: evidence-preserving distillation"),
    ("quantize", "scripts.quantize_model", "Module V: INT8 quantization + benchmark"),
    ("robustness", "scripts.run_robustness", "Module III/IV: perturbation robustness experiment"),
    ("evaluate", "scripts.evaluate", "Module IV/V: full evaluation (metrics, DESS, reliability, ablation)"),
    ("explain", "scripts.generate_gradcam", "Explainability: Grad-CAM (teacher vs student)"),
    ("visualize", "scripts.generate_visualizations", "Visualization: publication-quality figures from existing outputs"),
    ("export_dataset", "scripts.export_dataset", "Export harmonized local datasets with provenance"),
]


def _flag_name(stage_key: str) -> str:
    return "--" + stage_key.replace("_", "-")


def main():
    parser = argparse.ArgumentParser(
        description="AgriVerse one-command pipeline. Run stages individually or with --all.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for stage_key, _module, help_text in STAGES:
        parser.add_argument(_flag_name(stage_key), action="store_true", help=help_text)
    parser.add_argument("--all", action="store_true", help="Run every stage above, in order.")
    args = parser.parse_args()

    requested = [key for key, _m, _h in STAGES if args.all or getattr(args, key)]
    if not requested:
        parser.print_help()
        print("\nNo stage selected. Pass one or more flags above, or --all.")
        return

    print("=" * 70)
    print(f"AgriVerse pipeline: running {len(requested)} stage(s): {', '.join(requested)}")
    print("=" * 70)

    original_argv = sys.argv
    for stage_key, module_name, help_text in STAGES:
        if stage_key not in requested:
            continue

        print(f"\n{'#' * 70}\n# STAGE: {stage_key}  --  {help_text}\n{'#' * 70}\n")
        sys.argv = [module_name]  # each stage's own argparse only sees its own defaults here
        start = time.time()
        try:
            module = __import__(module_name, fromlist=["main"])
            module.main()
        except Exception as e:
            print(f"\n[run_pipeline.py] STAGE '{stage_key}' FAILED: {e}")
            print("Stopping pipeline. Fix the error above and re-run "
                  f"(you can resume from just this stage with: python {_flag_name(stage_key)[2:].replace('-', '_')}"
                  f" -> python -m {module_name}).")
            sys.argv = original_argv
            sys.exit(1)
        finally:
            sys.argv = original_argv
        print(f"\n[run_pipeline.py] Stage '{stage_key}' finished in {time.time() - start:.1f}s.")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

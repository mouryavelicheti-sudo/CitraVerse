#!/usr/bin/env python3
"""
MODULE V: quantize the trained FP32 student to INT8 and benchmark it
against the FP32 student.

    Student FP32 --> INT8 Quantization --> INT8 model --> Evaluation

Two quantization paths are supported (see configs/edge.yaml -> quantization.method):
  - "dynamic": works out-of-the-box on any CPU. IMPORTANT CAVEAT (explicitly
    called out here, not glossed over): torch.quantization.quantize_dynamic
    only converts the layer types you pass it (this project passes
    {nn.Linear}). For a CNN backbone, that means the convolutional feature
    extractor STAYS FP32 -- only the final Linear classifier head(s) become
    INT8. This script reports, per-layer, which modules actually ended up
    quantized, so "edge-optimized INT8 model" is a claim you can back with
    evidence rather than one that assumes the whole network was quantized.
  - "static": calibration-based, quantizes Conv2d + Linear via fused
    qconfig, giving a genuinely mostly-INT8 CNN, at the cost of needing a
    representative calibration set (drawn from the train split here) and a
    fixed, non-differentiable fusion of ops. Prefer this for the "true
    edge-optimized INT8 model" research claim; use "dynamic" only when you
    explicitly want the fast/no-calibration baseline for comparison.

Saves:
    models/student_int8.pt            (torch.save of the full quantized module --
                                        quantized models change architecture,
                                        so the state_dict-only pattern used for
                                        FP32 checkpoints does not apply here)
    outputs/metrics/quantization_report.json

Usage:
    python run_pipeline.py --quantize
    python scripts/quantize_model.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io_utils import load_yaml, resolve_path, save_json
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("quantize_model")


def _quantized_layer_report(model) -> dict:
    """Inspects the actual resulting module tree and reports which layer
    types ended up as quantized (torch.nn.quantized.*) vs. left FP32, so we
    never claim "INT8 model" without evidence of what was really converted.
    """
    counts = {}
    for module in model.modules():
        cls_name = module.__class__.__name__
        is_quantized = "quantized" in module.__class__.__module__
        key = f"{cls_name} ({'INT8' if is_quantized else 'FP32'})"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _is_fully_int8_cnn(layer_report: dict) -> bool:
    """True ONLY if every Conv/Linear-bearing layer in the report ended up
    quantized -- i.e. there are zero remaining FP32 Conv2d/Linear entries.
    This is the single source of truth for whether "edge-optimized INT8
    model" is an accurate claim; nothing in this script asserts that
    claim without checking this first (see requirement: never claim the
    entire CNN is INT8 unless it actually is).
    """
    for key, count in layer_report.items():
        if count == 0:
            continue
        is_weighted_layer = ("Conv" in key or "Linear" in key)
        is_fp32 = "FP32" in key
        if is_weighted_layer and is_fp32:
            return False
    return True


def main():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error("PyTorch is required for quantization. Install with: pip install torch torchvision")
        return

    from src.data.image_dataset import AgriVerseImageDataset, agriverse_collate, build_transforms
    from src.models.classifier import DiseaseClassifier
    from src.quantization.quantize import (
        benchmark_latency,
        dynamic_quantize,
        model_size_mb,
        resolve_quantized_backend,
        static_quantize,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--edge-config", default="configs/edge.yaml")
    parser.add_argument("--class-mapping-config", default="configs/class_mapping.yaml")
    parser.add_argument("--metadata", default="outputs/dataset_profiles/harmonized_metadata.csv")
    parser.add_argument("--student-checkpoint", default="models/student_best.pt")
    args = parser.parse_args()

    train_cfg = load_yaml(args.training_config)
    edge_cfg = load_yaml(args.edge_config)
    set_seed(train_cfg.get("seed", 42))

    student_ckpt_path = resolve_path(args.student_checkpoint)
    if not os.path.exists(student_ckpt_path):
        logger.error(f"{student_ckpt_path} not found. Run scripts/train_student.py first.")
        return

    canonical_classes = load_yaml(args.class_mapping_config).get("canonical_classes", [])
    num_classes = len(canonical_classes)
    student_backbone = train_cfg.get("student", {}).get("backbone", "mobilenet_v3_small")

    student_fp32 = DiseaseClassifier.build(student_backbone, num_classes, pretrained=False)
    student_fp32.load_state_dict(torch.load(student_ckpt_path, map_location="cpu"))
    student_fp32.eval()

    # DiseaseClassifier attaches a penultimate-feature extraction hook to the
    # backbone (see src/models/classifier.py::_get_feature_extractor). That
    # hook is a LOCAL function (`_get_feature_extractor.<locals>.hook`),
    # needed only for classification/distillation, where DiseaseClassifier's
    # own forward()/__call__ reads `self._features["feat"]`. Quantization and
    # benchmarking below never go through that wrapper -- they call
    # `student_fp32.net(...)` directly -- so the hook serves no purpose here.
    # Worse, if left attached it gets carried onto `quantized_net` (both
    # dynamic_quantize/static_quantize internally deep-copy the module before
    # converting it), and the later `torch.save(quantized_net, int8_path)`
    # tries to pickle that local closure and fails with:
    #   Can't get local object '_get_feature_extractor.<locals>.hook'
    # Detach it now, before quantization, since it is not needed again in
    # this script. This does not touch the trained weights/checkpoint.
    if getattr(student_fp32, "_hook_handle", None) is not None:
        student_fp32._hook_handle.remove()

    quant_cfg = edge_cfg.get("quantization", {})
    method = quant_cfg.get("method", "dynamic")
    backend = quant_cfg.get("backend", "qnnpack")
    image_size = train_cfg.get("image_size", 224)

    # Resolve the CONFIGURED backend against what this PyTorch installation
    # actually supports (torch.backends.quantized.supported_engines varies
    # by platform/build -- many Windows/x86 CPU wheels only support
    # "onednn" and have no "qnnpack"/"fbgemm" at all). dynamic_quantize()/
    # static_quantize() below do this same resolution internally before
    # they touch torch.backends.quantized.engine, so this call is purely
    # for accurate logging/reporting -- it never silently overrides what
    # those functions actually do.
    resolved_backend = resolve_quantized_backend(backend)
    if resolved_backend != backend:
        logger.warning(
            f"Configured backend '{backend}' (configs/edge.yaml) is not supported by this "
            f"PyTorch build. Supported engines: {list(torch.backends.quantized.supported_engines)}. "
            f"Falling back to '{resolved_backend}' for this run."
        )

    if method == "static":
        metadata_path = resolve_path(args.metadata)
        if not os.path.exists(metadata_path):
            logger.error(f"{metadata_path} not found (needed for static-quantization calibration data).")
            return
        class_to_idx = {c: i for i, c in enumerate(canonical_classes)}
        calib_ds = AgriVerseImageDataset(metadata_path, "train", class_to_idx, build_transforms(image_size, train=False))
        calib_loader = DataLoader(calib_ds, batch_size=train_cfg.get("batch_size", 32), shuffle=True,
                                   collate_fn=agriverse_collate)
        logger.info(f"Static quantization: calibrating on up to {quant_cfg.get('calibration_batches', 20)} train batches.")
        quantized_net = static_quantize(student_fp32.net, calib_loader, backend=backend,
                                         num_calibration_batches=quant_cfg.get("calibration_batches", 20))
    elif method == "dynamic":
        quantized_net = dynamic_quantize(student_fp32.net, backend=backend)
    else:
        logger.error(f"Unknown quantization method in configs/edge.yaml: {method} (expected dynamic|static)")
        return

    # --- Benchmark FP32 vs INT8: size + latency, on real forward passes. ---
    bench_cfg = edge_cfg.get("benchmark", {})
    dummy_input = torch.randn(bench_cfg.get("batch_size_for_latency", 1), 3, image_size, image_size)

    fp32_size = model_size_mb(student_fp32.net)
    int8_size = model_size_mb(quantized_net)

    fp32_latency = benchmark_latency(student_fp32.net.cpu(), dummy_input,
                                      num_warmup=bench_cfg.get("num_warmup_runs", 5),
                                      num_timed=bench_cfg.get("num_timed_runs", 30))
    int8_latency = benchmark_latency(quantized_net, dummy_input,
                                      num_warmup=bench_cfg.get("num_warmup_runs", 5),
                                      num_timed=bench_cfg.get("num_timed_runs", 30))

    layer_report = _quantized_layer_report(quantized_net)
    fully_int8_cnn = _is_fully_int8_cnn(layer_report)

    ckpt_dir = resolve_path(train_cfg.get("checkpoint_dir", "models"))
    os.makedirs(ckpt_dir, exist_ok=True)
    int8_path = os.path.join(ckpt_dir, "student_int8.pt")
    torch.save(quantized_net, int8_path)  # full-module save: quantized architecture differs from FP32 state_dict

    report = {
        "method": method,
        "requested_backend": backend,
        "actual_backend_used": resolved_backend,
        "supported_engines_on_this_machine": list(torch.backends.quantized.supported_engines),
        "fp32_model_size_mb": fp32_size,
        "int8_model_size_mb": int8_size,
        "size_reduction_pct": 100.0 * (1 - int8_size / fp32_size) if fp32_size > 0 else None,
        "fp32_latency": fp32_latency,
        "int8_latency": int8_latency,
        "speedup_x": (fp32_latency["mean_latency_ms"] / int8_latency["mean_latency_ms"])
        if int8_latency["mean_latency_ms"] > 0 else None,
        "quantized_layer_report": layer_report,
        # True ONLY if every Conv/Linear layer ended up quantized. This is
        # the field to check before ever writing "fully INT8 CNN" anywhere
        # (paper, README, slides) -- see _is_fully_int8_cnn() above.
        "fully_int8_cnn": fully_int8_cnn,
    }
    save_json(report, "outputs/metrics/quantization_report.json")

    print("\n" + "=" * 70)
    print(f"QUANTIZATION REPORT (method={method}, backend={resolved_backend}"
          + (f", requested='{backend}'" if resolved_backend != backend else "") + ")")
    print("=" * 70)
    print(f"  Model size:  FP32 {fp32_size:.2f} MB  ->  INT8 {int8_size:.2f} MB "
          f"({report['size_reduction_pct']:.1f}% smaller)")
    print(f"  Latency:     FP32 {fp32_latency['mean_latency_ms']:.2f} ms  ->  "
          f"INT8 {int8_latency['mean_latency_ms']:.2f} ms")
    print(f"  Fully INT8 CNN (every Conv/Linear quantized): {fully_int8_cnn}")
    if not fully_int8_cnn:
        n_fp32_weighted = sum(v for k, v in layer_report.items()
                               if ("Conv" in k or "Linear" in k) and "FP32" in k)
        print(f"  NOTE: {n_fp32_weighted} Conv/Linear layer(s) remain FP32 -- do NOT claim a fully "
              f"INT8 CNN. Use method: static in configs/edge.yaml if the paper needs that claim "
              f"to actually hold.")
    print(f"\nSaved: {int8_path}")
    print("Next step:\n  python run_pipeline.py --evaluate")


if __name__ == "__main__":
    main()

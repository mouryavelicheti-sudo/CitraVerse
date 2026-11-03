#!/usr/bin/env python3
"""
Generate Grad-CAM visualizations for representative test images from each
disease class, comparing Teacher FP32 vs Student FP32.

NOTE on INT8: quantized (INT8) models are inference-only -- gradients
cannot flow through them, so Grad-CAM cannot be generated directly on
student_int8.pt. Per src/explainability/gradcam.py's documented approach,
this script generates Grad-CAM on the Student FP32 checkpoint and reports
the INT8 model's prediction/confidence on the SAME image alongside it, so
you can see whether quantization changed the decision even though you
can't visualize INT8 activations directly.

Saves, per representative image:
    outputs/gradcam/<canonical_class>__teacher.png
    outputs/gradcam/<canonical_class>__student_fp32.png
    outputs/gradcam/gradcam_manifest.json   (predictions/confidences for
                                              teacher, student FP32, and
                                              student INT8 on every image)

Usage:
    python run_pipeline.py --explain
    python scripts/generate_gradcam.py --num-per-class 2
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.explainability.gradcam import GradCAM, LAST_CONV_LAYER, overlay_heatmap, resolve_target_layer
from src.utils.io_utils import load_yaml, resolve_path, save_json
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("generate_gradcam")


def main():
    try:
        import torch
        from torchvision import transforms
    except ImportError:
        logger.error("PyTorch is required for Grad-CAM. Install with: pip install torch torchvision")
        return

    from src.models.classifier import DiseaseClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="outputs/dataset_profiles/harmonized_metadata.csv")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--class-mapping-config", default="configs/class_mapping.yaml")
    parser.add_argument("--teacher-checkpoint", default="models/teacher_best.pt")
    parser.add_argument("--student-checkpoint", default="models/student_best.pt")
    parser.add_argument("--student-int8-checkpoint", default="models/student_int8.pt")
    parser.add_argument("--out-dir", default="outputs/gradcam")
    parser.add_argument("--num-per-class", type=int, default=2)
    args = parser.parse_args()

    cfg = load_yaml(args.training_config)
    set_seed(cfg.get("seed", 42))

    metadata_path = resolve_path(args.metadata)
    if not os.path.exists(metadata_path):
        logger.error(f"{metadata_path} not found. Run scripts/harmonize_datasets.py first.")
        return
    df = pd.read_csv(metadata_path)
    test_df = df[df["split"] == "test"]
    if len(test_df) == 0:
        logger.error("No test-split rows in metadata. Cannot pick representative images.")
        return

    canonical_classes = load_yaml(args.class_mapping_config).get("canonical_classes", [])
    class_to_idx = {c: i for i, c in enumerate(canonical_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    image_size = cfg.get("image_size", 224)

    teacher_backbone = cfg.get("teacher", {}).get("backbone", "resnet18")
    student_backbone = cfg.get("student", {}).get("backbone", "mobilenet_v3_small")

    teacher_ckpt = resolve_path(args.teacher_checkpoint)
    student_ckpt = resolve_path(args.student_checkpoint)
    int8_ckpt = resolve_path(args.student_int8_checkpoint)

    if not (os.path.exists(teacher_ckpt) and os.path.exists(student_ckpt)):
        logger.error("Teacher and/or student checkpoint missing. Run train_teacher.py / train_student.py first.")
        return

    teacher = DiseaseClassifier.build(teacher_backbone, len(canonical_classes), pretrained=False)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location="cpu",weights_only=False))
    teacher.eval()

    student = DiseaseClassifier.build(student_backbone, len(canonical_classes), pretrained=False)
    student.load_state_dict(torch.load(student_ckpt, map_location="cpu",weights_only=False))
    student.eval()

    student_int8 = None
    if os.path.exists(int8_ckpt):
        student_int8 = torch.load(int8_ckpt, map_location="cpu",weights_only=False)
        student_int8.eval()
    else:
        logger.warning(f"{int8_ckpt} not found -- INT8 predictions will be omitted from the manifest. "
                        f"Run scripts/quantize_model.py first for the full comparison.")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    out_dir = resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    teacher_cam = GradCAM(teacher.net, resolve_target_layer(teacher.net, teacher_backbone))
    student_cam = GradCAM(student.net, resolve_target_layer(student.net, student_backbone))

    manifest = []
    for canonical_class in canonical_classes:
        class_rows = test_df[test_df["canonical_class"] == canonical_class]
        if len(class_rows) == 0:
            logger.warning(f"No test images for class '{canonical_class}' -- skipping.")
            continue
        sample_rows = class_rows.sample(n=min(args.num_per_class, len(class_rows)), random_state=42)

        for _, row in sample_rows.iterrows():
            img_bgr = cv2.imread(row["image_path"])
            if img_bgr is None:
                logger.warning(f"Could not read {row['image_path']}, skipping.")
                continue
            img_bgr = cv2.resize(img_bgr, (image_size, image_size))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = tfm(img_rgb).unsqueeze(0)

            teacher_cam_map, teacher_pred_idx = teacher_cam.generate(tensor.clone().requires_grad_(True))
            teacher_overlay = overlay_heatmap(img_bgr, teacher_cam_map)
            teacher_out = os.path.join(out_dir, f"{canonical_class}__teacher__{os.path.basename(row['image_path'])}.png")
            cv2.imwrite(teacher_out, teacher_overlay)

            student_cam_map, student_pred_idx = student_cam.generate(tensor.clone().requires_grad_(True))
            student_overlay = overlay_heatmap(img_bgr, student_cam_map)
            student_out = os.path.join(out_dir, f"{canonical_class}__student_fp32__{os.path.basename(row['image_path'])}.png")
            cv2.imwrite(student_out, student_overlay)

            entry = {
                "image_path": row["image_path"],
                "ground_truth_class": canonical_class,
                "teacher_prediction": idx_to_class[teacher_pred_idx],
                "student_fp32_prediction": idx_to_class[student_pred_idx],
                "teacher_gradcam": teacher_out,
                "student_fp32_gradcam": student_out,
            }

            if student_int8 is not None:
                with torch.no_grad():
                    int8_logits = student_int8(tensor)
                    int8_probs = torch.softmax(int8_logits, dim=1)
                    int8_pred_idx = int(int8_probs.argmax(1).item())
                    entry["student_int8_prediction"] = idx_to_class[int8_pred_idx]
                    entry["student_int8_confidence"] = float(int8_probs.max().item())

            manifest.append(entry)
            logger.info(f"Grad-CAM generated for {row['image_path']} "
                        f"(teacher={entry['teacher_prediction']}, student={entry['student_fp32_prediction']})")

    save_json(manifest, os.path.join("outputs/gradcam", "gradcam_manifest.json"))
    print(f"\nGenerated Grad-CAM for {len(manifest)} representative images across "
          f"{len(set(m['ground_truth_class'] for m in manifest))} classes.")
    print(f"Saved to: {out_dir}")
    print("Next step:\n  python run_pipeline.py --export-dataset")


if __name__ == "__main__":
    main()

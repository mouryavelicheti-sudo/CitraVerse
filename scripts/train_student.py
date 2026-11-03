#!/usr/bin/env python3
"""
MODULE V: train the lightweight FP32 student via evidence-preserving
knowledge distillation from the trained teacher.

    Teacher FP32 (frozen)
        |-> teacher_logits
    Student (trainable)
        |-> student_logits, student_penultimate_feat
    Precomputed Z(x) (Module II, outputs/features/Z_signatures.npz)
        |-> z_target  (plays the role of Z_T(x) -- see
                        src/distillation/losses.py for why)

    L_student = lambda1 * L_CE(student_logits, labels)
              + lambda2 * L_KD(student_logits, teacher_logits, T)
              + lambda3 * L_signature(signature_proj(student_feat), z_target)

Saves:
    models/student_best.pt              (student net state_dict)
    models/signature_proj_best.pt       (student->Z(x) projection head)
    outputs/metrics/student_training_history.json
    outputs/metrics/student_summary.json

Usage:
    python run_pipeline.py --train-student
    python scripts/train_student.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io_utils import load_yaml, resolve_path, save_json
from src.utils.logging_utils import get_logger
from src.utils.seed import get_device, set_seed

logger = get_logger("train_student")


def _load_z_lookup(evidence_index_path: str, z_signatures_path: str) -> dict:
    """Returns {image_path: np.ndarray Z(x)} built from Module II outputs."""
    df = pd.read_csv(evidence_index_path)
    z = np.load(z_signatures_path)["Z"]
    if len(df) != len(z):
        raise ValueError(
            f"{evidence_index_path} ({len(df)} rows) and {z_signatures_path} "
            f"({len(z)} rows) are out of sync. Re-run scripts/extract_evidence.py."
        )
    return {row["image_path"]: z[i] for i, row in df.iterrows()}


def main():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error(
            "PyTorch is required to train the student model. "
            "Install with: pip install torch torchvision"
        )
        return

    from src.data.image_dataset import AgriVerseImageDataset, agriverse_collate, build_transforms
    from src.distillation.losses import build_distillation_loss
    from src.models.classifier import DiseaseClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="outputs/dataset_profiles/harmonized_metadata.csv")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--class-mapping-config", default="configs/class_mapping.yaml")
    parser.add_argument("--evidence-index", default="outputs/features/evidence_index.csv")
    parser.add_argument("--z-signatures", default="outputs/features/Z_signatures.npz")
    parser.add_argument("--teacher-checkpoint", default="models/teacher_best.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    metadata_path = resolve_path(args.metadata)
    teacher_ckpt_path = resolve_path(args.teacher_checkpoint)
    evidence_index_path = resolve_path(args.evidence_index)
    z_signatures_path = resolve_path(args.z_signatures)

    for required, hint in [
        (metadata_path, "Run scripts/harmonize_datasets.py first."),
        (teacher_ckpt_path, "Run scripts/train_teacher.py first."),
        (evidence_index_path, "Run scripts/extract_evidence.py first."),
        (z_signatures_path, "Run scripts/extract_evidence.py first."),
    ]:
        if not os.path.exists(required):
            logger.error(f"{required} not found. {hint}")
            return

    canonical_classes = load_yaml(args.class_mapping_config).get("canonical_classes", [])
    class_to_idx = {c: i for i, c in enumerate(canonical_classes)}
    num_classes = len(canonical_classes)

    image_size = cfg.get("image_size", 224)
    train_ds = AgriVerseImageDataset(metadata_path, "train", class_to_idx, build_transforms(image_size, train=True))
    val_ds = AgriVerseImageDataset(metadata_path, "val", class_to_idx, build_transforms(image_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32), shuffle=True,
                               num_workers=cfg.get("num_workers", 2), collate_fn=agriverse_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 32), shuffle=False,
                             num_workers=cfg.get("num_workers", 2), collate_fn=agriverse_collate)

    z_lookup = _load_z_lookup(evidence_index_path, z_signatures_path)
    z_dim = next(iter(z_lookup.values())).shape[0]
    logger.info(f"Loaded Z(x) evidence signatures for {len(z_lookup)} images (Z_dim={z_dim}).")

    # --- Teacher: load trained weights, freeze, eval mode only. ---
    teacher_cfg = cfg.get("teacher", {})
    teacher = DiseaseClassifier.build(teacher_cfg.get("backbone", "resnet18"), num_classes,
                                       pretrained=False).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt_path, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Student: trainable, lightweight backbone. ---
    student_cfg = cfg.get("student", {})
    student = DiseaseClassifier.build(student_cfg.get("backbone", "mobilenet_v3_small"), num_classes,
                                       pretrained=student_cfg.get("pretrained", True)).to(device)

    student_feat_dim = student.net.evidence_feature_dim
    signature_proj = nn.Linear(student_feat_dim, z_dim).to(device)

    dist_cfg = cfg.get("distillation", {})
    loss_fn = build_distillation_loss(
        lambda_ce=dist_cfg.get("lambda_ce", 0.4),
        lambda_kd=dist_cfg.get("lambda_kd", 0.4),
        lambda_signature=dist_cfg.get("lambda_signature", 0.2),
        temperature=dist_cfg.get("temperature", 4.0),
    )

    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(signature_proj.parameters()),
        lr=student_cfg.get("lr", 5e-4), weight_decay=student_cfg.get("weight_decay", 1e-4),
    )

    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "models"))
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_acc = -1.0
    history = []
    epochs = student_cfg.get("epochs", 15)

    for epoch in range(1, epochs + 1):
        student.train()
        running = {"total": 0.0, "L_CE": 0.0, "L_KD": 0.0, "L_signature": 0.0}
        correct, total = 0, 0

        for images, labels, meta in train_loader:
            images, labels = images.to(device), labels.to(device)

            z_batch = np.stack([z_lookup[p] for p in meta["image_path"]])
            z_batch = torch.from_numpy(z_batch).float().to(device)

            with torch.no_grad():
                teacher_logits, _teacher_probs, _teacher_feat = teacher(images)

            optimizer.zero_grad()
            student_logits, student_probs, student_feat = student(images)

            losses = loss_fn(student_logits, teacher_logits, student_feat, z_batch, labels, signature_proj)
            losses["total"].backward()
            optimizer.step()

            bs = images.size(0)
            for k in ("L_CE", "L_KD", "L_signature"):
                running[k] += losses[k] * bs
            running["total"] += losses["total"].item() * bs
            correct += (student_probs.argmax(1) == labels).sum().item()
            total += bs

        train_acc = correct / max(total, 1)
        for k in running:
            running[k] /= max(total, 1)

        val_acc, val_signature_mse = _evaluate(student, teacher, signature_proj, val_loader, z_lookup, device)

        logger.info(
            f"[Student] Epoch {epoch}/{epochs} - "
            f"train_loss={running['total']:.4f} (CE={running['L_CE']:.4f} "
            f"KD={running['L_KD']:.4f} sig={running['L_signature']:.4f}) "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_sig_mse={val_signature_mse:.4f}"
        )
        history.append({
            "epoch": epoch, "train_total_loss": running["total"], "train_L_CE": running["L_CE"],
            "train_L_KD": running["L_KD"], "train_L_signature": running["L_signature"],
            "train_acc": train_acc, "val_acc": val_acc, "val_signature_mse": val_signature_mse,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), os.path.join(ckpt_dir, "student_best.pt"))
            torch.save(signature_proj.state_dict(), os.path.join(ckpt_dir, "signature_proj_best.pt"))
            logger.info(f"  -> New best student checkpoint saved (val_acc={val_acc:.4f})")

    save_json(history, "outputs/metrics/student_training_history.json")
    save_json({
        "teacher_backbone": teacher_cfg.get("backbone"),
        "student_backbone": student_cfg.get("backbone"),
        "num_classes": num_classes,
        "z_dim": int(z_dim),
        "best_val_acc": best_val_acc,
        "distillation": dist_cfg,
    }, "outputs/metrics/student_summary.json")

    print(f"\nBest student validation accuracy: {best_val_acc:.4f}")
    print("Checkpoints: models/student_best.pt, models/signature_proj_best.pt")
    print("Next step:\n  python run_pipeline.py --quantize")


def _evaluate(student, teacher, signature_proj, loader, z_lookup, device):
    import torch
    import numpy as np

    student.eval()
    correct, total = 0, 0
    sig_errs = []
    with torch.no_grad():
        for images, labels, meta in loader:
            images, labels = images.to(device), labels.to(device)
            z_batch = np.stack([z_lookup[p] for p in meta["image_path"]])
            z_batch = torch.from_numpy(z_batch).float().to(device)

            _t_logits, _t_probs, _t_feat = teacher(images)
            _s_logits, s_probs, s_feat = student(images)

            z_hat = signature_proj(s_feat)
            sig_errs.append(torch.mean((z_hat - z_batch) ** 2).item())
            correct += (s_probs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return correct / max(total, 1), float(np.mean(sig_errs)) if sig_errs else 0.0


if __name__ == "__main__":
    main()

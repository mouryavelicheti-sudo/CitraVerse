#!/usr/bin/env python3
"""
MODULE V (teacher side) / MODULE IV: train the FP32 teacher disease classifier.

Usage:
    python run_pipeline.py --train-teacher
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io_utils import load_yaml, resolve_path, save_json
from src.utils.logging_utils import get_logger
from src.utils.seed import get_device, set_seed

logger = get_logger("train_teacher")


def main():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error(
            "PyTorch is required to train the teacher model. "
            "Install with: pip install torch torchvision"
        )
        return

    from src.data.image_dataset import AgriVerseImageDataset, agriverse_collate, build_transforms
    from src.models.classifier import DiseaseClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="outputs/dataset_profiles/harmonized_metadata.csv")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--class-mapping-config", default="configs/class_mapping.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    metadata_path = resolve_path(args.metadata)
    if not os.path.exists(metadata_path):
        logger.error(f"{metadata_path} not found. Run harmonize_datasets.py first.")
        return

    canonical_classes = load_yaml(args.class_mapping_config).get("canonical_classes", [])
    class_to_idx = {c: i for i, c in enumerate(canonical_classes)}

    image_size = cfg.get("image_size", 224)
    train_ds = AgriVerseImageDataset(metadata_path, "train", class_to_idx, build_transforms(image_size, train=True))
    val_ds = AgriVerseImageDataset(metadata_path, "val", class_to_idx, build_transforms(image_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32), shuffle=True,
                               num_workers=cfg.get("num_workers", 2), collate_fn=agriverse_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 32), shuffle=False,
                             num_workers=cfg.get("num_workers", 2), collate_fn=agriverse_collate)

    teacher_cfg = cfg.get("teacher", {})
    model = DiseaseClassifier.build(teacher_cfg.get("backbone", "resnet18"), len(canonical_classes),
                                     pretrained=teacher_cfg.get("pretrained", True)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=teacher_cfg.get("lr", 3e-4),
                                  weight_decay=teacher_cfg.get("weight_decay", 1e-4))
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "models"))
    os.makedirs(ckpt_dir, exist_ok=True)
    history = []

    epochs = teacher_cfg.get("epochs", 15)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels, _meta in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, probs, _feat = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (probs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        val_acc, val_loss = _evaluate(model, val_loader, criterion, device)
        train_acc = train_correct / max(train_total, 1)
        train_loss /= max(train_total, 1)

        logger.info(f"[Teacher] Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} "
                    f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "teacher_best.pt"))
            logger.info(f"  -> New best teacher checkpoint saved (val_acc={val_acc:.4f})")

    save_json(history, "outputs/metrics/teacher_training_history.json")
    save_json({"backbone": teacher_cfg.get("backbone"), "num_classes": len(canonical_classes),
               "best_val_acc": best_val_acc}, "outputs/metrics/teacher_summary.json")
    print(f"\nBest teacher validation accuracy: {best_val_acc:.4f}")
    print("Checkpoint: models/teacher_best.pt")
    print("Next step:\n  python run_pipeline.py --train-student")


def _evaluate(model, loader, criterion, device):
    import torch
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels, _meta in loader:
            images, labels = images.to(device), labels.to(device)
            logits, probs, _feat = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (probs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return correct / max(total, 1), total_loss / max(total, 1)


if __name__ == "__main__":
    main()

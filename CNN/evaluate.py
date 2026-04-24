"""
evaluate.py — Evaluate a trained checkpoint on the validation (or test) split.

Outputs:
  • Overall top-1 / top-5 accuracy
  • Per-class accuracy, precision, recall, F1
  • Confusion matrix (saved as PNG)
  • Classification report (saved as CSV)

Usage:
  python evaluate.py --checkpoint checkpoints/best_model.pt
  python evaluate.py --checkpoint checkpoints/best_model.pt --split test
  python evaluate.py --checkpoint checkpoints/best_model.pt --split test --use-hf
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

import config
from dataset import MTSDLocalDataset, MTSDHuggingFaceDataset, build_transforms, get_datasets
from model import load_checkpoint


# ─── Inference pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """Return (all_preds, all_labels) arrays."""
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_topk_accuracy(model, loader, device, k=5):
    model.eval()
    correct1 = correct5 = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            _, top5 = logits.topk(k, dim=1)
            correct1 += (top5[:, 0] == labels).sum().item()
            correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total    += labels.size(0)

    return correct1 / total, correct5 / total


def plot_confusion_matrix(cm, class_names, out_path, max_classes=50):
    """Plot a condensed confusion matrix (top-N classes by frequency)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not available — skipping confusion matrix plot.")
        return

    # Restrict to most-frequent classes for readability
    freq   = cm.sum(axis=1)
    top_idx = np.argsort(freq)[-max_classes:][::-1]
    cm_top  = cm[np.ix_(top_idx, top_idx)]
    names   = [class_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, 12))
    img = ax.imshow(cm_top, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(img, ax=ax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (top-{max_classes} classes by frequency)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # Load checkpoint
    model, label_map, trained_epoch, trained_acc = load_checkpoint(
        args.checkpoint, device)

    # Build inverse map: int → label string
    id_to_label = {v: k for k, v in label_map.items()}
    class_names  = [id_to_label[i] for i in range(len(id_to_label))]

    # Build dataset
    transform = build_transforms("val")   # no augment at eval time
    if args.use_hf:
        ds = MTSDHuggingFaceDataset(split=args.split, transform=transform,
                                    label_map=label_map)
    else:
        from pathlib import Path
        root    = args.mtsd_root
        ann_dir = str(Path(root) / "mtsd_fully_annotated_annotation" / "mtsd_v2_fully_annotated")
        img_dirs = (
            [str(Path(root) / f"mtsd_fully_annotated_images.train.{i}" / "images") for i in range(3)]
            if args.split == "train" else
            [str(Path(root) / f"mtsd_fully_annotated_images.{args.split}" / "images")]
        )
        ds = MTSDLocalDataset(
            ann_dir=ann_dir, img_dirs=img_dirs, split=args.split,
            transform=transform, label_map=label_map,
        )

    loader = DataLoader(
        ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = config.PIN_MEMORY,
    )

    # Top-k accuracy
    print(f"\nRunning top-k accuracy on {args.split} split …")
    top1, top5 = compute_topk_accuracy(model, loader, device, k=5)
    print(f"  Top-1 accuracy: {top1:.4f} ({top1*100:.2f}%)")
    print(f"  Top-5 accuracy: {top5:.4f} ({top5*100:.2f}%)")

    # Full inference for per-class metrics
    print("\nRunning full inference for per-class metrics …")
    preds, labels = run_inference(model, loader, device)

    # Classification report
    all_labels = list(range(len(class_names)))
    report = classification_report(
        labels, preds,
        labels=all_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    print(f"\n{'─'*60}")
    print("Classification Report (macro averages):")
    macro = report["macro avg"]
    print(f"  Precision : {macro['precision']:.4f}")
    print(f"  Recall    : {macro['recall']:.4f}")
    print(f"  F1        : {macro['f1-score']:.4f}")

    # Save CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import csv
    csv_path = out_dir / f"report_{args.split}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for cls_name in class_names:
            if cls_name in report:
                r = report[cls_name]
                writer.writerow([cls_name, r["precision"], r["recall"],
                                  r["f1-score"], r["support"]])
    print(f"Per-class CSV saved → {csv_path}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_path = out_dir / f"confusion_matrix_{args.split}.png"
    plot_confusion_matrix(cm, class_names, str(cm_path))

    return {"top1": top1, "top5": top5, "macro_f1": macro["f1-score"]}


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate traffic sign classifier")
    parser.add_argument("--checkpoint",  default=config.BEST_MODEL_PATH)
    parser.add_argument("--mtsd-root",   default=config.DATASET_ROOT,
                        help="Directory containing the mtsd_fully_annotated_* folders")
    parser.add_argument("--split",       default="val", choices=["val", "test"])
    parser.add_argument("--use-hf",      action="store_true")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--output-dir",  default="./eval_results")
    args = parser.parse_args()
    evaluate(args)

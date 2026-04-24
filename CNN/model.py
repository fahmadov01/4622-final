"""
model.py — EfficientNet-B3 fine-tuned for traffic sign classification.

Architecture decisions:
  • Backbone : google/efficientnet-b3 pretrained on ImageNet (via HuggingFace transformers)
  • Head     : Dropout → Linear(hidden, num_classes)
  • Strategy : Two-stage fine-tuning
      Stage 1 – freeze backbone, train head only (fast convergence)
      Stage 2 – unfreeze all layers with differential learning rates
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import EfficientNetModel, EfficientNetConfig

import config


class TrafficSignClassifier(nn.Module):
    """
    EfficientNet-B3 backbone with a custom classification head.

    Args:
        num_classes : number of traffic sign classes
        dropout     : dropout probability before the final linear layer
        pretrained  : load ImageNet-pretrained weights from HuggingFace
    """

    FEATURE_DIM = 1536   # EfficientNet-B3 top pooled feature dimension

    def __init__(
        self,
        num_classes: int,
        dropout: float = config.DROPOUT,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────────────
        if pretrained:
            print(f"Loading pretrained backbone: {config.HF_MODEL_NAME}")
            self.backbone = EfficientNetModel.from_pretrained(config.HF_MODEL_NAME)
        else:
            cfg = EfficientNetConfig.from_pretrained(config.HF_MODEL_NAME)
            self.backbone = EfficientNetModel(cfg)

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.FEATURE_DIM, num_classes),
        )

        # Initialise head weights
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) normalised image tensor
        Returns:
            logits: (B, num_classes)
        """
        outputs  = self.backbone(pixel_values=pixel_values)
        pooled   = outputs.pooler_output   # (B, FEATURE_DIM)
        logits   = self.classifier(pooled)
        return logits

    # ── Stage helpers ─────────────────────────────────────────────────────────

    def freeze_backbone(self):
        """Stage 1: freeze backbone, only train the head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("Backbone frozen — training head only.")

    def unfreeze_backbone(self):
        """Stage 2: unfreeze all layers for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen — end-to-end fine-tuning.")

    def get_param_groups(self) -> list:
        """
        Return separate parameter groups so the optimiser can apply
        different learning rates to the backbone vs. the head.
        """
        return [
            {"params": self.backbone.parameters(),   "lr": config.LR_BACKBONE},
            {"params": self.classifier.parameters(), "lr": config.LR_HEAD},
        ]

    # ── Convenience ───────────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def predict_topk(
        self,
        pixel_values: torch.Tensor,
        k: int = config.TOP_K,
        label_names: Optional[list] = None,
    ) -> list:
        """
        Run inference and return top-k predictions with confidence scores.

        Args:
            pixel_values : (1, 3, H, W) or (B, 3, H, W)
            k            : number of top predictions
            label_names  : optional list of class name strings (len == num_classes)
        Returns:
            list of dicts {rank, class_id, class_name, confidence}
        """
        self.eval()
        with torch.no_grad():
            logits = self(pixel_values)
            probs  = torch.softmax(logits, dim=-1)
            top_p, top_idx = torch.topk(probs, k=k, dim=-1)

        results = []
        for rank, (p, idx) in enumerate(zip(top_p[0].tolist(), top_idx[0].tolist())):
            name = label_names[idx] if label_names else str(idx)
            results.append({"rank": rank + 1, "class_id": idx, "class_name": name, "confidence": round(p, 4)})
        return results


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(
    model: TrafficSignClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    label_map: Dict[str, int],
    path: str = config.BEST_MODEL_PATH,
):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "val_acc":   val_acc,
        "label_map": label_map,
        "num_classes": model.num_classes,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    print(f"✓ Checkpoint saved → {path}  (epoch {epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(
    path: str,
    device: torch.device,
) -> Tuple[TrafficSignClassifier, Dict, int, float]:
    """Load a checkpoint and return (model, label_map, epoch, val_acc)."""
    ckpt       = torch.load(path, map_location=device)
    label_map  = ckpt["label_map"]
    num_classes = ckpt["num_classes"]

    model = TrafficSignClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    print(f"✓ Loaded checkpoint from {path}  "
          f"(epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    return model, label_map, ckpt["epoch"], ckpt["val_acc"]

"""
infer.py — Run inference on a single image or a folder of images.

Usage:
  python infer.py --checkpoint checkpoints/best_model.pt --image path/to/sign.jpg
  python infer.py --checkpoint checkpoints/best_model.pt --folder path/to/images/
  python infer.py --checkpoint checkpoints/best_model.pt --image sign.jpg --top-k 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms

import config
from model import load_checkpoint, TrafficSignClassifier


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def build_inference_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


def infer_single(
    model: TrafficSignClassifier,
    image_path: str,
    transform,
    label_names: List[str],
    device: torch.device,
    top_k: int = config.TOP_K,
) -> list:
    img    = load_image(image_path)
    tensor = transform(img).unsqueeze(0).to(device)
    return model.predict_topk(tensor, k=top_k, label_names=label_names)


def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    model, label_map, _, _ = load_checkpoint(args.checkpoint, device)
    id_to_label = {v: k for k, v in label_map.items()}
    label_names = [id_to_label[i] for i in range(len(id_to_label))]
    transform   = build_inference_transform()

    paths = []
    if args.image:
        paths = [args.image]
    elif args.folder:
        paths = [
            str(p) for p in Path(args.folder).iterdir()
            if p.suffix.lower() in VALID_EXTS
        ]
    else:
        print("Provide --image or --folder.")
        return

    for path in paths:
        preds = infer_single(model, path, transform, label_names, device, args.top_k)
        print(f"\n📷 {path}")
        for p in preds:
            bar = "█" * int(p["confidence"] * 30)
            print(f"  #{p['rank']:2d} [{bar:<30}] {p['confidence']*100:5.1f}%  {p['class_name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer traffic signs")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    parser.add_argument("--image",   default=None, help="Path to a single image")
    parser.add_argument("--folder",  default=None, help="Path to a folder of images")
    parser.add_argument("--top-k",   type=int, default=config.TOP_K)
    args = parser.parse_args()
    run(args)

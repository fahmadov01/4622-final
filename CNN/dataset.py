"""
dataset.py — Dataset loading for the Mapillary Traffic Sign Dataset (MTSD)

Supports two modes:
  1. Local MTSD download  — reads the official per-image JSON annotations
  2. HuggingFace mirror   — `sparshgarg57/mapillary_traffic_signs`

MTSD local layout (as extracted from the official download packages):

  mtsd_fully_annotated_annotation/
    mtsd_v2_fully_annotated/
      splits/
        train.txt   val.txt   test.txt   ← one image key per line
      annotations/
        <key>.json                        ← per-image annotation file
  mtsd_fully_annotated_images.train.0/
    images/  <key>.jpg
  mtsd_fully_annotated_images.train.1/
    images/  <key>.jpg
  mtsd_fully_annotated_images.train.2/
    images/  <key>.jpg
  mtsd_fully_annotated_images.val/
    images/  <key>.jpg
  mtsd_fully_annotated_images.test/
    images/  <key>.jpg

Each per-image annotation JSON looks like:
  {
    "width": 4160, "height": 3120, "ispano": false,
    "objects": [
      {
        "key": "...",
        "label": "regulatory--keep-right--g1",
        "bbox":  {"xmin": 100.0, "ymin": 200.0, "xmax": 150.0, "ymax": 260.0},
        "properties": {"occluded": false, "ambiguous": false, "out-of-frame": false, ...}
      }, ...
    ]
  }

A first-run annotation index is cached to config.CACHE_DIR so that
subsequent runs skip the slow per-image JSON scan.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


# ─── Transforms ───────────────────────────────────────────────────────────────

def build_transforms(split: str) -> transforms.Compose:
    """Return augmentation pipeline for a given split."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats — EfficientNet was pretrained on it
        std =[0.229, 0.224, 0.225],
    )

    if split == "train" and config.AUGMENT_TRAIN:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=15),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandAugment(num_ops=config.RAND_AUG_N, magnitude=config.RAND_AUG_M),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])


# ─── Local MTSD Dataset ───────────────────────────────────────────────────────

class MTSDLocalDataset(Dataset):
    """
    Loads sign patches from a locally downloaded MTSD directory.

    Each sample is a cropped bounding-box region around a single traffic sign.
    On first use, the annotation index is built by reading every per-image JSON
    and cached to disk; subsequent runs load the cache instantly.

    Args:
        ann_dir        : path to the mtsd_v2_fully_annotated/ directory
        img_dirs       : list of directories to search for <key>.jpg image files
        split          : "train", "val", or "test"
        transform      : torchvision transform pipeline (defaults to build_transforms)
        label_map      : {label_str: int} — pass the train dataset's map to val/test
        min_bbox_area  : skip signs whose cropped bbox area (px²) is below this
        skip_occluded  : skip signs with properties.occluded == True
        skip_ambiguous : skip signs with properties.ambiguous == True
        skip_out_of_frame : skip signs marked out-of-frame
        skip_other_sign   : skip the "other-sign" catch-all class
        cache_dir      : directory for the pickled sample index cache
    """

    def __init__(
        self,
        ann_dir: str,
        img_dirs: List[str],
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        label_map: Optional[Dict[str, int]] = None,
        min_bbox_area: int = 400,
        skip_occluded: bool = config.SKIP_OCCLUDED,
        skip_ambiguous: bool = config.SKIP_AMBIGUOUS,
        skip_out_of_frame: bool = config.SKIP_OUT_OF_FRAME,
        skip_other_sign: bool = config.SKIP_OTHER_SIGN,
        cache_dir: str = config.CACHE_DIR,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.ann_dir   = Path(ann_dir)
        self.img_dirs  = [Path(d) for d in img_dirs]
        self.split     = split
        self.transform = transform or build_transforms(split)

        # ── Validate paths ────────────────────────────────────────────────────
        splits_file = self.ann_dir / "splits" / f"{split}.txt"
        if not splits_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {splits_file}\n"
                "Expected the mtsd_v2_fully_annotated/ directory from the "
                "official MTSD download.  Set USE_HF_DATASET=True in config.py "
                "to use the HuggingFace mirror instead."
            )

        # ── Build or load sample index ────────────────────────────────────────
        # Cache key encodes the split + all filter settings so changing a flag
        # forces a rebuild automatically.
        cache_tag = (
            f"{split}_occ{int(skip_occluded)}_amb{int(skip_ambiguous)}"
            f"_oof{int(skip_out_of_frame)}_oth{int(skip_other_sign)}"
            f"_area{min_bbox_area}"
        )
        cache_path = Path(cache_dir) / f"index_{cache_tag}.pkl"

        if cache_path.exists():
            print(f"Loading cached annotation index: {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.samples  = cached["samples"]
            raw_label_map = cached["label_map"]
        else:
            print(f"Building annotation index for '{split}' split …")
            self.samples, raw_label_map = self._build_index(
                splits_file, min_bbox_area,
                skip_occluded, skip_ambiguous, skip_out_of_frame, skip_other_sign,
            )
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"samples": self.samples, "label_map": raw_label_map}, f)
            print(f"Index cached → {cache_path}")

        # ── Apply / build label map ───────────────────────────────────────────
        if label_map is not None:
            # Filter out any samples whose label is absent from the provided map
            # (can happen when val/test has labels not seen in train).
            self.label_map = label_map
            self.samples = [s for s in self.samples if s["label"] in self.label_map]
        else:
            self.label_map = raw_label_map

        self.num_classes = len(self.label_map)
        # Patch label_id into each sample using the final label_map
        for s in self.samples:
            s["label_id"] = self.label_map[s["label"]]

        print(
            f"[{split}] {len(self.samples):,} sign patches | "
            f"{self.num_classes} classes"
        )

    # ── Index builder ─────────────────────────────────────────────────────────

    def _build_index(
        self,
        splits_file: Path,
        min_bbox_area: int,
        skip_occluded: bool,
        skip_ambiguous: bool,
        skip_out_of_frame: bool,
        skip_other_sign: bool,
    ) -> Tuple[List[dict], Dict[str, int]]:
        """
        Read the split text file, locate each image, parse its annotation JSON,
        and collect one entry per valid bounding box.
        """
        # Build key → image path lookup by scanning all image directories once
        img_lookup: Dict[str, str] = {}
        for img_dir in self.img_dirs:
            if not img_dir.exists():
                continue
            for p in img_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    img_lookup[p.stem] = str(p)

        ann_dir = self.ann_dir / "annotations"

        image_keys = splits_file.read_text().splitlines()
        image_keys = [k.strip() for k in image_keys if k.strip()]

        samples: List[dict] = []
        labels_seen: set = set()
        skipped_no_image = skipped_filter = 0

        for key in image_keys:
            img_path = img_lookup.get(key)
            if img_path is None:
                skipped_no_image += 1
                continue

            ann_path = ann_dir / f"{key}.json"
            if not ann_path.exists():
                skipped_no_image += 1
                continue

            with open(ann_path) as f:
                ann = json.load(f)

            for obj in ann.get("objects", []):
                label = obj.get("label", "")

                if skip_other_sign and label == "other-sign":
                    skipped_filter += 1
                    continue

                props = obj.get("properties", {})
                if skip_occluded    and props.get("occluded", False):
                    skipped_filter += 1
                    continue
                if skip_ambiguous   and props.get("ambiguous", False):
                    skipped_filter += 1
                    continue
                if skip_out_of_frame and props.get("out-of-frame", False):
                    skipped_filter += 1
                    continue

                bbox = obj.get("bbox", {})

                # Panorama cross-boundary signs require stitching two crops —
                # skip them for simplicity (they are a small minority).
                if "cross_boundary" in bbox:
                    skipped_filter += 1
                    continue

                xmin = bbox.get("xmin", 0)
                ymin = bbox.get("ymin", 0)
                xmax = bbox.get("xmax", 0)
                ymax = bbox.get("ymax", 0)
                w = xmax - xmin
                h = ymax - ymin
                if w * h < min_bbox_area:
                    skipped_filter += 1
                    continue

                samples.append({
                    "img_path": img_path,
                    "bbox":     (xmin, ymin, xmax, ymax),   # stored as (x1,y1,x2,y2)
                    "label":    label,
                    "label_id": 0,   # filled in after label_map is finalised
                })
                labels_seen.add(label)

        label_map = {lbl: i for i, lbl in enumerate(sorted(labels_seen))}
        print(
            f"  Scanned {len(image_keys):,} images | "
            f"kept {len(samples):,} signs | "
            f"skipped {skipped_no_image:,} (no image/ann) + "
            f"{skipped_filter:,} (filtered)"
        )
        return samples, label_map

    # ── Crop helper ───────────────────────────────────────────────────────────

    @staticmethod
    def _crop_with_padding(
        img: Image.Image,
        bbox: Tuple,
        padding: float,
    ) -> Image.Image:
        """Expand bbox by `padding` fraction on each side, then crop."""
        W, H  = img.size
        x1, y1, x2, y2 = bbox
        pw = (x2 - x1) * padding
        ph = (y2 - y1) * padding
        cx1 = max(0, int(x1 - pw))
        cy1 = max(0, int(y1 - ph))
        cx2 = min(W, int(x2 + pw))
        cy2 = min(H, int(y2 + ph))
        return img.crop((cx1, cy1, cx2, cy2))

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s   = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        img = self._crop_with_padding(img, s["bbox"], config.CROP_PADDING)
        if self.transform:
            img = self.transform(img)
        return img, s["label_id"]


# ─── HuggingFace Mirror Dataset ───────────────────────────────────────────────

class MTSDHuggingFaceDataset(Dataset):
    """
    Wraps the `sparshgarg57/mapillary_traffic_signs` HuggingFace dataset.
    Images are already cropped sign patches in this mirror.
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        label_map: Optional[Dict[str, int]] = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets  # required for HF mirror")

        hf_split = {"train": "train", "val": "validation", "test": "test"}.get(split, split)
        print(f"Loading HuggingFace dataset {config.HF_DATASET} / {hf_split} …")
        raw = load_dataset(config.HF_DATASET, split=hf_split, trust_remote_code=True)

        self.transform = transform or build_transforms(split)

        all_labels = sorted(set(raw["label"]))
        if label_map is None:
            self.label_map: Dict[str, int] = {lbl: i for i, lbl in enumerate(all_labels)}
        else:
            self.label_map = label_map

        self.num_classes = len(self.label_map)
        self.data = [row for row in raw if row["label"] in self.label_map]
        print(f"[{split}] HF dataset: {len(self.data):,} samples, {self.num_classes} classes.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row   = self.data[idx]
        img   = row["image"].convert("RGB")
        label = self.label_map[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_datasets(
    root: str = config.DATASET_ROOT,
    use_hf: bool = config.USE_HF_DATASET,
):
    """
    Returns (train_dataset, val_dataset, num_classes, label_map).

    Picks between local MTSD and HuggingFace mirror based on `use_hf`.
    Automatically falls back to HF if local data is absent.

    Args:
        root   : directory containing the mtsd_fully_annotated_* folders
        use_hf : force use of the HuggingFace mirror
    """
    if not use_hf:
        ann_dir = Path(root) / "mtsd_fully_annotated_annotation" / "mtsd_v2_fully_annotated"
        if not (ann_dir / "splits" / "train.txt").exists():
            print(f"Local MTSD not found at {root}. Falling back to HuggingFace mirror.")
            use_hf = True

    if use_hf:
        train_ds = MTSDHuggingFaceDataset(split="train")
        val_ds   = MTSDHuggingFaceDataset(split="val", label_map=train_ds.label_map)
    else:
        ann_dir  = str(Path(root) / "mtsd_fully_annotated_annotation" / "mtsd_v2_fully_annotated")
        img_dirs = {
            split: [
                str(Path(root) / f"mtsd_fully_annotated_images.train.{i}" / "images")
                for i in range(3)
            ] if split == "train" else [
                str(Path(root) / f"mtsd_fully_annotated_images.{split}" / "images")
            ]
            for split in ("train", "val", "test")
        }
        train_ds = MTSDLocalDataset(
            ann_dir=ann_dir, img_dirs=img_dirs["train"], split="train")
        val_ds = MTSDLocalDataset(
            ann_dir=ann_dir, img_dirs=img_dirs["val"], split="val",
            label_map=train_ds.label_map)

    return train_ds, val_ds, train_ds.num_classes, train_ds.label_map


# ─── Mixup Collation ──────────────────────────────────────────────────────────

class MixupCollate:
    """
    Picklable collate_fn that applies Mixup augmentation.
    Must be a class (not a closure) so Windows multiprocessing can pickle it.
    If alpha=0, behaves like the default collate_fn.
    """

    def __init__(self, alpha: float = config.MIXUP_ALPHA):
        self.alpha = alpha

    def __call__(self, batch):
        import torch
        from torch.utils.data.dataloader import default_collate

        images, labels = default_collate(batch)
        if self.alpha <= 0 or not torch.is_grad_enabled():
            return images, labels

        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(images.size(0))
        mixed = lam * images + (1 - lam) * images[idx]
        return mixed, (labels, labels[idx], lam)


def mixup_collate(alpha: float = config.MIXUP_ALPHA) -> MixupCollate:
    return MixupCollate(alpha=alpha)

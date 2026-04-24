# Traffic Sign Classifier
**EfficientNet-B3 fine-tuned on the Mapillary Traffic Sign Dataset (MTSD)**

---

## What this does

Takes `google/efficientnet-b3` (pretrained on ImageNet via HuggingFace) and
fine-tunes it for traffic sign classification across ~300+ classes using the
MTSD v2 dataset — 36K annotated street-level images from around the world.

Each training sample is a **sign patch** cropped from a full street image using
the ground-truth bounding box (with configurable padding).

### Architecture

```
EfficientNetModel (HuggingFace)        ← pretrained backbone
  └─ pooler_output (1536-dim)
       └─ Dropout(0.3)
            └─ Linear(1536 → num_classes)   ← newly trained head
```

### Two-stage fine-tuning strategy

| Stage | Epochs | Backbone | Head LR | Backbone LR |
|-------|--------|----------|---------|-------------|
| 1     | 1–5    | Frozen   | 1e-3    | —           |
| 2     | 6–30   | Unfrozen | 1e-3    | 1e-4        |

Stage 1 trains only the classification head — fast and avoids catastrophic
forgetting of pretrained features. Stage 2 does end-to-end fine-tuning with
differential learning rates so the backbone adapts gently.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### Install PyTorch with CUDA (GPU — recommended)

First check which CUDA version your driver supports:
```bash
nvidia-smi      # look for "CUDA Version: XX.X" in the top-right corner
```

Then install the matching PyTorch build:
```bash
# CUDA 12.1 — works with driver >= 525 (most modern cards)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 — for older drivers
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify the GPU is detected before running training:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True  /  NVIDIA GeForce RTX XXXX (or similar)
```

### Install remaining dependencies
```bash
pip install -r requirements.txt
```

> **CPU-only?** Skip the CUDA install above and just run `pip install -r requirements.txt`.
> Training will work but expect 20–50× slower throughput.

---

## Transferring to a new machine

Copy the entire project folder to the GPU machine. The `cache/` directory
contains the pre-built annotation index — keep it to avoid re-scanning the
36K annotation JSONs on first run. The `checkpoints/` directory contains any
saved model weights — keep it if you want to resume training.

Files you **must** transfer (large):
- `mtsd_fully_annotated_images.train.0/`
- `mtsd_fully_annotated_images.train.1/`
- `mtsd_fully_annotated_images.train.2/`
- `mtsd_fully_annotated_images.val/`
- `mtsd_fully_annotated_images.test/`
- `mtsd_fully_annotated_annotation/`

Files you **should** transfer (small, saves time):
- `cache/` — annotation index (~10 MB, skips the 1-2 min rebuild)
- `checkpoints/` — saved weights (if any)

After copying, install dependencies on the new machine (see Setup above),
then run `python train.py` from the project directory.

---

## Dataset layout

Extract the official MTSD download packages into the **project root** (the
same directory as `train.py`).  The expected layout after extraction:

```
mlproject/                                  ← project root (DATASET_ROOT)
  mtsd_fully_annotated_annotation/
    mtsd_v2_fully_annotated/
      splits/
        train.txt
        val.txt
        test.txt
      annotations/
        <image_key>.json                    ← one JSON per image
  mtsd_fully_annotated_images.train.0/
    images/  <image_key>.jpg
  mtsd_fully_annotated_images.train.1/
    images/  <image_key>.jpg
  mtsd_fully_annotated_images.train.2/
    images/  <image_key>.jpg
  mtsd_fully_annotated_images.val/
    images/  <image_key>.jpg
  mtsd_fully_annotated_images.test/
    images/  <image_key>.jpg
```

If the packages are somewhere else, either set `MTSD_ROOT` in your environment
or pass `--mtsd-root /path/to/that/directory` to the training scripts.

### HuggingFace mirror (no registration)

If you do not have the official download, the `sparshgarg57/mapillary_traffic_signs`
HuggingFace mirror provides pre-cropped sign patches:

```bash
python train.py --use-hf
```

---

## Training

```bash
# Local MTSD (packages extracted in project root)
python train.py

# Custom root
python train.py --mtsd-root /path/to/mtsd

# HuggingFace mirror
python train.py --use-hf

# Custom epochs / batch size
python train.py --epochs 40 --batch-size 64

# Monitor with TensorBoard
tensorboard --logdir runs/
```

### First-run note

On the first run the dataset loader scans all annotation JSONs and builds a
sample index, which it caches to `./cache/`.  Subsequent runs load the cache
instantly (~1 s).

### What gets saved
- `checkpoints/best_model.pt` — checkpoint whenever val accuracy improves
- `runs/`                      — TensorBoard event files
- `cache/`                     — annotation index cache

---

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
python evaluate.py --checkpoint checkpoints/best_model.pt --split test

# Outputs:
#   eval_results/report_val.csv          ← per-class precision/recall/F1
#   eval_results/confusion_matrix_val.png
```

---

## Inference

```bash
# Single image
python infer.py --image path/to/sign.jpg

# Folder of images
python infer.py --folder path/to/signs/

# More top-k predictions
python infer.py --image sign.jpg --top-k 10
```

Example output:
```
📷 sign.jpg
   #1 [████████████████████          ]  69.3%  regulatory--speed-limit-50--g1
   #2 [████████                      ]  21.4%  regulatory--speed-limit-60--g1
   #3 [██                            ]   6.1%  regulatory--speed-limit-40--g1
   #4 [                              ]   1.8%  warning--pedestrians-crossing--g1
   #5 [                              ]   1.4%  complementary--maximum-speed-limit-50--g1
```

---

## Configuration

All key hyperparameters are in `config.py`:

| Parameter          | Default | Description                                   |
|--------------------|---------|-----------------------------------------------|
| `DATASET_ROOT`     | `"."`   | Project root containing the mtsd_* folders    |
| `IMG_SIZE`         | 224     | Input resolution                              |
| `CROP_PADDING`     | 0.15    | Extra padding around bbox crops              |
| `EPOCHS`           | 30      | Total training epochs                         |
| `BATCH_SIZE`       | 32      | Training batch size                           |
| `LR_HEAD`          | 1e-3    | Classification head learning rate             |
| `LR_BACKBONE`      | 1e-4    | Backbone fine-tuning learning rate            |
| `LABEL_SMOOTHING`  | 0.1     | Reduces overconfidence                        |
| `MIXUP_ALPHA`      | 0.2     | Mixup regularisation (0 to disable)          |
| `PATIENCE`         | 7       | Early stopping patience (epochs)              |
| `DROPOUT`          | 0.3     | Dropout before classifier head               |
| `SKIP_OCCLUDED`    | True    | Skip occluded signs during training           |
| `SKIP_AMBIGUOUS`   | True    | Skip ambiguous signs during training          |
| `SKIP_OTHER_SIGN`  | True    | Skip the "other-sign" catch-all class         |

---

## Expected performance

Based on published baselines using EfficientNet on MTSD-like data:

| Metric              | Expected range  |
|---------------------|-----------------|
| Top-1 val accuracy  | 72–82%          |
| Top-5 val accuracy  | 90–95%          |
| Macro F1            | 65–78%          |

> MTSD is long-tailed — many rare sign classes have few examples.
> Per-class performance varies significantly.

---

## Hardware recommendations

| Hardware     | Batch size | Time/epoch (est.) |
|--------------|-----------|-------------------|
| RTX 3090     | 64        | ~8 min            |
| RTX 4090     | 128       | ~4 min            |
| M1/M2 Mac    | 32        | ~25 min (MPS)     |
| CPU only     | 16        | Very slow         |

Mixed-precision (AMP) is enabled automatically on CUDA.

---

## File structure

```
mlproject/
  config.py        ← all hyperparameters in one place
  dataset.py       ← MTSD dataset classes (local + HF mirror)
  model.py         ← EfficientNet-B3 classifier + checkpoint utils
  train.py         ← two-stage training loop
  evaluate.py      ← evaluation + per-class metrics
  infer.py         ← inference on single images or folders
  requirements.txt
  README.md
  cache/           ← annotation index cache (auto-created)
  checkpoints/     ← saved model checkpoints (auto-created)
  runs/            ← TensorBoard logs (auto-created)
```

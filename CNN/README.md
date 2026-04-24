# Traffic Sign Classifier (CNN)

**EfficientNet-B3 fine-tuned on the Mapillary Traffic Sign Dataset (MTSD).**
Classifies cropped traffic-sign images into 400 classes.

## Results

Trained for 26 epochs; best checkpoint at epoch 19.

| Metric | Value |
|---|---|
| Top-1 accuracy (val) | **92.92%** |
| Top-5 accuracy (val) | 99.41% |
| Weighted F1 | 92.72% |
| Macro F1 (386 non-empty classes) | 89.02% |

Held-out evaluation was done on the MTSD val split. The MTSD test split ships without public annotations (Mapillary holds them for their benchmark server), so val is used as the held-out set here.

Per-class numbers: `eval_results/report_val.csv`. Confusion matrix: `eval_results/confusion_matrix_val.png`. Training curves: `runs/` (view with `tensorboard --logdir runs/`).

---

## Quick start — running inference

```bash
cd CNN
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

Install PyTorch matched to your hardware (optional GPU build):
```bash
# CUDA 12.1 (recommended on a GPU machine)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU-only (works everywhere, slower)
pip install torch torchvision
```

Download the trained checkpoint and place it at `CNN/checkpoints/best_model.pt`
(the 136 MB file is not stored in the repo — see **Getting the checkpoint** below).

Run predictions:
```bash
# single image
python infer.py --image path/to/sign.jpg

# folder of images
python infer.py --folder path/to/signs/

# change how many classes to show
python infer.py --image sign.jpg --top-k 10
```

Example output:
```
📷 sign.jpg
   #1 [████████████████████          ]  69.3%  regulatory--maximum-speed-limit-50--g1
   #2 [████████                      ]  21.4%  regulatory--maximum-speed-limit-60--g1
   #3 [██                            ]   6.1%  regulatory--maximum-speed-limit-40--g1
```

**Input expectations.** The model was trained on tight crops around a single sign (with 15% bbox padding). For best results, pass an image that is already cropped to one sign. Full street scenes will still produce a prediction, but accuracy falls off the further you are from a tight crop.

---

## Getting the checkpoint

The trained weights (`best_model.pt`, 136 MB) exceed GitHub's 100 MB file limit and are not in this repo.

To obtain it:
- Ask the repo owner for the file and drop it into `CNN/checkpoints/best_model.pt`, **or**
- Re-run training yourself (see **Retraining** below; takes several hours on a modern GPU).

---

## Re-evaluating

If you have the checkpoint:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --split val
```

Outputs `eval_results/report_val.csv` and `eval_results/confusion_matrix_val.png`. (These are already checked into the repo from the original run.)

Evaluation on `--split test` will not work — the MTSD test split has no public annotations.

---

## Retraining

You need the full MTSD v2 fully-annotated download (images + annotations, ~100 GB unpacked). Extract the archives into the `CNN/` directory with this layout:

```
CNN/
  mtsd_fully_annotated_annotation/mtsd_v2_fully_annotated/
    splits/{train,val,test}.txt
    annotations/<key>.json
  mtsd_fully_annotated_images.train.0/images/<key>.jpg
  mtsd_fully_annotated_images.train.1/images/<key>.jpg
  mtsd_fully_annotated_images.train.2/images/<key>.jpg
  mtsd_fully_annotated_images.val/images/<key>.jpg
  mtsd_fully_annotated_images.test/images/<key>.jpg
```

Or point at a custom location via `MTSD_ROOT=/path/to/mtsd` or `--mtsd-root`.

Then:
```bash
python train.py                           # local MTSD
python train.py --use-hf                  # HuggingFace mirror (pre-cropped patches)
python train.py --epochs 40 --batch-size 64
tensorboard --logdir runs/                # monitor
```

First run scans the annotation JSONs and caches an index to `cache/` (~1-2 min). Subsequent runs load the cache instantly.

### Training strategy

| Stage | Epochs | Backbone | Head LR | Backbone LR |
|-------|--------|----------|---------|-------------|
| 1     | 1–5    | Frozen   | 1e-3    | —           |
| 2     | 6–30   | Unfrozen | 1e-3    | 1e-4        |

Stage 1 trains only the classification head. Stage 2 does end-to-end fine-tuning with a lower backbone LR so pretrained features adapt gently. Regularization: label smoothing (0.1), Mixup (α=0.2), RandAugment, dropout 0.3, weight decay 1e-4. Cosine LR schedule with 2 warmup epochs. Early stopping patience 7.

Hyperparameters live in `config.py`.

---

## Files

```
CNN/
  config.py              all hyperparameters
  dataset.py             MTSD dataset classes (local + HF mirror)
  model.py               EfficientNet-B3 classifier + checkpoint utils
  train.py               two-stage training loop
  evaluate.py            evaluation + per-class metrics
  infer.py               inference on single images or folders
  requirements.txt
  eval_results/          committed val metrics from the trained run
    report_val.csv
    confusion_matrix_val.png
  runs/                  committed TensorBoard logs from the trained run
  checkpoints/           (not in repo — place best_model.pt here)
  cache/                 (auto-created on first train run)
```

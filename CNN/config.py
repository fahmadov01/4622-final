"""
Configuration for Traffic Sign Classifier
Fine-tuning EfficientNet-B3 on the Mapillary Traffic Sign Dataset (MTSD)
"""

import os

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Root = the directory that contains the mtsd_fully_annotated_* folders
# (i.e. the project root when the dataset archives were extracted here).
DATASET_ROOT   = os.environ.get("MTSD_ROOT", ".")

# Sub-paths derived from DATASET_ROOT — override if your layout differs
MTSD_ANN_DIR   = os.path.join(DATASET_ROOT,
                               "mtsd_fully_annotated_annotation",
                               "mtsd_v2_fully_annotated")

# Image dirs for each split (train images come in 3 numbered packages)
MTSD_IMG_DIRS  = {
    "train": [
        os.path.join(DATASET_ROOT, f"mtsd_fully_annotated_images.train.{i}", "images")
        for i in range(3)
    ],
    "val":  [os.path.join(DATASET_ROOT, "mtsd_fully_annotated_images.val",  "images")],
    "test": [os.path.join(DATASET_ROOT, "mtsd_fully_annotated_images.test", "images")],
}

HF_DATASET     = "sparshgarg57/mapillary_traffic_signs"   # HuggingFace mirror
USE_HF_DATASET = False        # Set True to use HF mirror instead of local files

# Sign filtering (local dataset only)
SKIP_OCCLUDED  = True         # skip signs marked occluded
SKIP_AMBIGUOUS = True         # skip signs marked ambiguous
SKIP_OUT_OF_FRAME = True      # skip signs marked out-of-frame
SKIP_OTHER_SIGN   = True      # skip the catch-all "other-sign" class

# ─── Model ────────────────────────────────────────────────────────────────────
HF_MODEL_NAME  = "google/efficientnet-b3"
NUM_CLASSES    = 400          # set dynamically at runtime from the data
IMG_SIZE       = 224          # EfficientNet-B3 default input size
CROP_PADDING   = 0.15         # extra padding around bounding boxes when cropping

# ─── Training ─────────────────────────────────────────────────────────────────
SEED           = 42
EPOCHS         = 30
BATCH_SIZE     = 64    # 64 fits comfortably on an 8 GB GPU; use 32 if you get OOM
NUM_WORKERS    = 8     # rule of thumb: number of CPU cores / 2; set 0 to disable multiprocessing
PIN_MEMORY     = True  # speeds up CPU→GPU transfers; automatically disabled when no GPU present

# Learning rate schedule
LR_HEAD        = 1e-3         # lr for newly added classification head
LR_BACKBONE    = 1e-4         # lr for pretrained backbone (lower = gentle fine-tuning)
WEIGHT_DECAY   = 1e-4
LR_SCHEDULER   = "cosine"     # "cosine" | "step"
WARMUP_EPOCHS  = 2
MIN_LR         = 1e-6

# Regularisation
DROPOUT        = 0.3
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA    = 0.2          # 0 to disable mixup

# Early stopping
PATIENCE       = 7            # epochs with no val improvement before stopping

# ─── Checkpointing ────────────────────────────────────────────────────────────
CHECKPOINT_DIR  = "./checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
LOG_DIR         = "./runs"
CACHE_DIR       = "./cache"   # annotation index cache (speeds up repeated runs)

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUGMENT_TRAIN = True
RAND_AUG_N    = 2             # RandAugment N ops
RAND_AUG_M    = 9             # RandAugment magnitude

# ─── Inference ────────────────────────────────────────────────────────────────
TOP_K         = 5             # top-K predictions to display

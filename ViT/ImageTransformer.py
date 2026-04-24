import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

# Annotation files
BASE = os.path.dirname(os.path.abspath(__file__))
ANNOTATION_DIR = os.path.join(
    BASE, "mtsd_fully_annotated_annotation", "mtsd_v2_fully_annotated", "annotations"
)

# Train/val/test split lists
SPLITS_DIR = os.path.join(
    BASE, "mtsd_fully_annotated_annotation", "mtsd_v2_fully_annotated", "splits"
)

# Image directories
IMAGE_DIRS = [
    os.path.join(BASE, "mtsd_fully_annotated_images.train.0", "images"),
    os.path.join(BASE, "mtsd_fully_annotated_images.train.1", "images"),
    os.path.join(BASE, "mtsd_fully_annotated_images.train.2", "images"),
    os.path.join(BASE, "mtsd_fully_annotated_images.val", "images"),
    os.path.join(BASE, "mtsd_fully_annotated_images.test", "images"),
]

# Padding applied to bounding boxes when cropping objects
BBOX_PADDING = 0.15

#Custom dataset for processing the images with annotations
class TrafficSignDataset(Dataset):
    def __init__(self, image_list, annotations, image_paths, processor):
        self.samples = image_list
        self.image_paths = image_paths
        self.processor = processor
        self.annotations = annotations

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id = self.samples[idx]
        img_path = self.image_paths[img_id]
        ann = self.annotations[img_id]

        with Image.open(img_path) as image:
            image = image.convert("RGB")
            x1, y1, x2, y2 = ann["bbox"]
            w, h = x2 - x1, y2 - y1
            x1 = max(0, x1 - BBOX_PADDING * w)
            y1 = max(0, y1 - BBOX_PADDING * h)
            x2 = min(image.width, x2 + BBOX_PADDING * w)
            y2 = min(image.height, y2 + BBOX_PADDING * h)
            crop = image.crop((x1, y1, x2, y2))

        inputs = self.processor(crop, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(ann["label"], dtype=torch.long)

        return inputs


#Helper function to read files
def read(path):
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]

#Actually load train/test/validate datasets 
def get_annotation_splits():
    test = read(os.path.join(SPLITS_DIR, "test.txt"))
    train = read(os.path.join(SPLITS_DIR, "train.txt"))
    val = read(os.path.join(SPLITS_DIR, "val.txt"))
    return test, train, val

"""
    This is to parse annotation JSON files and extract:
    - first valid traffic sign object per image
    - bounding box + label
"""
def build_annotations(annotation_dir):
    annotations = {}

    for file in os.listdir(annotation_dir):
        if not file.endswith(".json"):
            continue

        path = os.path.join(annotation_dir, file)

        with open(path, "r") as f:
            data = json.load(f)

        chosen = None
        for obj in data.get("objects", []):
            if obj["label"] == "other-sign":
                continue
            chosen = obj
            break

        if chosen is None:
            continue

        bb = chosen["bbox"]
        image_id = file.replace(".json", "")
        annotations[image_id] = {
            "label": chosen["label"],
            "bbox": (bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]),
        }

    return annotations

#This is to build a mapping from image_id to the full file path
def build_image_index():
    index = {}
    for d in IMAGE_DIRS:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith(".jpg"):
                index[fname[:-4]] = os.path.join(d, fname)
    return index

#This defines the evaluation metrics 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    top5 = np.argsort(logits, axis=-1)[:, -5:]
    return {
        "accuracy": float((preds == labels).mean()),
        "top5_accuracy": float(np.mean([l in t for l, t in zip(labels, top5)])),
    }


def main():
    #This loads the pretrained ViT model
    checkpoint = "google/vit-base-patch16-224-in21k"
    processor = AutoImageProcessor.from_pretrained(checkpoint)

    #This loads the dataset splits. Due to issues with the 
    #test dataset, we don't use it
    test_list, train_list, val_list = get_annotation_splits()

    #This loads the annotations 
    raw_annotations = build_annotations(ANNOTATION_DIR)

    #This builds the actual vocabulary for the labels 
    classes = sorted(set(a["label"] for a in raw_annotations.values()))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    #This converts annotations to numeric labels
    annotations = {
        k: {"label": label2id[v["label"]], "bbox": v["bbox"]}
        for k, v in raw_annotations.items()
    }
    image_paths = build_image_index() #This loads the image paths

    # This is to filter ids so that they are both in annotations and images
    def usable(ids):
        return [i for i in ids if i in annotations and i in image_paths]

    train_list = usable(train_list)
    val_list = usable(val_list)

    print(f"Classes: {len(label2id)} | train: {len(train_list)} | val: {len(val_list)}")

    # This creates the PyTorch datasets
    train_dataset = TrafficSignDataset(train_list, annotations, image_paths, processor)
    val_dataset = TrafficSignDataset(val_list, annotations, image_paths, processor)

    # This laods the pretrained ViT model
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # This is the training configuration
    args = TrainingArguments(
        output_dir="./vit-traffic-sign",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=50,
        remove_unused_columns=False,
        report_to="tensorboard",
        logging_dir="./vit-traffic-sign/logs",
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    #Set up for the HuggingFace Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    #Train the model
    trainer.train()

    #Save the final model
    trainer.save_model("./vit-traffic-sign/final")
    
    #Evaluate the model and print it out
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()

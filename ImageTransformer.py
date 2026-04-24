import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)

ANNOTATIONS_PATH = "./annotations"
IMAGE_DIR = "./images"
MODEL_NAME = "google/vit-base-patch16-224"
OUTPUT_DIR = "./output"
BATCH_SIZE = 8
EPOCHS = 5

"""
    This function reads the annotation JSON files and builds a list of samples.
    Each sample contains:
      - image filename
      - list of labels (multi-label classification)
"""
def build_samples(path):
    samples = []
    for f in os.listdir(path):
        if not f.endswith(".json"):
            continue

        with open(os.path.join(path, f)) as file:
            data = json.load(file)

        objects = data.get("objects", [])
        objects = [o for o in objects if not o["properties"]["ambiguous"]]

        if len(objects) == 0:
            continue

        labels = list(set(o["label"] for o in objects))
        image_file = f.replace(".json", ".jpg")

        samples.append({"image": image_file, "labels": labels})

    return samples


"""
    This function builds a label-to-index mapping for multi-label classification.
"""
def build_label_map(samples):
    all_labels = set()
    for s in samples:
        all_labels.update(s["labels"])
    labels = sorted(all_labels)
    return {l: i for i, l in enumerate(labels)}

"""
    This is a custom dataset class to load images and multi-hot encoded labels.
"""
class TrafficSignDataset(Dataset):
    def __init__(self, samples, label_map, image_dir, processor):
        self.samples = samples
        self.label_map = label_map
        self.image_dir = image_dir
        self.processor = processor
        self.num_labels = len(label_map)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img_path = os.path.join(self.image_dir, s["image"])
        image = Image.open(img_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        label_vec = torch.zeros(self.num_labels, dtype=torch.float32)
        for l in s["labels"]:
            label_vec[self.label_map[l]] = 1.0

        inputs["labels"] = label_vec
        return inputs

"""
    This function computes simple accuracy for multi-label classification.
    It uses sigmoid + thresholding at 0.5.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)

    accuracy = (preds == labels).mean()

    return {"accuracy": accuracy}


def main():
    # This loads the dataset samples
    samples = build_samples(ANNOTATIONS_PATH)

    #We are using a basic train/testing split of 90 percent training and 10 percent testing.
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    # This loads the ViT image processor
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
    # This creates the label mapping
    label_map = build_label_map(samples)

    # This builds the training and validation datasets
    train_dataset = TrafficSignDataset(train_samples, label_map, IMAGE_DIR, processor)
    val_dataset = TrafficSignDataset(val_samples, label_map, IMAGE_DIR, processor)

    # This loads the pretrained ViT model
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_map),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    )
    
    # This maps the labels to model config
    model.config.id2label = {i: l for l, i in label_map.items()}
    model.config.label2id = label_map

    #This is for the training configuration
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="no",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=2e-5,
        remove_unused_columns=False
    )

    # This is the trainer API (from HuggingFace)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    #This is to train the model
    trainer.train()

    #This is to evaluate the model
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()
"""
validate_classifier.py — CLIP Classifier Validation on FairFace val set
========================================================================
After training the CLIP classifier heads, run this to report per-class
accuracy, precision, recall, F1, and confusion matrices on FairFace's
own validation set.

These numbers go directly into the paper's methodology section to
establish how trustworthy the automated CLIP fidelity scorer is.

Usage:
    python eval/clip_trainer/validate_classifier.py

Output:
    eval/clip_trainer/validation_results.json
    Prints confusion matrices and classification reports to stdout.

Prerequisites:
    - CLIP model trained (eval/clip_fidelity_model/*.pt must exist)
    - FairFace val set (eval/clip_trainer/fairface_data/val/ + fairface_label_val.csv)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_fidelity_scorer import CLIPClassifierHead

# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR   = os.path.join(os.path.dirname(__file__), "fairface_data")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "clip_fidelity_model")
CLIP_MODEL = "openai/clip-vit-base-patch32"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


def main():
    val_csv = os.path.join(DATA_DIR, "fairface_label_val.csv")
    val_img = os.path.join(DATA_DIR, "val")

    if not os.path.exists(val_csv):
        print(f"ERROR: FairFace val CSV not found at {val_csv}")
        print("Download FairFace and extract to eval/clip_trainer/fairface_data/")
        sys.exit(1)

    # Load label maps
    label_maps_path = os.path.join(MODEL_DIR, "label_maps.json")
    if not os.path.exists(label_maps_path):
        print(f"ERROR: Trained model not found at {MODEL_DIR}")
        print("Run train_clip_fidelity.py first.")
        sys.exit(1)

    with open(label_maps_path) as f:
        label_maps = json.load(f)

    idx_to_age    = {v: k for k, v in label_maps["age"].items()}
    idx_to_gender = {v: k for k, v in label_maps["gender"].items()}
    idx_to_race   = {v: k for k, v in label_maps["race"].items()}

    # Load models
    print(f"Device: {DEVICE}")
    print("Loading CLIP + classifier heads...")
    processor  = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    clip_model.eval()

    age_head = CLIPClassifierHead(num_classes=len(label_maps["age"])).to(DEVICE)
    age_head.load_state_dict(torch.load(os.path.join(MODEL_DIR, "age_classifier.pt"), map_location=DEVICE))
    age_head.eval()

    gender_head = CLIPClassifierHead(num_classes=len(label_maps["gender"])).to(DEVICE)
    gender_head.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gender_classifier.pt"), map_location=DEVICE))
    gender_head.eval()

    race_head = CLIPClassifierHead(num_classes=len(label_maps["race"])).to(DEVICE)
    race_head.load_state_dict(torch.load(os.path.join(MODEL_DIR, "race_classifier.pt"), map_location=DEVICE))
    race_head.eval()

    # Load val data
    df = pd.read_csv(val_csv)
    print(f"Validation set: {len(df)} images\n")

    all_true_age, all_pred_age = [], []
    all_true_gender, all_pred_gender = [], []
    all_true_race, all_pred_race = [], []

    for start in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[start:start + BATCH_SIZE]
        images = []
        true_ages, true_genders, true_races = [], [], []

        for _, row in batch_df.iterrows():
            img_path = os.path.join(val_img, os.path.basename(str(row["file"])))
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=0)

            images.append(img)
            true_ages.append(str(row["age"]))
            true_genders.append(str(row["gender"]))
            true_races.append(str(row["race"]))

        inputs = processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            emb = clip_model.get_image_features(pixel_values=pixel_values)
            emb = emb / emb.norm(dim=-1, keepdim=True)

            age_preds    = age_head(emb).argmax(dim=-1).cpu().tolist()
            gender_preds = gender_head(emb).argmax(dim=-1).cpu().tolist()
            race_preds   = race_head(emb).argmax(dim=-1).cpu().tolist()

        all_true_age.extend(true_ages)
        all_pred_age.extend([idx_to_age.get(p, "?") for p in age_preds])
        all_true_gender.extend(true_genders)
        all_pred_gender.extend([idx_to_gender.get(p, "?") for p in gender_preds])
        all_true_race.extend(true_races)
        all_pred_race.extend([idx_to_race.get(p, "?") for p in race_preds])

        if (start // BATCH_SIZE + 1) % 20 == 0:
            print(f"  Processed {start + len(batch_df)}/{len(df)} images...")

    # Reports
    results = {}

    for name, y_true, y_pred in [
        ("age", all_true_age, all_pred_age),
        ("gender", all_true_gender, all_pred_gender),
        ("race", all_true_race, all_pred_race),
    ]:
        print(f"\n{'='*60}")
        print(f"  {name.upper()} CLASSIFIER — Validation Results")
        print(f"{'='*60}")

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        print(classification_report(y_true, y_pred, zero_division=0))

        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(f"Confusion matrix (rows=true, cols=predicted):")
        print(f"Labels: {labels}")
        print(cm)

        results[name] = {
            "accuracy": report.get("accuracy", 0),
            "macro_f1": report.get("macro avg", {}).get("f1-score", 0),
            "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0),
            "per_class": {
                k: v for k, v in report.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            },
            "confusion_matrix": cm.tolist(),
            "labels": labels,
        }

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()

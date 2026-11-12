"""
download_fairface.py — Download FairFace from HuggingFace
==========================================================
Downloads the FairFace dataset from HuggingFace and saves it in the
folder structure that train_clip_fidelity.py expects.

Usage:
    python eval/clip_trainer/download_fairface.py

Output:
    eval/clip_trainer/fairface_data/
        train/
            0.jpg, 1.jpg, ...
        val/
            0.jpg, 1.jpg, ...
        fairface_label_train.csv
        fairface_label_val.csv
"""

import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("Installing 'datasets' library...")
    os.system(f"{sys.executable} -m pip install datasets")
    from datasets import load_dataset

import pandas as pd
from PIL import Image


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fairface_data")


def download_and_save():
    print("Downloading FairFace from HuggingFace (this may take a while)...")
    print("Dataset URL: https://huggingface.co/datasets/HuggingFaceM4/FairFace\n")

    ds = load_dataset("HuggingFaceM4/FairFace")

    for split_name, folder_name in [("train", "train"), ("validation", "val")]:
        split = ds[split_name]
        img_dir = os.path.join(DATA_DIR, folder_name)
        os.makedirs(img_dir, exist_ok=True)

        print(f"Processing {split_name} split: {len(split)} images...")

        rows = []
        for i, example in enumerate(split):
            # Save image
            img = example["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            img_filename = f"{i}.jpg"
            img_path = os.path.join(img_dir, img_filename)

            if not os.path.exists(img_path):
                img.convert("RGB").save(img_path, "JPEG")

            # Collect label row
            rows.append({
                "file": f"{folder_name}/{img_filename}",
                "age": example.get("age", ""),
                "gender": example.get("gender", ""),
                "race": example.get("race", ""),
            })

            if (i + 1) % 5000 == 0:
                print(f"  Saved {i + 1}/{len(split)} images...")

        # Handle case where labels are integer indices instead of strings
        # HuggingFace datasets sometimes encode categorical columns as ClassLabel
        df = pd.DataFrame(rows)

        # Check if values are integers (ClassLabel encoded) and decode them
        features = split.features
        for col in ["age", "gender", "race"]:
            if col in features and hasattr(features[col], "names"):
                label_names = features[col].names
                df[col] = df[col].map(lambda x: label_names[x] if isinstance(x, int) and x < len(label_names) else x)

        csv_name = f"fairface_label_{split_name if split_name == 'train' else 'val'}.csv"
        csv_path = os.path.join(DATA_DIR, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_name} ({len(df)} rows)\n")

    print(f"Done! FairFace data saved to: {DATA_DIR}")
    print(f"\nVerify structure:")
    for root, dirs, files in os.walk(DATA_DIR):
        level = root.replace(DATA_DIR, "").count(os.sep)
        indent = "  " * level
        file_count = len(files)
        if file_count > 5:
            print(f"{indent}{os.path.basename(root)}/  ({file_count} files)")
        else:
            print(f"{indent}{os.path.basename(root)}/")
            for f in files:
                print(f"{indent}  {f}")


if __name__ == "__main__":
    download_and_save()

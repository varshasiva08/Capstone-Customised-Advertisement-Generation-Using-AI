"""
CLIP Demographic Fidelity Classifier — Fine-tuning on FairFace
==============================================================
Fine-tunes a CLIP-based classifier head on the FairFace dataset to score
how well a generated image matches an intended demographic profile.

What this does:
    - Takes FairFace images (labeled with age, gender, race)
    - Fine-tunes a small classification head on top of frozen CLIP embeddings
    - Produces 3 classifiers: age_classifier, gender_classifier, race_classifier
    - Saves them to eval/clip_fidelity_model/

What the fidelity scorer then does:
    - Given a generated image + intended profile
    - Runs all 3 classifiers
    - Compares predicted vs intended
    - Returns a fidelity score 0–10

Where to put this file:
    adfidelity/eval/clip_trainer/train_clip_fidelity.py

Dataset:
    FairFace — https://github.com/joojs/fairface
    Download fairface-img-margin025-trainval.zip from their GitHub releases
    Extract to: eval/clip_trainer/fairface_data/

    The folder structure should be:
        eval/clip_trainer/fairface_data/
            train/
                1.jpg, 2.jpg, ...
            val/
                1.jpg, 2.jpg, ...
            fairface_label_train.csv
            fairface_label_val.csv

How to run:
    cd adfidelity/
    pip install torch torchvision transformers pandas scikit-learn
    python eval/clip_trainer/train_clip_fidelity.py

Time:
    ~30–60 minutes on CPU. ~5–10 minutes on GPU.
    The script will print progress every 10 batches.

Output:
    eval/clip_fidelity_model/
        age_classifier.pt
        gender_classifier.pt
        race_classifier.pt
        label_maps.json     ← maps class indices to label names
"""

import json
import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR    = os.path.join(os.path.dirname(__file__), "fairface_data")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "clip_fidelity_model")
CLIP_MODEL  = "openai/clip-vit-base-patch32"   # ~600MB, downloads automatically
BATCH_SIZE  = 32
EPOCHS      = 5
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# FairFace label columns we care about
AGE_COL     = "age"
GENDER_COL  = "gender"
RACE_COL    = "race"

# ── Dataset ──────────────────────────────────────────────────────────────────

class FairFaceDataset(Dataset):
    def __init__(self, csv_path: str, img_root: str, processor, label_maps: dict):
        self.df        = pd.read_csv(csv_path)
        self.img_root  = img_root
        self.processor = processor
        self.label_maps = label_maps

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["file"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=0)

        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].squeeze(0)

        age_label    = self.label_maps["age"].get(str(row[AGE_COL]), 0)
        gender_label = self.label_maps["gender"].get(str(row[GENDER_COL]), 0)
        race_label   = self.label_maps["race"].get(str(row[RACE_COL]), 0)

        return pixel_values, age_label, gender_label, race_label


# ── Classifier head ───────────────────────────────────────────────────────────

class CLIPClassifierHead(nn.Module):
    """
    Small MLP head on top of frozen CLIP image embeddings.
    CLIP produces 512-dim embeddings. We add 2 linear layers.
    """
    def __init__(self, num_classes: int, embed_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_label_maps(csv_path: str) -> dict:
    """Build string→int label maps from the CSV."""
    df = pd.read_csv(csv_path)
    maps = {}
    for col in [AGE_COL, GENDER_COL, RACE_COL]:
        unique = sorted(df[col].dropna().unique().tolist())
        maps[col] = {str(v): i for i, v in enumerate(unique)}
    return maps


def train_one_head(
    loader,
    clip_model,
    head: CLIPClassifierHead,
    label_idx: int,
    label_name: str,
    epochs: int,
    lr: float,
):
    """Train one classifier head (age, gender, or race)."""
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    clip_model.eval()  # CLIP stays frozen
    head.train()

    for epoch in range(epochs):
        total_loss = 0.0
        correct    = 0
        total      = 0

        for batch_idx, batch in enumerate(loader):
            pixel_values = batch[0].to(DEVICE)
            labels       = batch[label_idx + 1].to(DEVICE)   # +1 because batch[0] is pixels

            with torch.no_grad():
                embeddings = clip_model.get_image_features(pixel_values=pixel_values)
                if hasattr(embeddings, "image_embeds"):
                    embeddings = embeddings.image_embeds
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            logits = head(embeddings)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds       = logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  [{label_name}] Epoch {epoch+1}/{epochs} "
                    f"Batch {batch_idx+1}/{len(loader)} "
                    f"Loss: {total_loss/(batch_idx+1):.4f} "
                    f"Acc: {correct/total*100:.1f}%"
                )

        print(
            f"  [{label_name}] Epoch {epoch+1} done — "
            f"Loss: {total_loss/len(loader):.4f} "
            f"Acc: {correct/total*100:.1f}%\n"
        )

    return head


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_csv = os.path.join(DATA_DIR, "fairface_label_train.csv")
    val_csv   = os.path.join(DATA_DIR, "fairface_label_val.csv")
    train_img = os.path.join(DATA_DIR, "train")

    # Verify data exists
    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"FairFace training CSV not found at {train_csv}\n"
            "Download FairFace from https://github.com/joojs/fairface\n"
            "and extract to eval/clip_trainer/fairface_data/"
        )

    print(f"Using device: {DEVICE}")
    print("Loading CLIP model (downloads ~600MB on first run)...")

    processor  = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)

    # Freeze all CLIP parameters — we only train the heads
    for param in clip_model.parameters():
        param.requires_grad = False

    print("Building label maps...")
    label_maps = build_label_maps(train_csv)

    # Save label maps so the scorer can use them
    label_maps_path = os.path.join(OUT_DIR, "label_maps.json")
    with open(label_maps_path, "w") as f:
        json.dump(label_maps, f, indent=2)
    print(f"Label maps saved to {label_maps_path}")
    print(f"  Age classes:    {list(label_maps[AGE_COL].keys())}")
    print(f"  Gender classes: {list(label_maps[GENDER_COL].keys())}")
    print(f"  Race classes:   {list(label_maps[RACE_COL].keys())}\n")

    print("Building dataset and dataloader...")
    dataset = FairFaceDataset(train_csv, train_img, processor, label_maps)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"Training on {len(dataset)} images in {len(loader)} batches.\n")

    # ── Train age classifier ──────────────────────────────────────────────────
    print("=== Training Age Classifier ===")
    age_head = CLIPClassifierHead(num_classes=len(label_maps[AGE_COL])).to(DEVICE)
    age_head = train_one_head(loader, clip_model, age_head, 0, "age", EPOCHS, LR)
    torch.save(age_head.state_dict(), os.path.join(OUT_DIR, "age_classifier.pt"))
    print(f"Age classifier saved.\n")

    # ── Train gender classifier ───────────────────────────────────────────────
    print("=== Training Gender Classifier ===")
    gender_head = CLIPClassifierHead(num_classes=len(label_maps[GENDER_COL])).to(DEVICE)
    gender_head = train_one_head(loader, clip_model, gender_head, 1, "gender", EPOCHS, LR)
    torch.save(gender_head.state_dict(), os.path.join(OUT_DIR, "gender_classifier.pt"))
    print(f"Gender classifier saved.\n")

    # ── Train race classifier ─────────────────────────────────────────────────
    print("=== Training Race Classifier ===")
    race_head = CLIPClassifierHead(num_classes=len(label_maps[RACE_COL])).to(DEVICE)
    race_head = train_one_head(loader, clip_model, race_head, 2, "race", EPOCHS, LR)
    torch.save(race_head.state_dict(), os.path.join(OUT_DIR, "race_classifier.pt"))
    print(f"Race classifier saved.\n")

    print("=== Training complete ===")
    print(f"All models saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

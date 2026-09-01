"""
eval/embedding_viz.py — PCA / t-SNE plot of CLIP embeddings

Supporting figure only (not evidence on its own — pair it with the
bootstrap CI numbers for the actual statistical claim). Shows whether
CPDC-corrected generated images sit closer to real demographic-group
clusters in CLIP embedding space than uncorrected ones do.

Where to put this file:
    eval/embedding_viz.py

Requires:
    pip install scikit-learn matplotlib
    (torch, transformers, PIL already required by clip_fidelity_scorer.py)

Expects two image sources:
    --generated-dir   folder of your generated images (filenames should
                       start with the target ethnicity, matching how
                       run_full_evaluation.py names them, e.g.
                       "South-Asian_slim_20s_seed42_cdvr.png")
    --real-dir        a folder of real reference images, one subfolder
                       per ethnicity label, e.g.
                       real_refs/South Asian/*.jpg
                       real_refs/East Asian/*.jpg
                       real_refs/African American/*.jpg
                       (a small sample from FairFace works fine here --
                       reuse whatever you already pulled for the FID/KID
                       study in capstone-eval.ipynb)

How to run:
    cd <repo root>
    python eval/embedding_viz.py \
        --generated-dir eval/results/cpdc_vs_cdvr/generated_images \
        --real-dir eval/real_refs \
        --method both

Output:
    eval/results/embedding_pca.png
    eval/results/embedding_tsne.png
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ETHNICITY_FROM_FILENAME = re.compile(
    r"^(South-Asian|East-Asian|African-American)", re.IGNORECASE
)
FILENAME_TO_LABEL = {
    "south-asian": "South Asian",
    "east-asian": "East Asian",
    "african-american": "African American",
}


def load_clip():
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return model, processor


@torch.no_grad()
def embed_image(path, model, processor):
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    features = model.get_image_features(pixel_values=pixel_values)
    return features.squeeze(0).cpu().numpy()


def collect_generated(generated_dir, model, processor):
    records = []
    for path in sorted(glob.glob(os.path.join(generated_dir, "*.png"))) + \
                sorted(glob.glob(os.path.join(generated_dir, "*.jpg"))):
        fname = os.path.basename(path)
        match = ETHNICITY_FROM_FILENAME.match(fname)
        if not match:
            continue
        label = FILENAME_TO_LABEL[match.group(1).lower()]
        corrected = "cdvr" in fname.lower() or "cpdc" in fname.lower()
        emb = embed_image(path, model, processor)
        records.append({
            "embedding": emb,
            "ethnicity": label,
            "source": "generated_corrected" if corrected else "generated_baseline",
        })
    return records


def collect_real(real_dir, model, processor, max_per_class=40):
    records = []
    for label_dir in sorted(glob.glob(os.path.join(real_dir, "*"))):
        if not os.path.isdir(label_dir):
            continue
        label = os.path.basename(label_dir)
        files = sorted(glob.glob(os.path.join(label_dir, "*.jpg"))) + \
                sorted(glob.glob(os.path.join(label_dir, "*.png")))
        for path in files[:max_per_class]:
            emb = embed_image(path, model, processor)
            records.append({"embedding": emb, "ethnicity": label, "source": "real"})
    return records


def plot(records, method, output_path):
    embeddings = np.stack([r["embedding"] for r in records])
    if method == "pca":
        reducer = PCA(n_components=2, random_state=0)
    else:
        reducer = TSNE(n_components=2, random_state=0, perplexity=min(30, len(records) - 1))
    coords = reducer.fit_transform(embeddings)

    ethnicities = sorted(set(r["ethnicity"] for r in records))
    sources = ["real", "generated_baseline", "generated_corrected"]
    color_map = {e: c for e, c in zip(ethnicities, plt.cm.Set1.colors)}
    marker_map = {"real": "o", "generated_baseline": "x", "generated_corrected": "^"}

    plt.figure(figsize=(8, 6))
    for src in sources:
        for eth in ethnicities:
            pts = [coords[i] for i, r in enumerate(records)
                   if r["source"] == src and r["ethnicity"] == eth]
            if not pts:
                continue
            pts = np.array(pts)
            plt.scatter(pts[:, 0], pts[:, 1],
                        c=[color_map[eth]], marker=marker_map[src],
                        label=f"{eth} ({src})", alpha=0.7, s=40)

    plt.legend(fontsize=7, loc="best", ncol=1)
    plt.title(f"CLIP embedding space ({method.upper()})")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--method", choices=["pca", "tsne", "both"], default="both")
    parser.add_argument("--output-dir", default="eval/results")
    parser.add_argument("--max-per-class", type=int, default=40,
                         help="cap on real reference images per ethnicity, for speed")
    args = parser.parse_args()

    model, processor = load_clip()

    print("Embedding generated images...")
    records = collect_generated(args.generated_dir, model, processor)
    print(f"  {len(records)} generated images embedded")

    print("Embedding real reference images...")
    records += collect_real(args.real_dir, model, processor, args.max_per_class)
    print(f"  total {len(records)} images embedded")

    if len(records) < 5:
        raise RuntimeError("Too few images embedded — check --generated-dir and "
                            "--real-dir paths and filename conventions.")

    os.makedirs(args.output_dir, exist_ok=True)
    methods = ["pca", "tsne"] if args.method == "both" else [args.method]
    for m in methods:
        plot(records, m, os.path.join(args.output_dir, f"embedding_{m}.png"))


if __name__ == "__main__":
    main()

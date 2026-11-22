"""Second-verifier check: CLIP zero-shot ethnicity/gender on the 145 rated images,
compared against the human labels. Run from the adfidelity folder.

  python clip_verifier_check.py
  python clip_verifier_check.py --model openai/clip-vit-base-patch32     (smaller/faster)
  python clip_verifier_check.py --no-crop                                (skip face crop)

Writes clip_scores.csv (send this back) and prints a summary.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--rating-dir", default=r"results\main\rating")
ap.add_argument("--ratings", default="ratings_person1.csv")
ap.add_argument("--key", default=os.path.join("results", "main", "rating", "rating_key_DO_NOT_SHARE.csv"))
ap.add_argument("--model", default="openai/clip-vit-large-patch14")
ap.add_argument("--out", default="clip_scores.csv")
ap.add_argument("--no-crop", action="store_true")
args = ap.parse_args()

ETH = {  # brief label -> text used for CLIP
    "South Asian": "a South Asian (Indian) person",
    "African American": "a Black (African American) person",
    "East Asian": "an East Asian person",
    "Latin American": "a Latino (Latin American) person",
    "Middle Eastern": "a Middle Eastern person",
    "White": "a White (European) person",
    "Southeast Asian": "a Southeast Asian person",
}
TEMPLATES = ["a photo of {}", "a close-up portrait photo of {}"]
GEN = {"female": "a woman", "male": "a man"}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(args.model).to(device).eval()
proc = CLIPProcessor.from_pretrained(args.model)


_size = model.config.vision_config.image_size
_dummy_pix = torch.zeros(1, 3, _size, _size).to(device)
_dummy_txt = proc(text=["a"], return_tensors="pt", padding=True).to(device)


def _text_features(inp):
    # full forward pass: returns the projected text embedding in every transformers version
    out = model(**inp, pixel_values=_dummy_pix)
    return out.text_embeds


def _image_features(inp):
    out = model(pixel_values=inp["pixel_values"], **_dummy_txt)
    return out.image_embeds


def text_emb(phrases):
    embs = []
    for ph in phrases:
        vs = []
        for t in TEMPLATES:
            inp = proc(text=[t.format(ph)], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                e = _text_features(inp)
            vs.append(torch.nn.functional.normalize(e, dim=-1))
        embs.append(torch.nn.functional.normalize(torch.stack(vs).mean(0), dim=-1))
    return torch.cat(embs)


eth_names = list(ETH)
eth_txt = text_emb([ETH[n] for n in eth_names])
gen_names = list(GEN)
gen_txt = text_emb([GEN[n] for n in gen_names])

extract_faces = None
if not args.no_crop:
    try:
        from deepface import DeepFace
        extract_faces = DeepFace.extract_faces
    except Exception as e:  # noqa
        print("face crop unavailable, using full images:", e)


def load(path):
    img = Image.open(path).convert("RGB")
    if extract_faces is None:
        return img, 0
    try:
        faces = extract_faces(img_path=path, detector_backend="retinaface", enforce_detection=False)
        f = max(faces, key=lambda d: d["facial_area"]["w"] * d["facial_area"]["h"])["facial_area"]
        if f["w"] < 20 or f["h"] < 20:
            return img, 0
        m = 0.35
        x0 = max(0, int(f["x"] - m * f["w"])); y0 = max(0, int(f["y"] - m * f["h"]))
        x1 = min(img.width, int(f["x"] + (1 + m) * f["w"])); y1 = min(img.height, int(f["y"] + (1 + m) * f["h"]))
        return img.crop((x0, y0, x1, y1)), 1
    except Exception:
        return img, 0


ratings = pd.read_csv(args.ratings)
key = pd.read_csv(args.key)
rows = []
for iid in ratings.image_id:
    p = os.path.join(args.rating_dir, "images", f"{iid}.png")
    if not os.path.exists(p):
        sys.exit(f"missing image {p}")
    img, cropped = load(p)
    inp = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        v = torch.nn.functional.normalize(_image_features(inp), dim=-1)
        pe = (100 * v @ eth_txt.T).softmax(-1)[0].cpu().numpy()
        pg = (100 * v @ gen_txt.T).softmax(-1)[0].cpu().numpy()
    r = {"image_id": iid, "cropped": cropped}
    r.update({f"p_eth_{n}": float(x) for n, x in zip(eth_names, pe)})
    r.update({f"p_gen_{n}": float(x) for n, x in zip(gen_names, pg)})
    rows.append(r)
    if len(rows) % 20 == 0:
        print(len(rows), "/", len(ratings))
S = pd.DataFrame(rows)
S.to_csv(args.out, index=False)

# ---- quick evaluation against the human labels (target = key file)
D = ratings.merge(key, on="image_id").merge(S, on="image_id")
D["h_e"] = (D.perceived_ethnicity == D.ethnicity).astype(int)
D["h_g"] = (D.perceived_gender == D.gender).astype(int)
D["p_t"] = [r[f"p_eth_{r['ethnicity']}"] for _, r in D.iterrows()]
D["top1"] = D[[f"p_eth_{n}" for n in eth_names]].idxmax(axis=1).str.replace("p_eth_", "")
D["clip_pass"] = (D.top1 == D.ethnicity).astype(int)
D["g_top"] = D[[f"p_gen_{n}" for n in gen_names]].idxmax(axis=1).str.replace("p_gen_", "")
D["clip_g_pass"] = (D.g_top == D.gender).astype(int)


def auc(score, y):
    y = np.asarray(y); s = np.asarray(score)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(pp > nn) + 0.5 * (pp == nn) for pp in pos for nn in neg]))


print("\n=== CLIP vs human labels (cropped faces: %d/%d) ===" % (D.cropped.sum(), len(D)))
print("ethnicity: CLIP-pass vs human match  agreement=%.3f  AUC(P_target)=%.3f" % ((D.clip_pass == D.h_e).mean(), auc(D.p_t, D.h_e)))
print("gender:    CLIP-pass vs human match  agreement=%.3f" % ((D.clip_g_pass == D.h_g).mean()))
print("\nper target ethnicity:  n | human says target | CLIP top1==target | agree | AUC(P_target)")
for g, s in D.groupby("ethnicity"):
    print("%-17s %3d | %.2f | %.2f | %.2f | %.2f" % (g, len(s), s.h_e.mean(), s.clip_pass.mean(), (s.clip_pass == s.h_e).mean(), auc(s.p_t, s.h_e)))
sa = D[D.ethnicity == "South Asian"]
print("\nSouth Asian targets: human-yes %d, human-no %d" % (sa.h_e.sum(), (1 - sa.h_e).sum()))
print("  CLIP passes %d of %d human-yes (sens %.2f); passes %d of %d human-no (false pass)" % (
    sa[sa.h_e == 1].clip_pass.sum(), sa.h_e.sum(), sa[sa.h_e == 1].clip_pass.mean(),
    sa[sa.h_e == 0].clip_pass.sum(), (1 - sa.h_e).sum()))
print("  (DeepFace reference on the same images: sens 0.22, AUC 0.54, kappa 0.06)")
print("\nSaved", args.out)
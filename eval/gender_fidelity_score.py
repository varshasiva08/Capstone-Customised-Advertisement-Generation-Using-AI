# eval/gender_fidelity_score.py
import os, torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def score_gender(image_path, target_gender="female"):
    image  = Image.open(image_path).convert("RGB")
    labels = ["a woman", "a man"]
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image.softmax(dim=1)[0]
    female_score = float(logits[0])
    male_score   = float(logits[1])
    match = (female_score > male_score) if target_gender == "female" else (male_score > female_score)
    return {
        "female_confidence": round(female_score, 3),
        "male_confidence":   round(male_score, 3),
        "gender_match":      match,
    }

# Score your images
MALE_FOLDER   = "eval/results/male_images"    # put your male images here
FEMALE_FOLDER = "eval/results/female_images"  # put your female images here

for folder, gender in [(FEMALE_FOLDER, "female"), (MALE_FOLDER, "male")]:
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue
    images = [f for f in os.listdir(folder) if f.endswith(('.jpg','.png'))]
    print(f"\n=== {gender.upper()} ({len(images)} images) ===")
    matches = []
    for fname in sorted(images):
        result = score_gender(os.path.join(folder, fname), gender)
        matches.append(result["gender_match"])
        print(f"  {fname}: match={result['gender_match']} female={result['female_confidence']} male={result['male_confidence']}")
    print(f"  Gender accuracy: {sum(matches)/len(matches):.1%}")
# save as eval/clip_zeroshot_score.py
import os, json, torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

CONFIGS = {
    'config_a_baseline': 'Baseline FLUX (generic Indian prompt)',
    'config_b_prompt':   'FLUX + Regional South Indian Prompt',
}
STUDY_DIR = 'eval/results/regional_prompt_study'

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

RACE_LABELS = [
    "a South Asian person", "an East Asian person",
    "an African American person", "a White person",
    "a Latino Hispanic person", "a Middle Eastern person",
    "a Southeast Asian person"
]

results = {}
for config_key, config_label in CONFIGS.items():
    config_dir = os.path.join(STUDY_DIR, config_key)
    images = sorted([f for f in os.listdir(config_dir) 
                     if f.endswith('.jpg') or f.endswith('.png')])
    print(f"\n=== {config_label} ({len(images)} images) ===")
    
    race_matches = []
    confidences  = []
    
    for i, fname in enumerate(images):
        image = Image.open(os.path.join(config_dir, fname)).convert("RGB")
        inputs = processor(
            text=RACE_LABELS, images=image,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = model(**inputs).logits_per_image.softmax(dim=1)[0]
        
        sa_score  = float(logits[0])
        predicted = RACE_LABELS[logits.argmax().item()]
        match     = logits.argmax().item() == 0
        
        race_matches.append(match)
        confidences.append(sa_score)
        print(f"  [{i+1}/{len(images)}] {fname}: SA={sa_score:.3f} match={match} predicted={predicted[:20]}")
    
    acc  = sum(race_matches) / len(race_matches)
    conf = sum(confidences)  / len(confidences)
    results[config_key] = {
        'label':        config_label,
        'race_accuracy': round(acc, 3),
        'mean_sa_confidence': round(conf, 3),
        'n': len(images)
    }
    print(f"  Race accuracy: {acc:.1%} | Mean SA confidence: {conf:.3f}")

print("\n=== FINAL RESULTS ===")
for k, v in results.items():
    print(f"{v['label']}")
    print(f"  Race accuracy: {v['race_accuracy']:.1%}")
    print(f"  Mean SA confidence: {v['mean_sa_confidence']:.3f}")

with open(os.path.join(STUDY_DIR, 'clip_zeroshot_scores.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {STUDY_DIR}/clip_zeroshot_scores.json")
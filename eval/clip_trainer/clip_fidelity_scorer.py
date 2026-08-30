"""
CLIP Demographic Fidelity Scorer
=================================
Uses the fine-tuned CLIP classifier heads to score how well a generated
image matches its intended demographic profile.

Replaces / complements the LLaVA-based fidelity_scorer.py.
No Ollama needed - runs entirely from saved .pt files.

Feeds race_target_confidence into CPDC (modules/cpdc.py) as the
ethnicity-axis error signal - CLIP is the more reliable judge on that
axis (96.3% vs LLaVA's 89.8%, see EVALUATION_RESULTS.md Table 3).

Where to put this file:
    adfidelity/eval/clip_fidelity_scorer.py

How to run (standalone test):
    cd adfidelity/
    python eval/clip_fidelity_scorer.py \
        --image outputs/your_image.png \
        --ethnicity "South Asian" \
        --age "30s" \
        --gender "Female"

Output:
    {
        "age_match":       true,
        "gender_match":    true,
        "race_match":      true,
        "age_score":       10.0,
        "gender_score":    10.0,
        "race_score":      10.0,
        "overall_fidelity": 10.0,
        "age_target_confidence": 0.91,
        "race_target_confidence": 0.94,
        "predicted": {
            "age":    "30-39",
            "gender": "Female",
            "race":   "Indian"
        }
    }
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# -- Config -------------------------------------------------------------------

CLIP_MODEL  = "openai/clip-vit-base-patch32"
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "clip_fidelity_model")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Map from FairFace race labels -> your config.yaml ethnicity labels
RACE_TO_ETHNICITY = {
    "Indian":           "South Asian",
    "Southeast Asian":  "South Asian",
    "Middle Eastern":   None,               # no direct mapping - excluded from scoring
    "East Asian":       "East Asian",
    "Black":            "African American",
    "White":            None,               # no direct mapping - excluded from scoring
    "Latino_Hispanic":  None,               # no direct mapping - excluded from scoring
}

# Map from FairFace age labels -> your config.yaml age labels
AGE_TO_CONFIG = {
    "0-2":   "20s",
    "3-9":   "20s",
    "10-19": "20s",
    "20-29": "20s",
    "30-39": "30s",
    "40-49": "40s",
    "50-59": "50s",
    "60-69": "50s",
    "more than 70": "50s",
}


# -- Classifier head (must match train_clip_fidelity.py) ----------------------

class CLIPClassifierHead(nn.Module):
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


# -- Scorer class ---------------------------------------------------------------

class CLIPFidelityScorer:
    """
    Loads the fine-tuned CLIP classifier heads and scores a generated image
    against an intended demographic profile.
    """

    def __init__(self):
        label_maps_path = os.path.join(MODEL_DIR, "label_maps.json")
        if not os.path.exists(label_maps_path):
            raise FileNotFoundError(
                f"Fine-tuned model not found at {MODEL_DIR}\n"
                "Run eval/clip_trainer/train_clip_fidelity.py first."
            )

        with open(label_maps_path) as f:
            self.label_maps = json.load(f)

        # Invert maps: int -> string label
        self.idx_to_age    = {v: k for k, v in self.label_maps["age"].items()}
        self.idx_to_gender = {v: k for k, v in self.label_maps["gender"].items()}
        self.idx_to_race   = {v: k for k, v in self.label_maps["race"].items()}

        print("Loading CLIP base model...")
        self.processor  = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
        self.clip_model.eval()

        print("Loading fine-tuned classifier heads...")
        self.age_head = CLIPClassifierHead(
            num_classes=len(self.label_maps["age"])
        ).to(DEVICE)
        self.age_head.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "age_classifier.pt"),
                       map_location=DEVICE)
        )
        self.age_head.eval()

        self.gender_head = CLIPClassifierHead(
            num_classes=len(self.label_maps["gender"])
        ).to(DEVICE)
        self.gender_head.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "gender_classifier.pt"),
                       map_location=DEVICE)
        )
        self.gender_head.eval()

        self.race_head = CLIPClassifierHead(
            num_classes=len(self.label_maps["race"])
        ).to(DEVICE)
        self.race_head.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "race_classifier.pt"),
                       map_location=DEVICE)
        )
        self.race_head.eval()

        print("CLIP Fidelity Scorer ready.\n")

    def score(self, image_path: str, profile: dict) -> dict:
        """
        Score a generated image against an intended demographic profile.

        Args:
            image_path: Path to the generated image file.
            profile:    Dict with keys: ethnicity, age, gender (optional).
                        Uses your config.yaml label format.
                        e.g. {"ethnicity": "South Asian", "age": "30s", "gender": "Female"}

        Returns:
            Dict with per-axis match booleans, scores, target-class
            confidences (for CPDC), and overall fidelity.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            embedding = self.clip_model.get_image_features(pixel_values=pixel_values)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            age_logits    = self.age_head(embedding)
            gender_logits = self.gender_head(embedding)
            race_logits   = self.race_head(embedding)

        # Predicted labels
        pred_age_idx    = age_logits.argmax(dim=-1).item()
        pred_gender_idx = gender_logits.argmax(dim=-1).item()
        pred_race_idx   = race_logits.argmax(dim=-1).item()

        pred_age_ff    = self.idx_to_age.get(pred_age_idx, "unknown")
        pred_gender_ff = self.idx_to_gender.get(pred_gender_idx, "unknown")
        pred_race_ff   = self.idx_to_race.get(pred_race_idx, "unknown")

        # Convert FairFace labels -> your config labels for comparison
        pred_age_cfg  = AGE_TO_CONFIG.get(pred_age_ff, "unknown")
        pred_race_cfg = RACE_TO_ETHNICITY.get(pred_race_ff, "unknown")

        # Intended values from profile
        intended_age       = profile.get("age", "").lower().strip()
        intended_ethnicity = profile.get("ethnicity", "").strip()
        intended_gender    = profile.get("gender", "Female").strip()

        # Match checks
        age_match    = (pred_age_cfg.lower() == intended_age.lower())
        gender_match = (pred_gender_ff.lower() == intended_gender.lower())
        # If FairFace predicted a race with no mapping to our labels, mark as unmapped
        race_unmapped = (pred_race_cfg is None)
        race_match   = False if race_unmapped else (pred_race_cfg.lower() == intended_ethnicity.lower())

        # Scores: 10 if match, partial credit from softmax confidence otherwise
        age_probs    = torch.softmax(age_logits, dim=-1)
        gender_probs = torch.softmax(gender_logits, dim=-1)
        race_probs   = torch.softmax(race_logits, dim=-1)

        age_conf    = age_probs.max().item()
        gender_conf = gender_probs.max().item()
        race_conf   = race_probs.max().item()

        age_score    = 10.0 if age_match    else round(age_conf * 5, 2)
        gender_score = 10.0 if gender_match else round(gender_conf * 5, 2)
        race_score   = 10.0 if race_match   else round(race_conf * 5, 2)

        overall = round((age_score + gender_score + race_score) / 3, 2)

        # NEW - confidence in the INTENDED target class, not just top-1.
        # This is what CPDC uses as its error signal: top-1 confidence is
        # meaningless once the prediction is wrong (it tells you how sure
        # the model was about the WRONG answer, not how close it is to
        # the right one). Several FairFace bins can map to one config
        # label (e.g. Indian + Southeast Asian both -> South Asian), so
        # this sums probability mass across all matching indices.
        age_target_indices = [i for i, lbl in self.idx_to_age.items()
                               if AGE_TO_CONFIG.get(lbl) == intended_age]
        race_target_indices = [i for i, lbl in self.idx_to_race.items()
                                if RACE_TO_ETHNICITY.get(lbl) == intended_ethnicity]

        age_target_confidence = (
            age_probs[0, age_target_indices].sum().item() if age_target_indices else 0.0
        )
        race_target_confidence = (
            race_probs[0, race_target_indices].sum().item() if race_target_indices else 0.0
        )

        return {
            "age_match":        age_match,
            "gender_match":     gender_match,
            "race_match":       race_match,
            "age_score":        age_score,
            "gender_score":     gender_score,
            "race_score":       race_score,
            "overall_fidelity": overall,
            "age_target_confidence":  age_target_confidence,   # NEW - for CPDC
            "race_target_confidence": race_target_confidence,  # NEW - for CPDC
            "predicted": {
                "age":    pred_age_ff,
                "gender": pred_gender_ff,
                "race":   pred_race_ff,
            }
        }


# -- CLI runner -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a generated image for demographic fidelity.")
    parser.add_argument("--image",     required=True,  help="Path to generated image")
    parser.add_argument("--ethnicity", required=True,  help="Intended ethnicity (e.g. 'South Asian')")
    parser.add_argument("--age",       required=True,  help="Intended age group (e.g. '30s')")
    parser.add_argument("--gender",    default="Female", help="Intended gender (default: Female)")
    args = parser.parse_args()

    scorer = CLIPFidelityScorer()
    result = scorer.score(
        image_path=args.image,
        profile={
            "ethnicity": args.ethnicity,
            "age":       args.age,
            "gender":    args.gender,
        }
    )

    print("\n=== Fidelity Score ===")
    print(json.dumps(result, indent=2))
    print(f"\nOverall fidelity: {result['overall_fidelity']}/10")

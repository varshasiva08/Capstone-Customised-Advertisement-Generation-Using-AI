"""
OPTIONAL second scorer: zero-shot CLIP (not an LLM; a contrastive image-text model).

Why it exists: results should be reported with a scorer that is NOT the one that
drove the correction loop. CLIP zero-shot is weak on age (36% on FairFace in the
earlier study) and moderate on race (63%), so treat it as a cross-check, not
ground truth - humans are the ground truth (scoring/human.py).

Needs `torch` + `transformers` and a one-off download of openai/clip-vit-base-patch32
from Hugging Face. NOT exercised in the CI sandbox this repo was refactored in
(no Hugging Face access there) - run `pytest tests/test_scoring.py -k clip` locally.
"""
from __future__ import annotations

from .types import FaceReading

_RACE_PROMPTS = {
    "indian": ["a photo of a South Asian person"],
    "asian": ["a photo of an East Asian person", "a photo of a Southeast Asian person"],
    "black": ["a photo of a Black person"],
    "white": ["a photo of a White person"],
    "middle eastern": ["a photo of a Middle Eastern person"],
    "latino hispanic": ["a photo of a Latino or Hispanic person"],
}
_GENDER_PROMPTS = {"Woman": "a photo of a woman", "Man": "a photo of a man"}
_AGE_CENTRES = {"a photo of a person in their 20s": 25, "a photo of a person in their 30s": 35,
                "a photo of a person in their 40s": 45, "a photo of a person in their 50s": 55,
                "a photo of a person in their 60s": 65}


class ClipZeroShotScorer:
    name = "clip"

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.model_id = model_id
        self._m = self._p = None

    def _load(self):
        if self._m is None:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor
            self._m = CLIPModel.from_pretrained(self.model_id).eval()
            self._p = CLIPProcessor.from_pretrained(self.model_id)

    def _probs(self, image, prompts):
        import torch
        inputs = self._p(text=prompts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            return self._m(**inputs).logits_per_image.softmax(dim=1)[0].tolist()

    def read(self, image) -> FaceReading:
        from PIL import Image
        self._load()
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        flat = [(k, p) for k, ps in _RACE_PROMPTS.items() for p in ps]
        pr = self._probs(image, [p for _, p in flat])
        race = {}
        for (k, _), v in zip(flat, pr):
            race[k] = race.get(k, 0.0) + 100 * v
        g = self._probs(image, list(_GENDER_PROMPTS.values()))
        gender = {k: 100 * v for k, v in zip(_GENDER_PROMPTS, g)}
        a = self._probs(image, list(_AGE_CENTRES))
        age = sum(v * c for v, c in zip(a, _AGE_CENTRES.values()))
        # CLIP has no face detector here: it scores the whole image.
        return FaceReading(True, 1.0, None, age, gender, race, self.name, 1)

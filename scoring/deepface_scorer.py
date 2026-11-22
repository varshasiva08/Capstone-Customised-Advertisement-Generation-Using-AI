"""
DeepFace-based attribute reader (age / gender / race) - the default verifier.

* Not an LLM / VLM. Classic CNN classifiers (weights are fetched once from the
  deepface GitHub releases into ~/.deepface).
* Face detection: RetinaFace (robust on the small faces in full-body ads).
* Reads ONLY the image - the target profile is never shown to the model, so it
  cannot be led towards the answer (unlike a VLM prompted with the target).

Known limits (state them in the paper):
  - race head has 6 classes; East Asian and Southeast Asian share one class.
  - age MAE is several years and biased by apparent demographics; treat as a
    noisy measurement and validate against human ratings.
  - classifiers can themselves be biased across groups; report accuracy per group.
"""
from __future__ import annotations

import os
import threading

import numpy as np
from PIL import Image

from .types import FaceReading

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def _to_bgr(image) -> np.ndarray:
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    arr = np.asarray(image.convert("RGB"))
    return arr[:, :, ::-1].copy()          # RGB -> BGR (DeepFace/OpenCV convention)


class DeepFaceScorer:
    name = "deepface"
    version = "+crop1"           # bump when preprocessing changes (cache key)

    def __init__(self, detector: str = "retinaface", min_face_confidence: float = 0.80):
        self.detector = detector
        self.min_face_confidence = min_face_confidence
        self._lock = threading.Lock()
        self._DeepFace = None

    def _lib(self):
        if self._DeepFace is None:
            from deepface import DeepFace      # heavy import, do it lazily
            self._DeepFace = DeepFace
        return self._DeepFace

    def warmup(self):
        """Build all models once (downloads weights on first run)."""
        blank = Image.new("RGB", (256, 256), "white")
        self.read(blank)

    def _refine(self, image, reg, margin=0.45, min_side=360):
        """Second pass for full-body shots: the face is tiny, so crop it with a
        margin, enlarge it and read the attributes again. Returns the analysis
        dict of the crop, or None to fall back to the first pass."""
        try:
            if isinstance(image, (str, os.PathLike)):
                image = Image.open(image)
            img = image.convert("RGB")
            x, y, w, h = (int(reg[k]) for k in ("x", "y", "w", "h"))
            if w < 8 or h < 8:
                return None
            mx, my = int(w * margin), int(h * margin)
            box = (max(0, x - mx), max(0, y - my),
                   min(img.width, x + w + mx), min(img.height, y + h + my))
            crop = img.crop(box)
            scale = max(1.0, min_side / min(crop.size))
            if scale > 1.0:
                crop = crop.resize((int(crop.width * scale), int(crop.height * scale)),
                                   Image.LANCZOS)
            with self._lock:
                out = self._DeepFace.analyze(
                    _to_bgr(crop), actions=["age", "gender", "race"],
                    detector_backend=self.detector, enforce_detection=False,
                    align=True, silent=True)
            out = [out] if isinstance(out, dict) else out
            out = [o for o in out if float(o.get("face_confidence", 0.0)) >= self.min_face_confidence]
            if not out:
                return None
            best = max(out, key=lambda d: d["region"]["w"] * d["region"]["h"])
            best["region"] = reg                  # keep the box in source-image coordinates
            return best
        except Exception:                          # noqa: BLE001
            return None

    def read(self, image) -> FaceReading:
        DeepFace = self._lib()
        bgr = _to_bgr(image)
        with self._lock:
            try:
                faces = DeepFace.analyze(
                    bgr, actions=["age", "gender", "race"],
                    detector_backend=self.detector, enforce_detection=False,
                    align=True, silent=True)
            except Exception as e:             # noqa: BLE001 - treat as "no face"
                return FaceReading(False, scorer=self.name, n_faces=0)
        if isinstance(faces, dict):
            faces = [faces]
        good = [f for f in faces
                if float(f.get("face_confidence", 0.0)) >= self.min_face_confidence]
        if not good:
            return FaceReading(False, scorer=self.name, n_faces=0,
                               face_confidence=max([float(f.get("face_confidence", 0.0))
                                                    for f in faces] or [0.0]))
        # one subject per ad: use the largest face
        f = max(good, key=lambda d: d["region"]["w"] * d["region"]["h"])
        reg = f["region"]
        f = self._refine(image, reg) or f
        return FaceReading(
            True, float(f["face_confidence"]),
            (int(reg["x"]), int(reg["y"]), int(reg["w"]), int(reg["h"])),
            float(f["age"]),
            {k: float(v) for k, v in f["gender"].items()},
            {k: float(v) for k, v in f["race"].items()},
            self.name, len(good))

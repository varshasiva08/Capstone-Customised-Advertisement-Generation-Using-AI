from __future__ import annotations


def get_scorer(name: str, cfg: dict = None):
    """Factory: 'deepface' (default verifier), 'clip' (optional independent evaluator)."""
    sc = (cfg or {}).get("scoring", {})
    name = (name or "deepface").lower()
    if name == "deepface":
        from .deepface_scorer import DeepFaceScorer
        return DeepFaceScorer(sc.get("detector", "retinaface"),
                              sc.get("min_face_confidence", 0.80))
    if name == "clip":
        from .clip_scorer import ClipZeroShotScorer
        return ClipZeroShotScorer()
    raise ValueError(f"Unknown scorer '{name}'. Available: deepface, clip. "
                     "('human' ratings are imported with scoring.human, not run per image.)")

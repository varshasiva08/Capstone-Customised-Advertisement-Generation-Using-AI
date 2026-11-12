"""Non-LLM image scoring: face-attribute readings + profile-fidelity evaluation."""
from .types import AxisResult, FaceReading, FidelityReport  # noqa: F401
from .fidelity import evaluate_profile, selection_score  # noqa: F401

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class FaceReading:
    """Raw attributes read from one image by a scorer (no target involved)."""
    face_found: bool
    face_confidence: float = 0.0
    box: Optional[tuple] = None                 # (x, y, w, h) in the source image
    age: Optional[float] = None                 # years (estimator output)
    gender: dict = field(default_factory=dict)  # {"Woman": 0-100, "Man": 0-100}
    race: dict = field(default_factory=dict)    # deepface classes, 0-100 each
    scorer: str = ""
    n_faces: int = 0

    def to_dict(self):
        d = asdict(self)
        d["box"] = list(self.box) if self.box else None
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if d.get("box"):
            d["box"] = tuple(d["box"])
        return cls(**d)


@dataclass
class AxisResult:
    axis: str                       # "ethnicity" | "gender" | "age" | "body_type"
    scored: bool                    # False => not automatically measurable
    passed: Optional[bool] = None   # verifier decision (None if unscored)
    strict: Optional[bool] = None   # stricter criterion (age: exact decade)
    error: float = 0.0              # normalised 0..1 distance from target
    direction: int = 0              # age only: -1 too young, +1 too old
    target: str = ""
    predicted: str = ""
    confidence: float = 0.0         # P(target) (ethnicity/gender) or years-off (age)
    note: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class FidelityReport:
    face_found: bool
    axes: dict                      # axis -> AxisResult
    reading: FaceReading

    @property
    def scored_axes(self):
        return [a for a in self.axes.values() if a.scored]

    @property
    def all_passed(self) -> bool:
        return self.face_found and all(a.passed for a in self.scored_axes)

    @property
    def fraction_passed(self) -> float:
        sa = self.scored_axes
        return sum(1 for a in sa if a.passed) / len(sa) if sa else 0.0

    def failing(self):
        return [a for a in self.scored_axes if not a.passed]

    def to_dict(self):
        return {
            "face_found": self.face_found,
            "all_passed": self.all_passed,
            "fraction_passed": self.fraction_passed,
            "axes": {k: v.to_dict() for k, v in self.axes.items()},
            "reading": self.reading.to_dict(),
        }

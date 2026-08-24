"""
BiasTracker — Demographic Distribution Monitor

Tracks the demographic distribution of generated ad images across a session
or evaluation batch. Flags underrepresentation when any demographic group
falls below a configurable threshold.

Paper contribution:
    Unlike prior ad generation work that optimises for engagement metrics,
    AdFidelity actively monitors demographic coverage across generated outputs.
    The BiasTracker logs each generation's profile and surfaces distribution
    imbalances — ensuring no ethnicity, body type, or age group is
    systematically under-generated relative to others.
"""

import json
import os
from collections import defaultdict
from datetime import datetime


class BiasTracker:
    """
    Tracks demographic distribution across generated ad images.

    Usage:
        tracker = BiasTracker()
        tracker.log(profile, image_path, fidelity_scores, cdvr_iterations)
        report = tracker.report()
        tracker.save("eval/bias_report.json")
    """

    def __init__(self, log_path: str = "eval/bias_log.json",
                 underrep_threshold: float = 0.15):
        """
        Args:
            log_path:             Where to persist the log.
            underrep_threshold:   Flag a group as underrepresented if its
                                  share of total outputs falls below this
                                  fraction (default 15%).
        """
        self.log_path  = log_path
        self.threshold = underrep_threshold
        self.entries   = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def log(self, profile: dict, image_path: str,
            fidelity_scores: dict = None, cdvr_iterations: int = 0) -> None:
        """
        Record one generation run.

        Args:
            profile:           Demographic profile dict (ethnicity, body_type, age).
            image_path:        Path to the generated image.
            fidelity_scores:   DFC output dict (age_score, ethnicity_score, etc.).
            cdvr_iterations:   How many CDVR correction iterations were needed.
        """
        entry = {
            "timestamp":          datetime.now().isoformat(),
            "profile":            profile,
            "image_path":         image_path,
            "fidelity":           fidelity_scores or {},
            "cdvr_iterations":    cdvr_iterations,
        }
        self.entries.append(entry)
        self._persist()

    # ------------------------------------------------------------------ #
    # Distribution analysis
    # ------------------------------------------------------------------ #

    def distribution(self) -> dict:
        """
        Return counts and percentages for each demographic dimension.
        """
        counts = {
            "ethnicity":  defaultdict(int),
            "body_type":  defaultdict(int),
            "age":        defaultdict(int),
        }
        total = len(self.entries)
        if total == 0:
            return {}

        for e in self.entries:
            p = e.get("profile", {})
            for dim in counts:
                val = p.get(dim, "unknown")
                counts[dim][val] += 1

        result = {}
        for dim, cnt in counts.items():
            result[dim] = {
                val: {"count": n, "pct": round(n / total * 100, 1)}
                for val, n in sorted(cnt.items())
            }
        return result

    def underrepresented(self) -> dict:
        """
        Return groups whose share is below self.threshold.
        """
        dist  = self.distribution()
        total = len(self.entries)
        if total == 0:
            return {}

        flagged = {}
        for dim, groups in dist.items():
            for val, stats in groups.items():
                if stats["pct"] / 100 < self.threshold:
                    if dim not in flagged:
                        flagged[dim] = []
                    flagged[dim].append({
                        "group": val,
                        "count": stats["count"],
                        "pct":   stats["pct"],
                    })
        return flagged

    def avg_fidelity(self) -> dict:
        """
        Return mean fidelity scores across all logged runs.
        """
        keys   = ["age_score", "gender_score", "ethnicity_score", "overall"]
        totals = defaultdict(float)
        count  = 0

        for e in self.entries:
            f = e.get("fidelity", {})
            if f:
                for k in keys:
                    totals[k] += f.get(k, 0)
                count += 1

        if count == 0:
            return {}
        return {k: round(v / count, 2) for k, v in totals.items()}

    def report(self) -> dict:
        """
        Full summary report — distribution, underrepresented groups,
        average fidelity, average CDVR iterations.
        """
        total = len(self.entries)
        avg_cdvr = (
            round(sum(e["cdvr_iterations"] for e in self.entries) / total, 2)
            if total else 0
        )
        return {
            "total_generated":      total,
            "distribution":         self.distribution(),
            "underrepresented":     self.underrepresented(),
            "avg_fidelity":         self.avg_fidelity(),
            "avg_cdvr_iterations":  avg_cdvr,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _persist(self) -> None:
        with open(self.log_path, "w") as f:
            json.dump(self.entries, f, indent=2)

    def save(self, report_path: str = None) -> None:
        """Save the full report (not just the raw log) to a JSON file."""
        path = report_path or self.log_path.replace("log", "report")
        with open(path, "w") as f:
            json.dump(self.report(), f, indent=2)
        print(f"[BiasTracker] Report saved → {path}")

    def load(self) -> None:
        """Load a previously saved log from disk."""
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                self.entries = json.load(f)
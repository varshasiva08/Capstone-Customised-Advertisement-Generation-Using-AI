"""Shared helpers for eval scripts.

Centralizes functionality duplicated across run_full_evaluation.py and
run_cpdc_vs_cdvr.py so the two stay in sync.
"""

from itertools import product as itertools_product

ETHNICITIES = ["South Asian", "East Asian", "African American", "White", "Latino Hispanic"]
BODY_TYPES  = ["slim", "medium", "plus-size"]
AGES        = ["20s", "30s", "40s", "50s"]


def build_profile_grid():
    """Returns list of 36 demographic profile dicts (3x3x4 grid)."""
    grid = []
    for eth, bt, age in itertools_product(ETHNICITIES, BODY_TYPES, AGES):
        grid.append({
            "ethnicity": eth,
            "body_type": bt,
            "age": age,
            "gender": "Female",
        })
    return grid

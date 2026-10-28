"""Shared helper to collect HF tokens from the environment.

Centralizes the HF_TOKEN_1/2/3 env-var reading that was previously
duplicated across motion.py, product_describe.py, flux_pipeline.py and
visual_checker.py.
"""

import os


def collect_hf_tokens():
    """Collect HF_TOKEN_1/2/3 (plus bare HF_TOKEN) from environment,
    deduplicated, in order."""
    tokens = []
    for key in ("HF_TOKEN_1", "HF_TOKEN_2", "HF_TOKEN_3", "HF_TOKEN"):
        value = os.getenv(key)
        if value and value not in tokens:
            tokens.append(value)
    return tokens


def primary_hf_token():
    """Return the first HF token, or None if none are configured."""
    tokens = collect_hf_tokens()
    return tokens[0] if tokens else None

"""Glue between the loop (profile -> prompt) and profiler.prompt_builder."""
from __future__ import annotations

from profiler.prompt_builder import build_prompt
from profiler.regional_prompt_builder import build_regional_prompt


def make_prompt_builder(clothing: str, background: str, cfg: dict, *, scene=None,
                        product_description=None, product_category="other",
                        background_focus="Auto"):
    """Returns build(profile, correction_levels=None, legacy_keys=None, iteration=0) -> str."""
    def build(profile, correction_levels=None, legacy_keys=None, iteration=0):
        kw = dict(correction_levels=correction_levels,
                  correction_keys=list(legacy_keys) if legacy_keys else None,
                  iteration=iteration, product_description=product_description,
                  product_category=product_category, scene=scene)
        if profile.get("region"):
            return build_regional_prompt(profile, clothing, background, cfg,
                                         region=profile["region"], **kw)
        return build_prompt(profile, clothing, background, cfg,
                            background_focus=background_focus, **kw)
    return build

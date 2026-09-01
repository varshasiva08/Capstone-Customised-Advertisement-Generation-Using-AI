"""
profiler/regional_prompt_builder.py — Structured Regional Prompt Synthesis

Extends the existing ethnicity-level prompt (config['options']['ethnicity'])
with a REGION sub-axis, e.g. "South Asian" -> "South Indian" / "North Indian"
/ "Bengali" / "Punjabi". Each region carries its own structured descriptor
block (skin tone range, facial feature cues, hair, regional attire cue)
instead of a single flat ethnicity string.

This is additive: build_prompt() in prompt_builder.py is untouched and still
works exactly as before for callers that don't pass a region. This module
wraps it.

Where to put this file:
    profiler/regional_prompt_builder.py

Add this block to config.yaml (any indentation level, top-level key):

    regions:
      South Asian:
        South Indian:
          descriptor: "South Indian features, deep brown skin tone, dark eyes, straight black hair"
          attire_cue: "traditional South Indian jewelry, temple-style gold earrings"
        North Indian:
          descriptor: "North Indian features, warm wheatish skin tone, dark eyes, dark hair"
          attire_cue: "Punjabi-style gold jewelry, jhumka earrings"
        Bengali:
          descriptor: "Bengali features, fair to warm skin tone, dark eyes, dark hair"
          attire_cue: "traditional Bengali gold jewelry, red-bordered saree cues"
      East Asian:
        East Asian:          # no sub-region split yet — passthrough
          descriptor: ""
          attire_cue: ""
      African American:
        African American:    # no sub-region split yet — passthrough
          descriptor: ""
          attire_cue: ""

Only "South Asian" needs real entries for your paper's headline claim.
Leave the other two ethnicities as passthrough (empty descriptor/attire_cue)
unless you want to extend regional splits to them too.
"""

from __future__ import annotations

from profiler.prompt_builder import build_prompt, load_config


def list_regions(ethnicity: str, config: dict = None) -> list[str]:
    """Return the list of defined regions for an ethnicity, or [] if none."""
    if config is None:
        config = load_config()
    return list(config.get("regions", {}).get(ethnicity, {}).keys())


def build_regional_prompt(profile: dict, clothing: str, background: str,
                           config: dict = None,
                           region: str = None,
                           correction_keys: list[str] = None,
                           iteration: int = 0,
                           correction_levels: dict = None,
                           product_description: str = None,
                           product_category: str = "handbag") -> str:
    """
    Same signature as build_prompt(), plus an optional `region` string.

    Args:
        profile: dict with 'ethnicity', 'body_type', 'age' — unchanged.
        region:  Optional sub-region name, must be a key under
                 config['regions'][profile['ethnicity']]. If None, or if
                 no regions are defined for this ethnicity, behaves
                 identically to build_prompt() (no regional injection).

    Returns:
        The complete prompt string, with the region's descriptor and
        attire cue injected after the subject description, before any
        CPDC/legacy correction tokens (so correction tokens still land
        last, closest to what the model "reads most recently").
    """
    if config is None:
        config = load_config()

    base_prompt = build_prompt(
        profile, clothing, background, config,
        correction_keys=correction_keys,
        iteration=iteration,
        correction_levels=correction_levels,
        product_description=product_description,
        product_category=product_category,
    )

    if not region:
        return base_prompt

    region_cfg = config.get("regions", {}).get(profile["ethnicity"], {}).get(region)
    if not region_cfg:
        # Region not defined for this ethnicity — fail loud in dev, not
        # silently in an eval run. Comment out the raise if you'd rather
        # it just no-op.
        raise ValueError(
            f"Region '{region}' not defined for ethnicity "
            f"'{profile['ethnicity']}'. Known regions: "
            f"{list_regions(profile['ethnicity'], config)}"
        )

    descriptor = (region_cfg.get("descriptor") or "").strip().rstrip(",")
    attire_cue = (region_cfg.get("attire_cue") or "").strip().rstrip(",")

    injection_parts = [p for p in [descriptor, attire_cue] if p]
    if not injection_parts:
        return base_prompt

    injection_str = ", ".join(injection_parts)

    # Inject right after "woman in her {age}" — same anchor point
    # prompt_builder.py uses for correction tokens, so region descriptors
    # sit next to the subject, and correction tokens (added by build_prompt
    # above) sit after them.
    anchor = f"woman in her {profile['age']}"
    if anchor in base_prompt:
        # Insert the region block immediately after the anchor, before
        # whatever build_prompt already inserted there (correction tokens).
        # We do this by re-splitting on the anchor once.
        idx = base_prompt.index(anchor) + len(anchor)
        return base_prompt[:idx] + f", {injection_str}" + base_prompt[idx:]
    else:
        return base_prompt.rstrip(", ") + f", {injection_str}"


# ---------- CLI test ----------
if __name__ == "__main__":
    profile = {"ethnicity": "South Asian", "body_type": "medium", "age": "30s"}

    print("=== No region (unchanged behaviour) ===")
    print(build_regional_prompt(profile, "White Blazer Suit", "Pure White"))

    print("\n=== South Indian region ===")
    print(build_regional_prompt(profile, "White Blazer Suit", "Pure White",
                                 region="South Indian"))

    print("\n=== South Indian region + CPDC correction on AF ===")
    print(build_regional_prompt(profile, "White Blazer Suit", "Pure White",
                                 region="South Indian",
                                 correction_levels={"AF": 2}))

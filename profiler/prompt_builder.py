"""
Builds the advertisement image prompt from a demographic profile,
clothing, and background choices.

The prompt is always framed as a professional fashion advertisement
photograph - never as a portrait or generic image. This is intentional:
  - It ensures the generated image has the compositional quality of a
    real ad (studio lighting, full body, clean background, product space).
  - It frames the model as a commercial subject, not a person, which
    reduces demographic bias in generation models that apply different
    aesthetics to different groups when not constrained.

Also handles CDVR prompt correction: when the DFC check fails on one or
more axes, correction tokens from config.yaml are injected into the prompt
before the next generation attempt.

CPDC (Confidence-Proportional Demographic Correction) replaces the old
fixed mild/strong severity ladder for STF and AF with a graduated,
error-proportional level (1-4), selected by modules/cpdc.py based on how
far off the previous attempt's scorer confidence was. BTF keeps the
original mild/strong path since no trained confidence classifier exists
for body type.
"""

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_prompt(profile: dict, clothing: str, background: str,
                 config: dict = None,
                 correction_keys: list[str] = None,
                 iteration: int = 0,
                 correction_levels: dict = None,
                 product_description: str = None,
                 product_category: str = "handbag") -> str:
    """
    Build the full advertisement image prompt.

    Args:
        profile:         dict with 'ethnicity', 'body_type', 'age'.
        clothing:        Selected clothing style (from config options).
        background:      Selected background colour (from config options).
        config:          Loaded config.yaml dict. Loaded from disk if None.
        correction_keys: List of LEGACY correction keys to apply, e.g.
                         ['BTF']. Used only for the binary mild/strong path
                         (body type). Empty or None means no correction.
        iteration:       Which CDVR iteration this is (0 = first attempt).
                         Controls legacy severity: 0->none, 1->mild, 2+->strong.
        correction_levels: dict of CPDC graduated correction levels, e.g.
                         {"STF": 3, "AF": 2}. Keys are axis names (STF,
                         AF), values are integer levels 1-4 looked up as
                         config['corrections'][axis]['level_N']. This is
                         the CPDC path and takes priority over the legacy
                         iteration-based severity for whichever axes it
                         covers.
        product_description: Optional short text description of a product
                         (e.g. "black leather structured handbag with gold
                         buckle"). If provided, the model is instructed to
                         generate the subject naturally wearing/carrying it,
                         so the product is drawn integrated into the
                         pose/lighting rather than composited afterward.
        product_category: How the product is used - "handbag" (carried on
                         shoulder), "sunglasses" (worn on face), "jewelry"
                         (worn - necklace/earrings), or "other" (generic
                         hand-held). Picks the matching phrasing template.

    Returns:
        The complete prompt string. Never shown in the UI.
    """
    if config is None:
        config = load_config()

    body_type_extras = config["body_type_extras"].get(profile["body_type"], "")

    base_prompt = config["prompt_template"].format(
        body_type=profile["body_type"],
        ethnicity=profile["ethnicity"],
        age=profile["age"],
        body_type_extras=body_type_extras,
        clothing=clothing,
        background=background,
    ).strip()

    if product_description:
        clause_templates = {
            "handbag": (
                ", body and shoulders facing the camera, one hand on hip, a "
                "medium-sized {desc} sized proportionally to her body, roughly "
                "torso-height, not oversized, hanging from her other shoulder "
                "and resting at her side, carried naturally like a real "
                "handbag, still facing forward toward camera, face clearly "
                "visible, product clearly visible and in focus, realistic "
                "contact shadow where the bag meets her arm"
            ),
            "sunglasses": (
                ", body facing the camera, head turned slightly toward camera, "
                "wearing {desc} on her face, sunglasses fitted naturally and "
                "correctly sized to her face, resting on the bridge of her "
                "nose, temples over her ears, clearly visible, sharp focus on "
                "the eyewear, realistic reflections and shadow on the lenses"
            ),
            "jewelry": (
                ", body and shoulders facing the camera, face clearly visible, "
                "wearing {desc}, fitted naturally, clearly visible, sharp "
                "focus, realistic scale relative to her features"
            ),
            "other": (
                ", body and shoulders facing the camera, one hand on hip, "
                "other hand holding a medium-sized {desc} sized proportionally "
                "to her body, not oversized, held naturally at waist level, "
                "still facing forward toward camera, face clearly visible, "
                "product clearly visible and in focus, realistic grip and "
                "shadow"
            ),
        }
        template = clause_templates.get(product_category, clause_templates["handbag"])
        product_clause = template.format(desc=product_description)
        # Insert right after the pose/clothing description, before the
        # lighting/background tail, so it reads as part of the main subject.
        insert_after = "visible feet, empty space reserved for product placement and logo"
        if insert_after in base_prompt:
            base_prompt = base_prompt.replace(insert_after, "visible feet" + product_clause)
        else:
            base_prompt = base_prompt.rstrip(", ") + product_clause

    corrections_cfg = config.get("corrections", {})
    correction_tokens = []

    # CPDC graduated-level path - STF (ethnicity) and AF (age)
    if correction_levels:
        for axis, level in correction_levels.items():
            if level <= 0:
                continue
            axis_cfg = corrections_cfg.get(axis, {})
            token = axis_cfg.get(f"level_{level}", "")
            if token:
                correction_tokens.append(token.strip().rstrip(","))

    # Legacy binary path - BTF only (no trained confidence classifier
    # exists for body type, so it stays on the original mild/strong ladder)
    if correction_keys and iteration > 0:
        severity = "mild" if iteration == 1 else "strong"
        for key in correction_keys:
            axis_cfg = corrections_cfg.get(key, {})
            if "mild" in axis_cfg:
                token = axis_cfg.get(severity, "")
                if token:
                    correction_tokens.append(token.strip().rstrip(","))

    if not correction_tokens:
        return base_prompt

    correction_str = ", ".join(correction_tokens)

    # Inject correction tokens right after the subject description
    # (after "woman in her {age}") so they modify the subject, not the scene
    inject_after = f"woman in her {profile['age']}"
    if inject_after in base_prompt:
        return base_prompt.replace(
            inject_after,
            f"woman in her {profile['age']}, {correction_str}",
            1  # replace only first occurrence
        )
    else:
        # Fallback: prepend correction tokens
        return correction_str + ", " + base_prompt


# ---------- CLI test ----------
if __name__ == "__main__":
    profile = {
        "ethnicity": "African American",
        "body_type": "plus-size",
        "age": "40s"
    }

    print("=== Iteration 0 (first attempt, no corrections) ===")
    print(build_prompt(profile, "White Blazer Suit", "Pure White"))

    print("\n=== Legacy path: iteration 1, mild correction on BTF ===")
    print(build_prompt(profile, "White Blazer Suit", "Pure White",
                       correction_keys=["BTF"], iteration=1))

    print("\n=== CPDC path: graduated level 3 on AF, level 2 on STF ===")
    print(build_prompt(profile, "White Blazer Suit", "Pure White",
                       correction_levels={"AF": 3, "STF": 2}))

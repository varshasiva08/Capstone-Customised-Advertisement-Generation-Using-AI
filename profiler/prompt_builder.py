"""
Builds the advertisement image prompt from a demographic profile,
clothing, and background choices.

The prompt is always framed as a professional fashion advertisement
photograph — never as a portrait or generic image. This is intentional:
  - It ensures the generated image has the compositional quality of a
    real ad (studio lighting, full body, clean background, product space).
  - It frames the model as a commercial subject, not a person, which
    reduces demographic bias in generation models that apply different
    aesthetics to different groups when not constrained.

Also handles CDVR prompt correction: when the DFC check fails on one or
more axes, correction tokens from config.yaml are injected into the prompt
before the next generation attempt.

Correction token severity escalates across iterations:
    Iteration 1: mild correction tokens
    Iteration 2: strong correction tokens
    Iteration 3: strong + explicit negative framing (last attempt)

Scene backgrounds / background effects:
    `scene` (optional) replaces the studio scene clause — "studio
    photography, seamless {background} background, softbox lighting" — with
    a free-form scene description, e.g. "realistic luxury hotel lobby with
    cinematic lighting". Each entry in config['background_presets'] carries
    its own lighting, so nothing studio-related contradicts it.
    scene=None keeps the original wording byte-for-byte.
"""

import yaml


# Fallback if config.yaml has no `studio_scene` key (older configs).
DEFAULT_STUDIO_SCENE = "studio photography, seamless {background} background, softbox lighting"

# Custom scene text is user-supplied: cap it so a pasted paragraph cannot
# swamp the prompt, and flatten newlines so the prompt stays one line.
MAX_SCENE_CHARS = 200


def clean_scene(scene) -> str:
    """Normalise user-supplied scene text. Returns "" if nothing usable.

    Collapses all whitespace/newlines, strips stray punctuation at the
    edges, and caps at MAX_SCENE_CHARS. Idempotent, so it is safe to call
    in app.py before threading the value to other modules.
    """
    if not scene:
        return ""
    text = " ".join(str(scene).split())
    text = text.strip(" ,.;:-")
    if len(text) > MAX_SCENE_CHARS:
        text = text[:MAX_SCENE_CHARS].rstrip(" ,.;:-")
    return text


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_prompt(profile: dict, clothing: str, background: str,
                 config: dict = None,
                 correction_keys: list[str] = None,
                 iteration: int = 0,
                 product_description: str = None,
                 product_category: str = "handbag",
                 correction_levels: dict = None,
                 scene: str = None) -> str:
    """
    Build the full advertisement image prompt.

    Args:
        profile:         dict with 'ethnicity', 'body_type', 'age'.
        clothing:        Selected clothing style (from config options).
        background:      Selected background colour (from config options).
                         Ignored when `scene` is provided.
        config:          Loaded config.yaml dict. Loaded from disk if None.
        correction_keys: List of correction keys to apply, e.g. ['BTF', 'STF'].
                         Empty or None means no correction (first generation).
        iteration:       Which CDVR iteration this is (0 = first attempt).
                         Controls correction severity: 0→none, 1→mild, 2→strong.
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
        correction_levels: Graduated CPDC levels as {axis: level}, e.g.
                         {'STF': 3}. Reads `level_N` tokens from
                         config['corrections'][axis] (falling back to
                         mild/strong). Takes precedence over the
                         iteration-based mild/strong pick when provided;
                         `correction_keys` can still be combined with it
                         for the binary axes (e.g. BTF).
        scene:           Optional free-form scene / background-effect text
                         (e.g. "realistic luxury hotel lobby with cinematic
                         lighting", or a key's value from
                         config['background_presets']). Replaces the studio
                         scene clause. None/empty keeps the original
                         studio wording exactly.

    Returns:
        The complete prompt string. Never shown in the UI.
    """
    if config is None:
        config = load_config()

    body_type_extras = config["body_type_extras"].get(profile["body_type"], "")

    # --- Scene clause ---------------------------------------------------
    scene = clean_scene(scene)
    studio_text = (config.get("studio_scene") or DEFAULT_STUDIO_SCENE) \
        .format(background=background)
    template = config["prompt_template"]
    has_scene_placeholder = "{scene_clause}" in template

    # User scene text is inserted as a format *value*, never run through
    # .format() itself, so braces in custom input can't break rendering.
    scene_clause = scene if (scene and has_scene_placeholder) else studio_text

    base_prompt = template.format(
        body_type=profile["body_type"],
        ethnicity=profile["ethnicity"],
        age=profile["age"],
        body_type_extras=body_type_extras,
        clothing=clothing,
        background=background,
        scene_clause=scene_clause,
    ).strip()

    if scene and not has_scene_placeholder:
        # Legacy template without {scene_clause}: swap the studio text out
        # after formatting so the scene is never silently dropped.
        base_prompt = base_prompt.replace(studio_text, scene, 1)

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

    # No corrections needed on first attempt or if all axes passed.
    # `correction_levels` (graduated CPDC) can be supplied without
    # `correction_keys`; the legacy binary path still requires iteration > 0.
    if not correction_keys and not correction_levels:
        return base_prompt
    if iteration == 0 and not correction_levels:
        return base_prompt

    correction_tokens = []
    corrections_cfg = config.get("corrections", {})

    # Graduated CPDC levels take precedence when provided (axis -> level).
    # Reads level_N tokens from config, falling back to mild/strong so an
    # axis without level_N entries (e.g. BTF) still corrects sensibly.
    for key, level in (correction_levels or {}).items():
        axis_cfg = corrections_cfg.get(key, {})
        token = axis_cfg.get(f"level_{level}") \
            or axis_cfg.get("mild" if level <= 1 else "strong", "")
        if token:
            correction_tokens.append(token.strip().rstrip(","))

    # Legacy binary path: severity picked from the iteration number.
    if not correction_levels:
        severity = "mild" if iteration == 1 else "strong"
        for key in correction_keys or []:
            if key in corrections_cfg:
                token = corrections_cfg[key].get(severity, "")
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

    print("\n=== Iteration 1 (mild correction on BTF + AF) ===")
    print(build_prompt(profile, "White Blazer Suit", "Pure White",
                       correction_keys=["BTF", "AF"], iteration=1))

    print("\n=== Iteration 2 (strong correction on BTF + STF) ===")
    print(build_prompt(profile, "White Blazer Suit", "Pure White",
                       correction_keys=["BTF", "STF"], iteration=2))

    print("\n=== Graduated CPDC correction_levels (STF=3, AF=1) ===")
    print(build_prompt(profile, "White Blazer Suit", "Pure White",
                       correction_levels={"STF": 3, "AF": 1}, iteration=1))

    print("\n=== Scene background (free-form) ===")
    print(build_prompt(profile, "White Blazer Suit", "Pure White",
                       scene="realistic luxury hotel lobby with cinematic lighting"))

    print("\n=== Scene background (config preset) ===")
    cfg = load_config()
    preset = cfg["background_presets"]["Neon City Bokeh"]
    print(build_prompt(profile, "White Blazer Suit", "Pure White", cfg,
                       scene=preset))

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

# Custom and Storyboard scene text is user-supplied: cap it so a pasted
# paragraph cannot swamp the prompt, while preserving a useful description.
MAX_SCENE_CHARS = 400


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
                 scene: str = None,
                 background_focus: str = "Auto") -> str:
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
        background_focus: Depth-of-field preference applied to every scene
                          type: "Auto", "Blurred", or "Sharp".

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
    focus_clauses = {
        "auto": "natural depth of field with the subject and background balanced",
        "blurred": "softly blurred background, shallow depth of field, subject in sharp focus",
        "sharp": "sharp detailed background, deep depth of field, subject and environment in focus",
    }
    focus_clause = focus_clauses.get(str(background_focus).strip().lower())
    if focus_clause is None:
        raise ValueError(
            f"Unknown background_focus '{background_focus}' "
            "(expected 'Auto', 'Blurred', or 'Sharp')"
        )
    scene_clause = f"{scene_clause}, {focus_clause}"

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
        base_prompt = base_prompt.replace(studio_text, scene_clause, 1)

    if product_description:
        # Phrasing per product category, written to give FLUX the physical
        # cues it otherwise misses: explicit scale anchors, visible grip,
        # contact shadows and partial occlusion by the body/hands.
        clause_templates = {
            "handbag": (
                ", body and shoulders facing the camera, one arm relaxed at "
                "her side with the elbow slightly bent and her hand holding a "
                "{desc} at hip level, the bag is sized proportionally to her "
                "body — about the width of her torso, not oversized and not "
                "tiny — her fingers visibly wrap around the handle, if it has "
                "a strap the strap rests on her shoulder with realistic "
                "tension and a slight indentation in the fabric, the lower "
                "edge of the bag reaches her hip, her forearm partly occludes "
                "the bag, still facing forward toward the camera, face clearly "
                "visible, the product clearly visible and in sharp focus, "
                "realistic contact shadow where the bag meets her body"
            ),
            "sunglasses": (
                ", body facing the camera, head turned slightly toward the "
                "camera, wearing {desc} on her face, the sunglasses are "
                "correctly sized to her face and rest on the bridge of her "
                "nose with the temples passing over and behind her ears, the "
                "lenses align with her eyes, slight occlusion of her eyebrows "
                "and cheeks, clearly visible, sharp focus on the eyewear, "
                "realistic reflections in the lenses and a soft shadow from "
                "the frame on her face"
            ),
            "jewelry": (
                ", body and shoulders facing the camera, face clearly "
                "visible, wearing {desc} on her bare skin, the piece sits "
                "naturally with the correct scale relative to her features — "
                "a necklace resting on her collarbone or just below the "
                "neckline, earrings sitting on her earlobes, a bracelet "
                "wrapping around her wrist or a ring around her finger — the "
                "lobe, wrist or finger partly occludes it, the metal catches "
                "the light realistically, clearly visible, sharp focus, "
                "realistic contact shadow where it meets the skin"
            ),
            "watch": (
                ", body and shoulders facing the camera, one wrist angled "
                "toward the camera, wearing {desc} on her wrist, the band "
                "wraps snugly around the wrist with the clasp hidden, the "
                "watch face is about the width of her wrist (not oversized), "
                "her other hand may rest near it, clearly visible, sharp "
                "focus, realistic contact shadow under the watch"
            ),
            "clothing": (
                ", full body facing the camera, wearing {desc} that fits her "
                "body correctly and drapes naturally over her shoulders, "
                "waist and hips with realistic folds and wrinkles, the "
                "garment follows her posture, fabric texture, seams and "
                "hemline visible, the outfit is the right length for her "
                "body, nothing oversized or ill-fitting, still facing forward "
                "toward the camera, face clearly visible, sharp focus on the "
                "clothing"
            ),
            "footwear": (
                ", full body visible from head to toe, wearing {desc} on her "
                "feet, the shoes fit correctly and are the right proportion "
                "to her legs, both feet planted naturally on the ground, the "
                "shoes show realistic contact with the floor and a soft cast "
                "shadow, visible feet, sharp focus on the footwear"
            ),
            "headwear": (
                ", body facing the camera, wearing {desc} on her head, the "
                "piece sits naturally with its edge following the curve of "
                "her head, correctly sized and not oversized, her hair sits "
                "naturally around it, a soft shadow falls from it onto her "
                "face, clearly visible, sharp focus"
            ),
            "other": (
                ", body and shoulders facing the camera, holding a {desc} "
                "sized proportionally to her body in one hand at about waist "
                "or chest level, her fingers curl naturally around it, fully "
                "visible and not occluded by her body, still facing forward "
                "toward the camera, face clearly visible, the product clearly "
                "visible and in sharp focus, realistic grip and contact shadow"
            ),
        }
        # Neutral fallback (was "handbag", which forced a bag onto every
        # unrecognised product).
        template = clause_templates.get(product_category, clause_templates["other"])
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

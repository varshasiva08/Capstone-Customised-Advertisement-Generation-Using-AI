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
"""

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_prompt(profile: dict, clothing: str, background: str,
                 config: dict = None,
                 correction_keys: list[str] = None,
                 iteration: int = 0) -> str:
    """
    Build the full advertisement image prompt.

    Args:
        profile:         dict with 'ethnicity', 'body_type', 'age'.
        clothing:        Selected clothing style (from config options).
        background:      Selected background colour (from config options).
        config:          Loaded config.yaml dict. Loaded from disk if None.
        correction_keys: List of correction keys to apply, e.g. ['BTF', 'STF'].
                         Empty or None means no correction (first generation).
        iteration:       Which CDVR iteration this is (0 = first attempt).
                         Controls correction severity: 0→none, 1→mild, 2→strong.

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

    # No corrections needed on first attempt or if all axes passed
    if not correction_keys or iteration == 0:
        return base_prompt

    # Determine severity based on iteration number
    severity = "mild" if iteration == 1 else "strong"

    correction_tokens = []
    corrections_cfg = config.get("corrections", {})

    for key in correction_keys:
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

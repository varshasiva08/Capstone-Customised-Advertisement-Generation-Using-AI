"""
tests/test_product_integration.py

Automated tests for the Product Integration module.

Covers what CAN be automated:
  - prompt_builder.build_prompt() produces correct output per product category
  - product_describe._clean_description() correctly parses messy model output
  - category coverage check (catches the handbag/sunglasses/jewelry/other vs.
    the app's clothing/other keyword categories mismatch)
  - (optional, network-dependent) a live call to describe_product()

What is NOT automated here (see tests/manual_test_log.md instead):
  - Visual realism (pose, sizing, shadows) — needs a human to look at the image
  - End-to-end UI flow through Streamlit

Run with:
    python3 tests/test_product_integration.py

Exits with code 0 if all non-skipped tests pass, 1 otherwise.
"""

import os
import sys

# Allow running this file directly (python3 tests/test_product_integration.py)
# from the project root, without needing to install the project as a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from profiler.prompt_builder import build_prompt


# ---------------------------------------------------------------------------
# Test runner scaffolding (kept dependency-free — no pytest required)
# ---------------------------------------------------------------------------

_results = []  # list of (test_name, status, message)


def _record(name, status, message=""):
    _results.append((name, status, message))
    icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️ "}[status]
    print(f"{icon} {name}" + (f" — {message}" if message else ""))


def _load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
    )
    with open(config_path) as f:
        return yaml.safe_load(f)


TEST_PROFILE = {"ethnicity": "South Asian", "body_type": "plus-size", "age": "40s"}


# ---------------------------------------------------------------------------
# F5 / C2-C5 — build_prompt() per category
# ---------------------------------------------------------------------------

def test_build_prompt_no_product():
    """F1/F5 — prompt builds fine with no product at all (baseline)."""
    try:
        cfg = _load_config()
        prompt = build_prompt(TEST_PROFILE, "White Blazer Suit", "Warm Beige", cfg)
        assert isinstance(prompt, str) and len(prompt) > 0
        assert "handbag" not in prompt.lower()  # no product clause should be injected
        _record("test_build_prompt_no_product", "PASS")
    except Exception as e:
        _record("test_build_prompt_no_product", "FAIL", str(e))


def test_build_prompt_each_known_category():
    """C2-C5 — each category the prompt builder actually knows about."""
    cfg = _load_config()
    categories = {
        "handbag": "cream faux leather tote bag with chain strap",
        "sunglasses": "black oversized cat-eye sunglasses with gold frame",
        "jewelry": "gold hoop earrings",
        "other": "silver wristwatch with leather strap",
    }
    for category, desc in categories.items():
        try:
            prompt = build_prompt(
                TEST_PROFILE, "White Blazer Suit", "Warm Beige", cfg,
                product_description=desc, product_category=category,
            )
            assert desc in prompt, "product description missing from prompt"
            assert "face clearly visible" in prompt, "front-facing instruction missing"
            _record(f"test_build_prompt_category[{category}]", "PASS")
        except Exception as e:
            _record(f"test_build_prompt_category[{category}]", "FAIL", str(e))


def test_category_coverage_matches_app_keywords():
    """
    C1 — regression test for the known gap: app.py's PRODUCT_KEYWORDS defines
    5 categories (handbag, sunglasses, jewelry, clothing, other) but
    prompt_builder.py's clause_templates only defines 4. This test will FAIL
    until that gap is fixed, which is the point - it documents and catches it.
    """
    # Mirrors app.py's PRODUCT_KEYWORDS keys. Ideally this list would be
    # imported directly from a shared module rather than duplicated here -
    # see the note in the testing plan about extracting PRODUCT_KEYWORDS.
    app_categories = {"handbag", "sunglasses", "jewelry", "clothing", "other"}

    import inspect
    import profiler.prompt_builder as pb_module
    source = inspect.getsource(pb_module.build_prompt)

    # Extract the categories actually handled in clause_templates by checking
    # which of the known category names appear as dict keys in the source.
    handled = {cat for cat in app_categories if f'"{cat}":' in source}

    missing = app_categories - handled
    if missing:
        _record(
            "test_category_coverage_matches_app_keywords", "FAIL",
            f"prompt_builder.py has no template for: {sorted(missing)} "
            f"(falls back to 'handbag' phrasing for these)"
        )
    else:
        _record("test_category_coverage_matches_app_keywords", "PASS")


# ---------------------------------------------------------------------------
# F4 — description cleaning logic
# ---------------------------------------------------------------------------

def test_clean_description_parsing():
    """F4 — messy vision-model output gets reduced to a clean phrase."""
    from modules.product_describe import _clean_description

    cases = [
        (
            "Here's a description of the product in one short phrase:\n"
            "**Cream faux leather tote bag with chain strap.**",
            "Cream faux leather tote bag with chain strap",
        ),
        ("black leather structured handbag with gold buckle",
         "black leather structured handbag with gold buckle"),
        ('"a nice bag."', "a nice bag"),
    ]
    all_ok = True
    for raw, expected in cases:
        got = _clean_description(raw)
        if got != expected:
            _record("test_clean_description_parsing", "FAIL",
                    f"expected {expected!r}, got {got!r}")
            all_ok = False
            break
    if all_ok:
        _record("test_clean_description_parsing", "PASS")


# ---------------------------------------------------------------------------
# D1 (optional/live) — real call to the vision model
# Skipped automatically if no HF token or --live flag not passed, since this
# costs an API call and needs network + a valid .env.
# ---------------------------------------------------------------------------

def test_describe_product_live(sample_image_path):
    """D1 — live end-to-end call to Gemma-3 via HF. Requires network + token."""
    if not os.getenv("HF_TOKEN_1"):
        _record("test_describe_product_live", "SKIP", "no HF_TOKEN_1 set")
        return
    if not sample_image_path or not os.path.exists(sample_image_path):
        _record("test_describe_product_live", "SKIP", "no sample product image provided")
        return
    try:
        from modules.product_describe import describe_product
        from PIL import Image
        img = Image.open(sample_image_path)
        desc = describe_product(img)
        assert isinstance(desc, str) and len(desc) > 0
        assert desc != "stylish accessory", "got the silent-failure fallback value"
        _record("test_describe_product_live", "PASS", f"got: {desc!r}")
    except Exception as e:
        _record("test_describe_product_live", "FAIL", str(e))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    from dotenv import load_dotenv
    load_dotenv()

    sample_image_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("Running Product Integration test suite...\n")

    test_build_prompt_no_product()
    test_build_prompt_each_known_category()
    test_category_coverage_matches_app_keywords()
    test_clean_description_parsing()
    test_describe_product_live(sample_image_path)

    passed = sum(1 for _, s, _ in _results if s == "PASS")
    failed = sum(1 for _, s, _ in _results if s == "FAIL")
    skipped = sum(1 for _, s, _ in _results if s == "SKIP")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped "
          f"(of {len(_results)} total)")
    print(f"{'='*50}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

"""
Quick provider smoke-test for AdFidelity.

Verifies the free providers end-to-end without opening the Streamlit UI:

    python quick_test.py            # image + product description
    python quick_test.py --video    # also test the free Wan Space (slow queue)

Run this after filling in .env. Image generation needs
CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN; the product-description step
needs Ollama running with `ollama pull llava`.
"""

import argparse
import os
import sys

import yaml
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


def load_cfg() -> dict:
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_env(cfg: dict) -> bool:
    ok = True
    gen_cfg = cfg.get("generation", {}) or {}
    pd_cfg = cfg.get("product_describe", {}) or {}

    if gen_cfg.get("provider", "cloudflare") == "cloudflare":
        for key in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
            if not os.getenv(key):
                print(f"  [MISSING] {key} is empty in .env")
                ok = False
        if ok:
            print("  [OK] Cloudflare credentials present")
    else:
        print(f"  [SKIP] generation.provider = {gen_cfg.get('provider')} (not cloudflare)")

    if pd_cfg.get("provider", "ollama") == "ollama":
        print(f"  [INFO] product description uses local Ollama ({pd_cfg.get('model', 'llava')})")
    return ok


def test_image(cfg: dict) -> Image.Image:
    print("\n[1/3] Image generation (FLUX.1-schnell)...")
    from generator.flux_pipeline import generate_image

    img, device, res, steps = generate_image(
        "a red ceramic teapot on a white table, clean studio product photo", 42, cfg
    )
    os.makedirs("outputs", exist_ok=True)
    out = os.path.join("outputs", "quick_test_image.png")
    img.save(out)
    print(f"  OK -> {device} | {res[0]}x{res[1]} | {steps} steps | saved {out}")
    return img


def test_describe(cfg: dict) -> None:
    print("\n[2/3] Product description (local Ollama)...")
    from modules.product_describe import describe_product

    probe = Image.new("RGB", (256, 256), "white")
    print(f"  OK -> {describe_product(probe, cfg)!r}")


def test_video(cfg: dict, image: Image.Image) -> None:
    print("\n[3/3] Video generation (free HF Space — this can take a while)...")
    from modules.motion import animate_image

    path = animate_image(
        image,
        config=cfg,
        profile={"ethnicity": "South Asian", "body_type": "slim", "age": "30s"},
        clothing="White Blazer Suit",
        motion_style="sway",
    )
    print(f"  OK -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true", help="also test the Wan Space")
    args = parser.parse_args()

    cfg = load_cfg()

    print("Environment:")
    if not check_env(cfg):
        print("\nFix the missing values in .env and re-run.")
        return 1

    try:
        image = test_image(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL: {type(e).__name__}: {e}")
        return 1

    try:
        test_describe(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL: {type(e).__name__}: {e}")

    if args.video:
        try:
            test_video(cfg, image)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL: {type(e).__name__}: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

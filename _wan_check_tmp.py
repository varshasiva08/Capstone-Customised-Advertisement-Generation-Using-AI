"""Throwaway: direct Wan Space spot-check (reuse existing quick_test image)."""
import os
import yaml
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

img = Image.open(os.path.join("outputs", "quick_test_image.png")).convert("RGB")

from modules.motion import animate_image

print("calling animate_image (Wan Space)...")
path = animate_image(
    img,
    config=cfg,
    profile={"ethnicity": "South Asian", "body_type": "slim", "age": "30s"},
    clothing="White Blazer Suit",
    motion_style="sway",
)
print("OK ->", path)
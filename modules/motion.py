"""
Module 5 - Motion / Animation

Takes the final advertisement image and animates it via the
HuggingFace Inference API (Wan2.1-I2V-14B-720P, fal-ai provider).

Pipeline:
  1. Build a demographically-aware motion prompt from the user's
     profile and clothing choice — contextual motion descriptions
     derived from the brand brief rather than a fixed driving video.
  2. Submit the character image + motion prompt to Wan2.1-I2V via
     the HuggingFace Inference API (no local VRAM required).
  3. Save and return the generated video.

Contribution:
  The motion prompt is not hardcoded — it is constructed dynamically
  from the demographic profile and clothing context parsed from the
  user's brand brief. A woman in a saree receives a different motion
  description than one in corporate wear, producing more natural and
  contextually appropriate animation per demographic target.
"""

import os
import time
import tempfile

import yaml
from dotenv import load_dotenv
from PIL import Image
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN_1")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(config: dict = None) -> dict:
    if config is not None:
        return config
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Demographically-aware motion prompt builder
# ---------------------------------------------------------------------------

def build_motion_prompt(profile: dict, clothing: str = "", motion_style: str = "sway") -> str:
    """
    Build a locked-down motion prompt for Wan2.1-I2V.

    Two styles:
      - "sway": gentle body sway and slight turn to show product from
                different angles. Best for jewellery, bags, accessories.
      - "walk": walks forward a few steps then stops in a natural pose.
                Best for clothing, sarees, suits.

    Args:
        profile:      Demographic profile dict
        clothing:     Clothing description
        motion_style: "sway" or "walk"

    Returns:
        Motion prompt string.
    """
    ethnicity  = profile.get("ethnicity", "woman")
    age        = profile.get("age", "adult")
    clothing_l = clothing.lower() if clothing else ""

    # Base identity anchor — keeps Wan2.1 from drifting
    identity = (
        f"{ethnicity} woman in her {age}, "
        f"standing in a studio, seamless background, "
        f"professional advertisement photography, "
        f"sharp focus, full body visible from head to toe, "
        f"front-facing, face clearly visible"
    )

    # Negative intent — lock out the dancing
    no_dance = (
        "no dancing, no jumping, no exaggerated movement, "
        "no spinning, no hand waving, no dramatic gestures, "
        "subtle natural movement only, fixed camera"
    )

    if motion_style == "walk":
        motion = (
            "takes three slow confident steps forward toward camera, "
            "then stops and stands still in a natural relaxed pose, "
            "slight weight shift to one side, "
            "arms relaxed at sides"
        )
    else:  # sway
        motion = (
            "gentle subtle body sway in place, "
            "slight turn of the torso left then right to show the outfit, "
            "stays in the same spot throughout, "
            "minimal natural movement, like a model holding a pose"
        )

    # Clothing-aware refinement
    if any(w in clothing_l for w in ["saree", "sari", "lehenga"]):
        motion += ", fabric draping naturally with the movement"
    elif any(w in clothing_l for w in ["suit", "blazer", "formal"]):
        motion += ", upright professional posture throughout"

    return f"{identity}, {motion}, {no_dance}"


# ---------------------------------------------------------------------------
# HuggingFace Inference API (Wan2.1-I2V)
# ---------------------------------------------------------------------------

def _call_wan_api(image: Image.Image, motion_prompt: str, cfg: dict) -> bytes:
    """
    Submit image + motion prompt to Wan2.1-I2V-14B-720P via HuggingFace
    Inference API (fal-ai provider).

    Args:
        image:         PIL Image — the final advertisement image
        motion_prompt: Demographically-aware motion description
        cfg:           Loaded config dict

    Returns:
        Raw video bytes (mp4).
    """
    if not HF_TOKEN:
        raise RuntimeError(
            "No HF_TOKEN found in environment (.env). "
            "Set HF_TOKEN=your_huggingface_token to use Wan2.1-I2V."
        )

    wan_cfg   = cfg.get("wan", {})
    model_id  = wan_cfg.get("model", "Wan-AI/Wan2.1-I2V-14B-720P")
    provider  = wan_cfg.get("provider", "fal-ai")

    print(f"[motion] Calling {model_id} via {provider}...")

    client = InferenceClient(
        provider=provider,
        api_key=HF_TOKEN,
    )

    # Save image to temp file for submission
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.convert("RGB").save(tmp.name)
        tmp_path = tmp.name

    try:
        video_bytes = client.image_to_video(
            image=tmp_path,
            prompt=motion_prompt,
            model=model_id,
        )
    finally:
        os.unlink(tmp_path)

    # huggingface_hub may return bytes or a generator depending on version
    if hasattr(video_bytes, "read"):
        return video_bytes.read()
    if isinstance(video_bytes, (bytes, bytearray)):
        return bytes(video_bytes)

    # Some versions return an iterable
    chunks = b""
    for chunk in video_bytes:
        chunks += chunk
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def animate_image(image: Image.Image, config: dict = None,
                  profile: dict = None, clothing: str = "",
                  motion_style: str = "sway") -> str:
    """
    Animate the final advertisement image into a short video.

    Args:
        image:        The final advertisement image (PIL Image).
        config:       Loaded config.yaml dict.
        profile:      Demographic profile dict.
        clothing:     Clothing description from the user's brief.
        motion_style: "sway" (gentle turn, best for accessories/jewellery)
                      or "walk" (walks forward then poses, best for clothing).

    Returns:
        File path to the generated video, saved under outputs/.
    """
    cfg = _load_config(config)

    motion_prompt = build_motion_prompt(
        profile or {}, clothing or "", motion_style
    )
    print(f"[motion] Style: {motion_style} | Prompt: {motion_prompt}")

    video_bytes = _call_wan_api(image, motion_prompt, cfg)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"animation_{int(time.time())}.mp4")
    with open(out_path, "wb") as f:
        f.write(video_bytes)

    print(f"[motion] Saved → {out_path}")
    return out_path
"""
Module 5 - Motion / Animation

Takes the final advertisement image (product already baked in by FLUX,
per Module 4's description-based approach) and animates it via Viggle.

Note: `style` is kept for UI compatibility with the existing dropdown
("Subtle sway" / "Camera pan" / "Zoom in"), but only one motion is
actually implemented right now — a forward walk. Every style value
currently produces the same animation; the dropdown labels are
unchanged pending more driving videos being added.

Unlike Module 4's product compositing, this does NOT strip/reattach
the product before animating — testing showed Viggle's own rendering
of the product during motion transfer (with V4_Preview) is reasonably
faithful on its own, so we let it render natively. If a given profile's
product renders badly, the strip/reattach approach documented during
testing (crop the product region out via GrabCut, inpaint it out of the
source photo before submission, then re-composite it back in tracked to
the wrist via MediaPipe) is the fallback path, not currently wired in.
"""

import os
import time
import tempfile

import cv2
import numpy as np
import requests
import imageio
import yaml
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

VIGGLE_API_KEY = os.getenv("VIGGLE_API_KEY")


def _load_config(config: dict = None) -> dict:
    if config is not None:
        return config
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _viggle_headers():
    if not VIGGLE_API_KEY:
        raise RuntimeError("No VIGGLE_API_KEY found in environment (.env)")
    return {"Authorization": f"Bearer {VIGGLE_API_KEY}"}


def _submit_to_viggle(character_image_path, driving_video_path, viggle_cfg, poll_every=3):
    base = viggle_cfg["base_url"]
    resp = requests.post(
        f"{base}/api/render",
        headers=_viggle_headers(),
        files={
            "ref_image": open(character_image_path, "rb"),
            "driving_video": open(driving_video_path, "rb"),
        },
        data={
            "background_mode": viggle_cfg.get("background_mode", "transparent"),
            "model": viggle_cfg.get("model", "V4_Preview"),
        },
    )
    resp.raise_for_status()
    job = resp.json()

    while True:
        status = requests.get(f"{base}/api/render/{job['job_id']}", headers=_viggle_headers()).json()
        if status["status"] == "complete":
            return status
        if status["status"] == "error":
            raise RuntimeError(f"Viggle job failed: {status.get('error_message')}")
        time.sleep(poll_every)


def _download_outputs(status, tmp_dir):
    video_path = os.path.join(tmp_dir, "anim.mp4")
    with open(video_path, "wb") as f:
        f.write(requests.get(status["cdn_url"]).content)

    mask_path = None
    if status.get("mask_cdn_url"):
        mask_path = os.path.join(tmp_dir, "mask.mp4")
        with open(mask_path, "wb") as f:
            f.write(requests.get(status["mask_cdn_url"]).content)

    return video_path, mask_path


def _sample_background_color(image: Image.Image, patch_frac: float = 0.05) -> tuple:
    """
    Sample the four corners of the source image — guaranteed to be pure
    background in these studio-style ad photos — and average them.
    Avoids needing a hardcoded color per background choice in config.yaml.
    """
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    ph, pw = max(1, int(h * patch_frac)), max(1, int(w * patch_frac))
    corners = [
        arr[0:ph, 0:pw], arr[0:ph, w - pw:w],
        arr[h - ph:h, 0:pw], arr[h - ph:h, w - pw:w],
    ]
    pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return tuple(int(v) for v in pixels.mean(axis=0))


def _composite_onto_plate(image: Image.Image, animated_path, mask_path, output_path):
    cap_anim = cv2.VideoCapture(animated_path)
    cap_mask = cv2.VideoCapture(mask_path)

    fps = cap_anim.get(cv2.CAP_PROP_FPS) or 8
    w = int(cap_anim.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_anim.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg_color = _sample_background_color(image)
    bg = np.full((h, w, 3), bg_color, dtype=np.uint8)

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p")
    n = 0
    while True:
        ret_a, frame = cap_anim.read()
        ret_m, mask_frame = cap_mask.read()
        if not ret_a or not ret_m:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask_gray = (cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                     if len(mask_frame.shape) == 3 else mask_frame)
        mask_gray = cv2.resize(mask_gray, (w, h))
        mask_gray = cv2.GaussianBlur(mask_gray, (5, 5), 0)
        alpha = mask_gray.astype(float) / 255.0
        alpha3 = np.stack([alpha, alpha, alpha], axis=2)

        composite = (frame * alpha3 + bg * (1 - alpha3)).astype(np.uint8)
        writer.append_data(composite)
        n += 1

    cap_anim.release()
    cap_mask.release()
    writer.close()
    print(f"[motion] Saved → {output_path} ({n} frames at {w}x{h})")
    return output_path


def animate_image(image: Image.Image, style: str = "Subtle sway", config: dict = None) -> str:
    """
    Args:
        image: the final advertisement image (product already baked in).
        style: kept for UI compatibility — every style currently produces
               the same walking animation.
        config: loaded config.yaml dict. Loaded from disk if None.

    Returns:
        File path to the generated video clip, saved under outputs/.
    """
    cfg = _load_config(config)
    viggle_cfg = cfg.get("viggle", {})
    driving_video_path = viggle_cfg.get("driving_video")

    if not driving_video_path or not os.path.exists(driving_video_path):
        raise RuntimeError(
            f"Driving video not found at '{driving_video_path}'. "
            "Set viggle.driving_video in config.yaml and make sure the file exists."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        char_path = os.path.join(tmp_dir, "character.png")
        image.convert("RGB").save(char_path)

        print("[motion] Submitting to Viggle...")
        status = _submit_to_viggle(char_path, driving_video_path, viggle_cfg)
        animated_path, mask_path = _download_outputs(status, tmp_dir)

        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"animation_{int(time.time())}.mp4")
        _composite_onto_plate(image, animated_path, mask_path, out_path)

    return out_path
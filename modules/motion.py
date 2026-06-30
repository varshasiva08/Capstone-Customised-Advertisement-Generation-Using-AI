"""
Module 5 - Motion / Animation

Animates the final advertisement image using Viggle AI motion transfer,
driven by a pre-saved local walking video selected via a style dropdown
in app.py. No upload required from the end user.
"""

import os
import time
import requests
import cv2
import numpy as np
import imageio
from PIL import Image

VIGGLE_BASE = "https://apis.viggle.ai"

# Pre-saved driving videos, mapped to dropdown style names.
# Add/replace files in assets/motion_videos/ and update paths here.
STYLE_VIDEOS = {
    "Subtle sway":  "assets/motion_videos/subtle_sway.mp4",
    "Camera pan":   "assets/motion_videos/camera_pan.mp4",
    "Zoom in":      "assets/motion_videos/zoom_in.mp4",
    "Runway walk":  "assets/motion_videos/runway_walk.mp4",
}


def _headers(api_key):
    return {"Authorization": f"Bearer {api_key}"}


def _make_background_plate(image_path: str) -> str:
    out_path = image_path.rsplit(".", 1)[0] + "_bgplate.png"
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_bg = np.array([5, 15, 100])
    upper_bg = np.array([35, 140, 255])
    bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
    model_mask = cv2.bitwise_not(bg_mask)

    kernel = np.ones((15, 15), np.uint8)
    model_mask = cv2.morphologyEx(model_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    model_mask = cv2.dilate(model_mask, kernel, iterations=3)

    bg_rough = cv2.inpaint(img, model_mask, inpaintRadius=35, flags=cv2.INPAINT_NS)
    bg_clean = cv2.inpaint(bg_rough, model_mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)

    cv2.imwrite(out_path, bg_clean)
    return out_path


def _submit_viggle_job(api_key: str, ref_image_path: str, driving_video_path: str) -> str:
    with open(ref_image_path, "rb") as img, open(driving_video_path, "rb") as vid:
        job = requests.post(
            f"{VIGGLE_BASE}/api/render",
            headers=_headers(api_key),
            files={
                "ref_image": (os.path.basename(ref_image_path), img, "image/png"),
                "driving_video": (os.path.basename(driving_video_path), vid, "video/mp4"),
            },
            data={"background_mode": "transparent"},
        ).json()
    if "detail" in job:
        raise RuntimeError(f"Viggle error: {job['detail']}")
    return job["job_id"]


def _poll_viggle_job(job_id: str, timeout: int = 300, poll_interval: int = 3) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        status = requests.get(f"{VIGGLE_BASE}/api/render/{job_id}").json()
        if status["status"] == "complete":
            return status
        if status["status"] == "failed":
            raise RuntimeError(f"Viggle job failed: {status.get('error_message')}")
        time.sleep(poll_interval)
    raise TimeoutError("Viggle render timed out")


def _download(url: str, out_path: str) -> str:
    with open(out_path, "wb") as f:
        f.write(requests.get(url).content)
    return out_path


def _composite_onto_background(bg_image_path, animated_path, mask_path, output_path):
    cap_anim = cv2.VideoCapture(animated_path)
    cap_mask = cv2.VideoCapture(mask_path)
    fps = cap_anim.get(cv2.CAP_PROP_FPS) or 8
    w = int(cap_anim.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_anim.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg = cv2.imread(bg_image_path)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LANCZOS4)

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p")
    while True:
        ret_a, frame = cap_anim.read()
        ret_m, mask_frame = cap_mask.read()
        if not ret_a or not ret_m:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) if len(mask_frame.shape) == 3 else mask_frame
        mask_gray = cv2.resize(mask_gray, (w, h))
        mask_gray = cv2.GaussianBlur(mask_gray, (5, 5), 0)
        alpha = mask_gray.astype(float) / 255.0
        alpha3 = np.stack([alpha] * 3, axis=2)
        composite = (frame * alpha3 + bg * (1 - alpha3)).astype(np.uint8)
        writer.append_data(composite)

    cap_anim.release()
    cap_mask.release()
    writer.close()
    return output_path


def animate_image(
    image: Image.Image,
    style: str = "Subtle sway",
    api_key: str | None = None,
    output_dir: str = "outputs",
) -> str:
    """
    Args:
        image: the final advertisement image (PIL.Image).
        style: one of the keys in STYLE_VIDEOS — maps to a pre-saved
               local driving video (no upload needed from the user).
        api_key: Viggle API key. Falls back to VIGGLE_API_KEY env var.
        output_dir: where intermediate/final files are written.

    Returns:
        File path to the generated, background-composited video clip.
    """
    api_key = api_key or os.getenv("VIGGLE_API_KEY")
    if not api_key:
        raise RuntimeError("No Viggle API key provided (arg or VIGGLE_API_KEY env var)")

    if style not in STYLE_VIDEOS:
        raise ValueError(f"Unknown style '{style}'. Choose from: {list(STYLE_VIDEOS)}")

    driving_video_path = STYLE_VIDEOS[style]
    if not os.path.exists(driving_video_path):
        raise FileNotFoundError(
            f"Driving video not found at '{driving_video_path}'. "
            f"Make sure it's saved in assets/motion_videos/."
        )

    os.makedirs(output_dir, exist_ok=True)
    tmp_img_path = os.path.join(output_dir, f"_motion_input_{int(time.time())}.png")
    image.save(tmp_img_path)

    bg_plate_path = _make_background_plate(tmp_img_path)

    job_id = _submit_viggle_job(api_key, tmp_img_path, driving_video_path)
    status = _poll_viggle_job(job_id)

    base = os.path.splitext(os.path.basename(tmp_img_path))[0]
    anim_path = _download(status["cdn_url"], f"{output_dir}/{base}_anim_raw.mp4")
    mask_path = _download(status["mask_cdn_url"], f"{output_dir}/{base}_mask.mp4")

    final_path = f"{output_dir}/{base}_walking_ad.mp4"
    _composite_onto_background(bg_plate_path, anim_path, mask_path, final_path)
    return final_path
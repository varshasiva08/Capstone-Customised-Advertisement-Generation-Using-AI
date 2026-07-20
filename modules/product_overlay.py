# """
# Module 4 - Product / Brand Compositor (placeholder)

# Owner: [teammate]

# Takes the generated advertisement image and composites a product
# photo and/or brand logo onto it. The prompt template already leaves
# "empty space reserved for product placement and logo" in the frame,
# so there's room to work with.

# Suggested approach:
#   1. Remove the background from the uploaded product image (e.g. rembg)
#   2. Paste it onto the model image at `position`, scaled to fit
#   3. Optionally stamp a logo in a corner as a watermark

# This module is intentionally not implemented yet - app.py only wires
# up the UI controls (file uploader + position selector) for it.
# """

# from PIL import Image


# def composite_product(model_image: Image.Image,
#                        product_image: Image.Image,
#                        position: str = "Bottom Right") -> Image.Image:
#     """
#     Args:
#         model_image: the generated advertisement image.
#         product_image: the uploaded product/logo image.
#         position: one of "Top Left", "Top Right", "Bottom Right".

#     Returns:
#         A new PIL Image with the product/logo composited on top.
#     """
#     raise NotImplementedError("Module 4 - product compositing not implemented yet")














"""
Module 4 - Product / Brand Compositor

Owner: [your name]

Takes the generated advertisement image and composites a product photo
onto it realistically. The prompt template already leaves "empty space
reserved for product placement and logo" in the frame, so there's room
to work with.

Realism pipeline (in order):
    1. Remove the product image's background        (OpenCV GrabCut)
    2. Resize product to a sane fraction of the ad's width
    3. Match the product's lighting/tone to the region it's placed on
    4. Draw a soft drop shadow under the product so it looks grounded
    5. Alpha-composite with feathered edges (no hard "sticker" cutout)

Background removal uses OpenCV's GrabCut algorithm rather than an AI
model (e.g. rembg). GrabCut works well for typical product photography
(centered subject, plain/solid background) and — importantly — installs
via a pre-built wheel with no native compilation required, unlike
rembg's numba/llvmlite dependency chain which fails to build on several
common setups (older macOS, missing LLVM toolchain, etc).

Each step is a separate function so they can be tested/tuned in isolation.
"""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ---------- Position -> anchor mapping ----------

POSITIONS = {
    "Top Left": "top_left",
    "Top Right": "top_right",
    "Bottom Right": "bottom_right",
    "Bottom Left": "bottom_left",
}

MARGIN_FRAC = 0.04          # edge margin, as a fraction of the shorter side
PRODUCT_WIDTH_FRAC = 0.22   # product width, as a fraction of model width


# ---------- Step 1: background removal (OpenCV GrabCut) ----------

def remove_background(product_image: Image.Image, margin_frac: float = 0.04) -> Image.Image:
    """
    Return an RGBA image with the product's background removed, using
    OpenCV's GrabCut algorithm. Assumes the product is roughly centered
    with some background visible around it (typical product photography).
    """
    rgb_img = product_image.convert("RGB")
    arr = np.array(rgb_img)  # H x W x 3, RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Initial rectangle: assume the product sits inside a margin around
    # the edges, background is more likely near the border.
    mx = int(w * margin_frac)
    my = int(h * margin_frac)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    try:
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # Fallback: if GrabCut fails (e.g. degenerate image), treat
        # everything as foreground rather than erroring out.
        alpha = np.full((h, w), 255, dtype=np.uint8)
        rgba = np.dstack([arr, alpha])
        return Image.fromarray(rgba, "RGBA")

    # Pixels marked "definite/probable foreground" become opaque
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")

    # Clean up small holes/noise in the mask
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    rgba = np.dstack([arr, fg_mask])
    return Image.fromarray(rgba, "RGBA")


# ---------- Step 2: resize ----------

def resize_product(product_rgba: Image.Image, model_image: Image.Image,
                    width_frac: float = PRODUCT_WIDTH_FRAC) -> Image.Image:
    """Scale the product so its width is a fixed fraction of the model image."""
    target_w = int(model_image.width * width_frac)
    scale = target_w / product_rgba.width
    target_h = int(product_rgba.height * scale)
    return product_rgba.resize((max(1, target_w), max(1, target_h)), Image.LANCZOS)


# ---------- Step 3: lighting / tone match ----------

def match_lighting(product_rgba: Image.Image, model_image: Image.Image,
                    box: tuple[int, int, int, int]) -> Image.Image:
    """
    Sample the average brightness of the region on the model image where
    the product will sit, then nudge the product's brightness to match,
    so it doesn't look pasted from a totally different photoshoot.
    """
    region = model_image.convert("RGB").crop(box)
    region_arr = np.asarray(region, dtype=np.float32)
    region_mean = region_arr.mean()

    prod_rgb = product_rgba.convert("RGB")
    prod_arr = np.asarray(prod_rgb, dtype=np.float32)
    alpha = np.asarray(product_rgba.split()[-1], dtype=np.float32) / 255.0

    mask = alpha > 0.1
    if mask.sum() == 0:
        return product_rgba
    prod_mean = prod_arr[mask].mean()
    if prod_mean < 1e-3:
        return product_rgba

    # Blend 60% toward scene brightness, keep 40% of product's own contrast
    target_mean = 0.6 * region_mean + 0.4 * prod_mean
    gain = np.clip(target_mean / prod_mean, 0.7, 1.4)  # avoid extreme shifts

    adjusted = np.clip(prod_arr * gain, 0, 255).astype(np.uint8)
    adjusted_img = Image.fromarray(adjusted, mode="RGB")

    out = Image.new("RGBA", product_rgba.size)
    out.paste(adjusted_img, (0, 0))
    out.putalpha(product_rgba.split()[-1])
    return out


# ---------- Step 4: drop shadow ----------

def make_shadow(product_rgba: Image.Image, blur_radius: int = 12,
                 opacity: int = 90, y_offset: int = 10) -> Image.Image:
    """Soft blurred silhouette shadow, offset slightly, for grounding."""
    alpha = product_rgba.split()[-1]
    shadow = Image.new("RGBA", product_rgba.size, (0, 0, 0, 0))
    black = Image.new("RGBA", product_rgba.size, (0, 0, 0, opacity))
    shadow = Image.composite(black, shadow, alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))

    pad = blur_radius * 2 + y_offset
    canvas = Image.new("RGBA", (product_rgba.width + pad * 2,
                                 product_rgba.height + pad * 2), (0, 0, 0, 0))
    canvas.paste(shadow, (pad, pad + y_offset), shadow)
    return canvas


# ---------- Step 5: feathered alpha composite ----------

def feather_alpha(product_rgba: Image.Image, radius: float = 1.5) -> Image.Image:
    """Slightly blur the alpha channel so edges aren't a hard cutout."""
    r, g, b, a = product_rgba.split()
    a = a.filter(ImageFilter.GaussianBlur(radius))
    return Image.merge("RGBA", (r, g, b, a))


# ---------- Position math ----------

def _anchor_xy(position: str, model_image: Image.Image,
               product_size: tuple[int, int]) -> tuple[int, int]:
    margin = int(min(model_image.size) * MARGIN_FRAC)
    pw, ph = product_size
    mw, mh = model_image.size
    key = POSITIONS.get(position, "bottom_right")

    if key == "top_left":
        return margin, margin
    if key == "top_right":
        return mw - pw - margin, margin
    if key == "bottom_left":
        return margin, mh - ph - margin
    return mw - pw - margin, mh - ph - margin  # bottom_right default


# ---------- Public entry point ----------

def composite_product(model_image: Image.Image,
                       product_image: Image.Image,
                       position: str = "Bottom Right") -> Image.Image:
    """
    Composite a product image realistically onto the generated ad photo.

    Args:
        model_image: the generated advertisement image.
        product_image: the uploaded product/logo image.
        position: one of "Top Left", "Top Right", "Bottom Right", "Bottom Left".

    Returns:
        A new PIL Image (RGB) with the product composited on top.
    """
    base = model_image.convert("RGBA")

    # 1. Remove background
    product_rgba = remove_background(product_image)

    # 2. Resize to a sensible scale relative to the ad photo
    product_rgba = resize_product(product_rgba, base)

    # Work out where it will land before sampling lighting there
    x, y = _anchor_xy(position, base, product_rgba.size)
    box = (
        max(0, x), max(0, y),
        min(base.width, x + product_rgba.width),
        min(base.height, y + product_rgba.height),
    )

    # 3. Match lighting/tone to that region of the scene
    product_rgba = match_lighting(product_rgba, base, box)

    # 4. Feather edges so it's not a hard sticker cutout
    product_rgba = feather_alpha(product_rgba, radius=1.5)

    # 5. Build and paste the drop shadow first, then the product on top
    shadow_layer = make_shadow(product_rgba)
    shadow_pad = (shadow_layer.width - product_rgba.width) // 2
    base.alpha_composite(shadow_layer, (x - shadow_pad, y - shadow_pad))
    base.alpha_composite(product_rgba, (x, y))

    return base.convert("RGB")


# ---------- CLI test ----------
if __name__ == "__main__":
    model_img = Image.new("RGB", (768, 1280), (230, 220, 210))
    product_img = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(product_img)
    draw.ellipse((50, 50, 350, 350), fill=(200, 30, 30, 255))
    product_img = product_img.convert("RGB")

    result = composite_product(model_img, product_img, "Bottom Right")
    result.save("test_composite_output.png")
    print("Saved test_composite_output.png —", result.size)

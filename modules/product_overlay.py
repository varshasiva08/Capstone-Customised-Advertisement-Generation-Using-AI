"""
Module 4 - Product / Brand Compositor (placeholder)

Owner: [teammate]

Takes the generated advertisement image and composites a product
photo and/or brand logo onto it. The prompt template already leaves
"empty space reserved for product placement and logo" in the frame,
so there's room to work with.

Suggested approach:
  1. Remove the background from the uploaded product image (e.g. rembg)
  2. Paste it onto the model image at `position`, scaled to fit
  3. Optionally stamp a logo in a corner as a watermark

This module is intentionally not implemented yet - app.py only wires
up the UI controls (file uploader + position selector) for it.
"""

from PIL import Image


def composite_product(model_image: Image.Image,
                       product_image: Image.Image,
                       position: str = "Bottom Right") -> Image.Image:
    """
    Args:
        model_image: the generated advertisement image.
        product_image: the uploaded product/logo image.
        position: one of "Top Left", "Top Right", "Bottom Right".

    Returns:
        A new PIL Image with the product/logo composited on top.
    """
    raise NotImplementedError("Module 4 - product compositing not implemented yet")

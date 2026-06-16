"""
Module 5 - Motion / Animation (placeholder)

Owner: [teammate]

Takes the final advertisement image (after product compositing) and
produces a short animated clip, e.g. via Stable Video Diffusion or
AnimateDiff.

This module is intentionally not implemented yet - app.py only wires
up the UI control (animation style selector) for it.
"""

from PIL import Image


def animate_image(image: Image.Image, style: str = "Subtle sway") -> str:
    """
    Args:
        image: the final advertisement image.
        style: one of "Subtle sway", "Camera pan", "Zoom in".

    Returns:
        File path to the generated video clip.
    """
    raise NotImplementedError("Module 5 - animation not implemented yet")

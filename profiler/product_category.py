"""
Shared product-category vocabulary and inference for AdFidelity.

Used in two places so they can never drift apart:

  * ``app.py`` - ``detect_product_in_brief()`` scans the *brief text* for a
    product mention (drives the "upload your <x>" chat message).
  * ``app.py`` - the generate path infers the category from the *vision
    description* of the uploaded photo, so the correct pose phrasing is used
    even when the brief never names the product (previously everything that
    wasn't named in the brief silently became a "handbag").

Matching is whole-word (with an optional trailing "s") and the *earliest*
mention in the text wins, so "black top-handle handbag" classifies as a
handbag rather than clothing.
"""

import re

PRODUCT_KEYWORDS = {
    "handbag":    ["handbag", "bag", "purse", "tote", "clutch", "satchel",
                   "backpack", "sling", "crossbody", "shoulder bag"],
    "sunglasses": ["sunglasses", "sunglass", "glasses", "eyewear", "shades",
                   "goggles"],
    "jewelry":    ["jewellery", "jewelry", "necklace", "earring", "bracelet",
                   "ring", "pendant", "chain", "anklet", "bangle", "brooch",
                   "locket", "choker"],
    "watch":      ["watch", "wristwatch", "smartwatch"],
    # Deliberately avoids very generic words ("top", "suit", "outfit") that
    # would mis-fire on things like a "top-handle bag".
    "clothing":   ["saree", "sari", "kurta", "kurti", "lehenga", "salwar",
                   "dress", "gown", "shirt", "blouse", "jacket", "coat",
                   "sweater"],
    "footwear":   ["shoe", "shoes", "heel", "heels", "sandal", "sandals",
                   "sneaker", "sneakers", "boot", "boots", "flats",
                   "footwear"],
    "headwear":   ["cap", "hat", "beanie", "headband", "turban", "hijab"],
    "other":      ["belt", "scarf", "wallet", "accessory", "accessories"],
}


def _first_match(lower_text: str, keywords):
    """Earliest whole-word (optional plural) match -> (pos, keyword) or None."""
    best = None
    for kw in keywords:
        m = re.search(rf"\b{re.escape(kw)}s?\b", lower_text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), kw)
    return best


def infer_category_and_keyword(text):
    """
    Return ``(category, matched_keyword)`` for the earliest product mention,
    or ``(None, None)`` when nothing matches.
    """
    if not text:
        return None, None
    lower = str(text).lower()
    best = None
    for category, keywords in PRODUCT_KEYWORDS.items():
        hit = _first_match(lower, keywords)
        if hit and (best is None or hit[0] < best[0]):
            best = (hit[0], category, hit[1])
    if best is None:
        return None, None
    return best[1], best[2]


def infer_category_from_text(text):
    """Category only (or None). Convenience wrapper."""
    return infer_category_and_keyword(text)[0]

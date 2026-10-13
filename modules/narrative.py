"""
Module 7 - Storyboard mode: context-aware 20s narrative advertisement.

Turns a free-form context ("busy coffee shop", "sunset beach", "corporate
office", "shopping mall", "city street...") into a short TV-style ad built
from ONE verified hero image so the person, outfit and location are
identical in every frame and every shot:

  Shot 1  (aspiration) hero image            -> Wan 2.2 clip (walk)
  Shot 2  (lifestyle)  SAME hero image       -> Wan 2.2 clip (sway)
  Shot 3  (product)    uploaded product      -> real-product beauty close-up
  Brand card           existing ad_compositor card, faded in
  Voiceover            edge-tts (free): an aspiration line ("Step out of the
                       ordinary.") over the open, then a benefit CTA
                       ("{brand} — own the moment. Find your nearest
                       showroom.") that ends exactly as the card appears
  Assembly             frames via cv2 -> each clip re-timed to a common fps
                       (no hard looping: a longer slot becomes a slow push-in),
                       short dissolves, brand card, imageio/libx264 + ffmpeg audio

The hero is checked by local llava against the brief (demographic + clothing,
plus the product when one is described) and re-rolled up to 3x; if a product
is uploaded it is drawn INTO the hero prompt, so the model is holding/wearing
it in every clip (identity stays consistent). Shot 3 is then a real-product
beauty close-up of the uploaded photo. The brand card owns the remainder of
the timeline (~4-5s), so the finished ad is exactly the target duration. Every
configured HF account is tried for Wan; if all fail, shots 1-2 degrade to a
Ken Burns zoom clip so the ad still completes.

Everything is free: Cloudflare FLUX, the public Wan2.2 I2V Space, Ollama
(shot script + tagline + llava verification), and edge-tts (TTS).
"""

import json
import os
import random
import re
import shutil
import subprocess
import time

import cv2
import imageio
import numpy as np
import requests
import yaml
from PIL import Image, ImageFilter

from profiler.prompt_builder import build_prompt, clean_scene

SHOT_BUDGET_SEC = 16.0   # shots get 16s of the 20s ad
CARD_BUDGET_SEC = 4.0    # brand card gets ~4s: 0.5s fade + 3.5s hold


def _safe_print(*args):
    """Console-safe default logger: ASCII-replaces unicode chars that a
    cp1252 terminal cannot encode (Streamlit passes its own log callable)."""
    try:
        print(*args)
    except UnicodeEncodeError:
        s = " ".join(str(a) for a in args)
        print(s.encode("ascii", "replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(config: dict = None) -> dict:
    if config is not None:
        return config
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Shot script — Ollama writes it; fallback templates if it mangles JSON
# ---------------------------------------------------------------------------

def _extract_json(text: str):
    """Pull the first JSON object out of a model reply (tolerates fences
    and surrounding prose)."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def _validate_shots(data) -> list | None:
    """Validate an LLM shot script into our shot dicts (max 3, safe values)."""
    if not isinstance(data, dict):
        return None
    raw = data.get("shots")
    if not isinstance(raw, list) or not raw:
        return None
    shots = []
    for s in raw[:3]:
        if not isinstance(s, dict):
            continue
        scene = clean_scene(s.get("scene"))
        if not scene:
            continue
        action = str(s.get("action") or "").strip()[:160]
        motion = "walk" if str(s.get("motion") or "").strip().lower() == "walk" else "sway"
        try:
            dur = float(s.get("duration") or 6)
        except (TypeError, ValueError):
            dur = 6
        dur = max(4.0, min(7.0, dur))
        shots.append({
            "scene": scene,
            "action": action,
            "motion": motion,
            "duration": dur,
        })
    return shots or None


def _ask_ollama_shots(profile, clothing, brand, context, product_desc, cfg):
    """Ask Ollama to write a 3-shot storyboard for the context."""
    import ollama
    host = ((cfg or {}).get("ollama") or {}).get("host", "http://localhost:11434")
    model = ((cfg or {}).get("ollama") or {}).get("model", "phi3:mini")

    eth  = profile.get("ethnicity", "woman")
    body = profile.get("body_type", "average")
    age  = profile.get("age", "adult")
    prod = product_desc or "the brand's product (a handbag)"

    prompt = (
        "You are a TV commercial storyboard writer for a premium fashion ad. "
        "One model appears throughout and ONE location is used for every shot, "
        "so the ad feels cinematic and consistent.\n"
        f"Context / setting: {context or 'a stylish venue'}. "
        f"Model: {body} {eth} woman in her {age}, wearing {clothing}. "
        f"Product for the ad: {prod}. Brand: {brand}. "
        "Write exactly 3 shots for a 20-second advertisement built around "
        "selling the product emotionally:\n"
        "  Shot 1 (aspiration, motion \"walk\"): she owns the space — a "
        "confident, elegant entrance; the scene must feel aspirational and "
        "beautiful.\n"
        "  Shot 2 (lifestyle, motion \"sway\"): a candid moment of joy and "
        "confidence — she is moving, happy, enjoying the moment, fabric and "
        "light in play.\n"
        "  Shot 3 (intimacy): an emotional close-up on the product — describe "
        "it so the viewer desires it (texture, finish, detail, warm light).\n"
        "Return ONLY valid JSON and nothing else:\n"
        '{"shots":[{"scene":"<scene, 6-15 words>","action":"<action, 6-15 words>",'
        '"motion":"walk|sway","duration":5.0}]}'
    )
    resp = ollama.Client(host=host).chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.4, "num_predict": 400},
    )
    return _validate_shots(_extract_json(resp["message"]["content"]))


def _fallback_shots(context):
    """Keyword-based templates + a generic fallback, so the feature always
    produces a sensible 3-shot plan even if Ollama misbehaves."""
    c = (context or "").lower()

    def t(s1, s2, s3):
        return [
            {"scene": s1,
             "action": "She walks in with quiet confidence and owns the space, natural stride.",
             "motion": "walk", "duration": 6.5},
            {"scene": s2,
             "action": "She turns in a candid moment of joy, fabric and hair moving in the light.",
             "motion": "sway", "duration": 6.5},
            {"scene": s3,
             "action": "Slow intimate reveal of the product — its texture and finish in warm light.",
             "motion": "sway", "duration": 4.5},
        ]

    if any(k in c for k in ("cafe", "café", "coffee", "coffee shop")):
        return t(
            "pavement outside a stylish coffee shop at golden hour, warm string lights, glass storefront, cozy street ambience",
            "inside a cozy coffee shop, wooden tables, a latte-art coffee cup, plants and bookshelves, shallow depth of field",
            "close view inside the coffee shop, the product placed on the table beside the coffee cup, sharp focus")
    if "beach" in c:
        return t(
            "sunlit sandy beach at golden hour, gentle waves, warm breeze, soft haze",
            "seaside promenade, the model walking along the shore, wind in the fabric, ocean bokeh behind",
            "close view of the product held in hand against the ocean backdrop, sharp focus")
    if any(k in c for k in ("office", "corporate", "work")):
        return t(
            "modern corporate office lobby, glass walls, warm daylight, polished floor",
            "open-plan office with warm light, the model present near the desk area",
            "the product placed on a desk in sharp focus, soft office bokeh behind")
    if any(k in c for k in ("mall", "shopping")):
        return t(
            "bright modern shopping mall atrium, glass balustrades, warm retail lighting",
            "inside a designer storefront area of the mall, ambient retail lighting",
            "storefront table display with the product in sharp focus, festive lighting")
    if any(k in c for k in ("street", "road", "pavement", "city", "sidewalk")):
        return t(
            "city street at evening, warm streetlights, soft bokeh traffic behind",
            "city sidewalk with a warm storefront glow, gentle breeze",
            "close view of the product in hand against the city bokeh, sharp focus")
    if "park" in c:
        return t(
            "green city park path at morning, soft sunlight through trees, dewy grass",
            "park bench area with leafy bokeh, the model seated gracefully",
            "close view of the product resting beside her on the bench, sharp focus")

    g = context or "a stylish venue"
    return t(
        f"{g}, atmospheric setting, warm natural light, professional photography",
        f"inside the {g}, mid-distance view of the model in the environment",
        f"close view at the {g}, the product in sharp focus")


def _normalize_durations(shots, budget=SHOT_BUDGET_SEC):
    """Clamp each duration to 4-7s and scale the total down to `budget`.

    The 4.0s floor can push the scaled total back over the budget, so any
    residual overflow is shaved off the longest shots afterwards. The result
    is guaranteed to sum to <= budget.
    """
    shots = [
        {**s, "duration": max(4.0, min(7.0, float(s["duration"])))}
        for s in shots
    ]
    if not shots:
        return shots
    total = sum(s["duration"] for s in shots)
    if total > budget:
        scale = budget / total
        shots = [
            {**s, "duration": max(4.0, min(7.0, s["duration"] * scale))}
            for s in shots
        ]
        # Shave overflow (created by the 4.0 floor) off the longest shot,
        # 0.1s at a time, until the plan fits the budget.
        for _ in range(80):
            total = sum(s["duration"] for s in shots)
            if total <= budget + 1e-6:
                break
            idx = max(range(len(shots)), key=lambda i: shots[i]["duration"])
            if shots[idx]["duration"] - 4.0 <= 1e-6:
                break  # everything is already at the floor
            shots[idx]["duration"] = max(
                4.0, round(shots[idx]["duration"] - min(0.1, total - budget), 1)
            )
    return [{**s, "duration": round(s["duration"], 1)} for s in shots]


def _generate_shots(profile, clothing, brand, context, product_desc, cfg):
    """Produce a 3-shot plan: Ollama first, fallback templates second.

    Always returns EXACTLY 3 shots. If the LLM yields fewer (or a shot is
    malformed), the tail is topped up from the deterministic fallback, so the
    storyboard can never crash on a short plan. Durations are normalised once
    later, where the real budget is known.
    """
    try:
        shots = _ask_ollama_shots(profile, clothing, brand, context, product_desc, cfg)
    except Exception:
        shots = None
    if not shots:
        shots = _fallback_shots(context)
    shots = list(shots)
    if len(shots) < 3:
        filler = _fallback_shots(context)
        shots = shots + filler[len(shots):3]
    return shots[:3]


# ---------------------------------------------------------------------------
# Voiceover (edge-tts, free)
# ---------------------------------------------------------------------------

def tts_voiceover(text: str, out_path: str, voice: str = "en-US-JennyNeural") -> str | None:
    """Synthesise a voiceover mp3 with edge-tts. Returns None on failure."""
    try:
        import asyncio
        import edge_tts
        comm = edge_tts.Communicate(text, voice)
        if hasattr(comm, "save_sync"):
            comm.save_sync(out_path)
        else:
            asyncio.run(comm.save(out_path))
        if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            return out_path
    except Exception:
        return None
    return None


def _ken_burns_clip(img, out_path, duration, fps, zoom_in=True):
    """Render a smooth cinematic zoom on a still image into an mp4.

    Free fallback whenever Wan motion is unavailable (all HF accounts failed)
    and for the product-reveal shot: zoom-in for the entering/posing shots, a
    gentle zoom-out dolly for the product close-up. Returns out_path, or None
    on failure.
    """
    try:
        arr = np.asarray(img.convert("RGB"))
        H, W = arr.shape[:2]
        n = max(1, int(round(duration * fps)))
        s0, s1 = (1.0, 1.12) if zoom_in else (1.12, 1.0)
        frames = []
        for i in range(n):
            t = i / max(1, n - 1)
            s = s0 + (s1 - s0) * t
            nw = int(W * s) | 1
            nh = int(H * s) | 1
            big = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            x0 = (nw - W) // 2
            y0 = (nh - H) // 2
            frames.append(big[y0:y0 + H, x0:x0 + W])
        writer = imageio.get_writer(
            out_path, fps=fps, codec="libx264", pixelformat="yuv420p"
        )
        for f in frames:
            writer.append_data(f)
        writer.close()
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            return out_path
    except Exception:
        return None
    return None


def _product_beauty(hero, product_img, out_path, duration, fps=30.0):
    """Shot 3 as a real-product beauty close-up: the uploaded product, its
    background removed and grounded with a soft shadow, centred on a soft
    backdrop colour-matched to the hero, then a slow Ken Burns push-in.

    Reuses product_overlay's GrabCut / resize / lighting / shadow helpers so
    the result matches the app's other compositing. Saves the composed still
    for inspection. Returns out_path, or a hero push-in on failure.
    """
    try:
        from modules.product_overlay import (
            remove_background, resize_product, match_lighting,
            feather_alpha, make_shadow,
        )
        W, H = hero.size
        # Soft studio backdrop derived from the hero (blur + darken).
        bg = hero.convert("RGB").filter(
            ImageFilter.GaussianBlur(radius=max(6, W // 40)))
        bg = Image.blend(bg, Image.new("RGB", (W, H), (18, 18, 18)), 0.45)

        prod = remove_background(product_img)
        prod = resize_product(prod, bg, width_frac=0.62)
        if prod.height > int(0.92 * H) or prod.width > int(0.92 * W):
            prod.thumbnail((int(0.92 * W), int(0.92 * H)), Image.LANCZOS)
        x = (W - prod.width) // 2
        y = (H - prod.height) // 2
        prod = match_lighting(prod, bg, (x, y, x + prod.width, y + prod.height))
        prod = feather_alpha(prod, radius=1.5)

        shadow = make_shadow(prod, blur_radius=max(10, W // 60),
                             opacity=110, y_offset=max(6, H // 80))
        pad = (shadow.width - prod.width) // 2
        canvas = bg.convert("RGBA")
        canvas.alpha_composite(shadow, (max(0, x - pad), max(0, y - pad)))
        canvas.alpha_composite(prod, (x, y))
        still = canvas.convert("RGB")
        still.save(os.path.join(os.path.dirname(out_path),
                                "story_shot3_product.png"))
        return _ken_burns_clip(still, out_path, duration, fps, zoom_in=True)
    except Exception:
        return _ken_burns_clip(hero, out_path, duration, fps, zoom_in=True)


# ---------------------------------------------------------------------------
# Hero verification (local llava — free, offline)
# ---------------------------------------------------------------------------

def _hero_matches_profile(img, profile, clothing, cfg, product_desc=None):
    """Ask local Ollama llava whether the generated hero matches the brief
    (demographic + clothing, plus the product when one is described). Returns
    True on failure so verification never blocks the ad."""

    pro = ((cfg or {}).get("product_describe") or {})
    host = (pro.get("host") or "http://localhost:11434").rstrip("/")
    model = pro.get("model") or "llava"

    import base64
    from io import BytesIO

    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    eth      = str(profile.get("ethnicity", "woman"))
    body     = str(profile.get("body_type", ""))
    age      = str(profile.get("age", ""))
    attire   = clean_scene(clothing) or "the specified outfit"
    body_txt = f", {body} build" if body else ""
    age_txt  = f", in her {age}" if age else ""
    prod_txt = (f", and holding or wearing {product_desc}" if product_desc else "")

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": (
                f"Does this advertisement photo show a {eth} woman{body_txt}"
                f"{age_txt}, wearing {attire}{prod_txt}? "
                "Reply with exactly 'Yes' or 'No'."
            ),
            "images": [b64],
        }],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 8},
    }
    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        answer = ((resp.json().get("message") or {}).get("content") or "").strip().lower()
        return answer.startswith("yes")
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Assembly: frames -> 20s mp4 -> mux voiceover
# ---------------------------------------------------------------------------

def _read_clip(path, fps=24.0, W=None, H=None):
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if W and H and frames and (
        frames[0].shape[1] != W or frames[0].shape[0] != H
    ):
        frames = [
            cv2.resize(f, (W, H), interpolation=cv2.INTER_LANCZOS4)
            for f in frames
        ]
    return frames, fps


def _slot_frames(frames, src_fps, dst_fps, slot_sec, push=True):
    """Map a clip onto a fixed-duration slot at a common fps WITHOUT looping.

    The clip plays at its natural speed for its full length; if the slot is
    longer, the tail becomes a slow cinematic push-in on the last frame (no
    visible repetition or jump-cut). Returns exactly int(slot_sec * dst_fps)
    frames.
    """
    n_src = len(frames)
    if n_src == 0:
        return frames
    slot_sec = max(0.1, float(slot_sec))
    src_fps = src_fps or dst_fps or 24.0
    dst_fps = dst_fps or src_fps
    n_out = max(2, int(round(slot_sec * dst_fps)))
    clip_end = n_src / max(1e-9, src_fps)
    last = frames[-1]
    H, W = last.shape[:2]
    out = []
    for i in range(n_out):
        t = i / max(1e-9, dst_fps)
        if t < clip_end:
            idx = min(n_src - 1, int(round(t * src_fps)))
            out.append(frames[idx])
        elif push:
            # extension: gentle push-in on the final frame (cinematic hold)
            frac = min(1.0, (t - clip_end) / max(1e-6, slot_sec - clip_end))
            scale = 1.0 + 0.06 * frac
            nw = max(2, int(W * scale)) | 1
            nh = max(2, int(H * scale)) | 1
            big = cv2.resize(last, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            x0, y0 = (nw - W) // 2, (nh - H) // 2
            out.append(big[y0:y0 + H, x0:x0 + W])
        else:
            out.append(last)
    return out


def _blend_bridge(prev_frames, next_frames, n):
    """Crossfade the last n frames of prev into the first n frames of next."""
    if n <= 0 or len(prev_frames) < n or len(next_frames) < n:
        return prev_frames + next_frames
    out = list(prev_frames[:-n])
    for i in range(n):
        # Eased (smoothstep) dissolve that reaches the next frame fully at
        # i = n-1, so there is no 1-frame pop at the cut.
        a = (i + 1) / n
        a = a * a * (3 - 2 * a)
        f1 = prev_frames[-n + i].astype(np.float32)
        f2 = next_frames[i].astype(np.float32)
        out.append((f1 * (1 - a) + f2 * a).astype(np.uint8))
    out.extend(next_frames[n:])
    return out


def _mux_audio(video_path, vo_items, out_path):
    """Mix one or more voiceover segments [(path, offset_sec), ...] into the
    mp4, using the ffmpeg binary bundled with imageio-ffmpeg."""
    from imageio_ffmpeg import get_ffmpeg_exe
    items = [(p, off) for p, off in vo_items if p and os.path.exists(p)]
    if not items:
        raise ValueError("no voiceover files given")
    exe = get_ffmpeg_exe()
    cmd = [exe, "-y", "-i", video_path]
    for p, _ in items:
        cmd += ["-i", p]
    segs = []
    for i, (_, off) in enumerate(items):
        delay = int(round(max(0.0, off) * 1000))
        segs.append(f"[{i + 1}:a]aresample=48000,"
                    f"aformat=channel_layouts=stereo,"
                    f"adelay={delay}|{delay}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(items)))
    # dropout_transition=0 stops amix ramping the remaining VO when one
    # segment ends (keeps the second beat at full level under the first).
    fc = (";".join(segs)
          + f";{mix}amix=inputs={len(items)}:normalize=1:dropout_transition=0[a]")
    cmd += [
        "-filter_complex", fc,
        "-map", "0:v",
        "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _audio_duration(path) -> float:
    """Duration in seconds of an audio file, via the bundled ffmpeg."""
    from imageio_ffmpeg import get_ffmpeg_exe
    try:
        r = subprocess.run([get_ffmpeg_exe(), "-i", path],
                           capture_output=True, text=True, timeout=60)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
        if m:
            h, mi, s = (float(g) for g in m.groups())
            return h * 3600 + mi * 60 + s
    except Exception:
        pass
    return 3.0


def assemble_storyboard(
    clip_paths, shot_durations, brand_name, tagline,
    out_path, product_img=None, target_duration=20.0, fps=24.0,
    vo_items=None, log=_safe_print,
):
    """Build the final 20s ad from clips that all derive from ONE hero image:

    - every clip re-timed to the first clip's fps (no hard looping; a longer
      slot becomes a slow push-in on the clip's last frame)
    - short eased dissolves between shots (same scene, so cuts read smoothly)
    - the brand card owns the remainder of the timeline (~4-5s: a 0.5s fade
      plus a hold), so the finished ad is exactly `target_duration` seconds
    - voiceover segments [(path, offset_sec_or_None), ...]; an offset of None
      is resolved here to end 0.2s into the REAL (frame-accurate) card start,
      so the CTA always lands on the card
    """
    from modules.ad_compositor import _build_brand_card

    # First clip fixes resolution + common fps for the whole ad
    first_frames, fps = _read_clip(clip_paths[0], fps)
    if not first_frames:
        raise RuntimeError(f"No frames read from {clip_paths[0]}")
    W, H = first_frames[0].shape[1], first_frames[0].shape[0]

    total_frames = int(target_duration * fps)
    fade_n = max(1, int(0.5 * fps))

    # --- Re-time each clip onto its slot, all at the common fps ---
    shot_lists = []
    for path, dur in zip(clip_paths, shot_durations):
        frames, src_fps = _read_clip(path, fps, W, H)
        shot_lists.append(_slot_frames(frames, src_fps, fps, dur))

    # --- Join with short, eased dissolves (smooth, no pop) ---
    joined = shot_lists[0]
    for i in range(1, len(shot_lists)):
        bridge = min(int(0.35 * fps), len(joined) // 4, len(shot_lists[i]) // 4)
        joined = _blend_bridge(joined, shot_lists[i], bridge)

    # Shots end here; the brand card owns the rest of the timeline. This is
    # the frame-accurate card start (dissolves already accounted for).
    card_start_sec = len(joined) / fps

    # --- Brand card (colour sampled from the last frame's corners) ---
    last = joined[-1]
    corners = np.array([last[0, 0], last[0, -1], last[-1, 0], last[-1, -1]])
    bg = tuple(int(v) for v in corners.mean(axis=0))
    if sum(bg) < 300:
        bg = (245, 245, 245)
    # Cut the product out for the card too, so it reads as a clean product
    # (not a pasted rectangle) — same treatment as the shot-3 still.
    card_product = product_img
    if product_img is not None:
        try:
            from modules.product_overlay import remove_background
            card_product = remove_background(product_img)
        except Exception:
            card_product = product_img
    card = _build_brand_card(W, H, brand_name, tagline, card_product, bg_color=bg)

    # Eased fade from the last shot into the card, then hold the card to the
    # end. Card = remainder, so the ad is exactly target_duration seconds.
    for i in range(fade_n):
        a = (i + 1) / fade_n
        joined.append((last * (1 - a) + card * a).astype(np.uint8))
    if len(joined) < total_frames:
        joined.extend([card] * (total_frames - len(joined)))
    joined = joined[:total_frames]

    # Resolve VO offsets: None -> end 0.2s into the real card start.
    resolved = []
    for p, off in (vo_items or []):
        if off is None:
            off = max(0.0, card_start_sec - _audio_duration(p) + 0.2)
        resolved.append((p, off))

    tmp_video = out_path + ".novo.mp4"
    writer = imageio.get_writer(
        tmp_video, fps=fps, codec="libx264", pixelformat="yuv420p"
    )
    for f in joined:
        writer.append_data(f)
    writer.close()

    if resolved:
        try:
            _mux_audio(tmp_video, resolved, out_path)
            log("[narrative] Voiceover mixed in.")
        except Exception as e:
            log(f"[narrative] Audio mix failed ({e}) — keeping silent video.")
            shutil.copy(tmp_video, out_path)
    else:
        shutil.copy(tmp_video, out_path)
    os.unlink(tmp_video)

    log(f"[narrative] Final ad: {len(joined) / fps:.1f}s → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_storyboard(
    profile, clothing, brand_name, context=None, cfg=None,
    product_file=None, brief="", target_duration=20.0, log=_safe_print,
):
    """Hero-anchored storyboard: ONE verified FLUX hero drives EVERY shot.

    Pipeline: context -> Ollama shot script (durations/motions only) -> one
    llava-verified hero image (re-rolled up to 3x), with the uploaded product
    drawn INTO the prompt so the model holds/wears it -> shot 1 (aspiration)
    + shot 2 (lifestyle) animate the SAME hero with Wan -> shot 3 is a
    real-product beauty close-up of the uploaded photo -> voiceover (CTA ends
    as the brand card appears) -> strict `target_duration` mp4. Same person,
    same outfit, same location, product in hand — by construction.

    Wan is resilient: every configured HF account is tried, and if all fail
    for a shot it degrades to a cinematic Ken Burns zoom rather than failing.
    """
    cfg = _load_config(cfg)
    os.makedirs("outputs", exist_ok=True)
    run_id = int(time.time())
    from generator.flux_pipeline import generate_image
    from modules.motion import animate_image
    from modules.product_describe import describe_product
    from modules.ad_compositor import generate_tagline
    from profiler.product_category import infer_category_from_text

    # --- Product: described so FLUX draws her HOLDING/WEARING it, and kept
    #     for the real-product beauty close-up in shot 3 ---
    product_img_pil = None
    product_desc = None
    product_category = "other"
    if product_file is not None:
        product_img_pil = Image.open(product_file).convert("RGB")
        log("👁️ Identifying product...")
        product_desc = describe_product(product_img_pil)
        product_category = (
            infer_category_from_text(product_desc)
            or infer_category_from_text(brief)
            or "other"
        )
        log(f"🛍️ Product: {product_desc}")
        log(f"🧩 Product type: {product_category}")

    # --- Shot script: durations + motions only (scene text drives the hero) ---
    log("🎬 Writing shot script (Ollama)...")
    shots = _generate_shots(profile, clothing, brand_name, context, product_desc, cfg)
    shots = _normalize_durations(
        shots, max(10.0, target_duration - CARD_BUDGET_SEC)
    )
    if not shots or len(shots) < 3:
        raise RuntimeError("Could not produce a 3-shot script for this context.")
    for i, s in enumerate(shots, 1):
        log(f"   Shot {i} [{s['motion']}] {s['duration']}s")

    hero_scene = shots[0]["scene"] or (context or "a stylish venue")

    # --- ONE hero image; llava-verified, re-rolled up to 3x ---
    hero_img = None
    for attempt in range(1, 4):
        log(f"🖼️ Hero image attempt {attempt}/3 (FLUX): {hero_scene[:70]}...")
        prompt = build_prompt(
            profile, clothing, "Pure White", cfg,
            product_description=product_desc,
            product_category=product_category,
            scene=hero_scene,
        )
        seed = random.randint(0, 999999)
        hero_img, provider, res, steps_used = generate_image(prompt, seed, cfg)
        hero_img.save(f"outputs/story_{run_id}_hero.png")
        if _hero_matches_profile(hero_img, profile, clothing, cfg,
                                 product_desc=product_desc):
            log("✅ Hero verified against the brief (llava).")
            break
        log("⚠️ Hero doesn't match the brief — generating again...")
    else:
        log("⚠️ Keeping last hero (brief could not be verified after 3 tries).")

    # --- The product was drawn INTO the hero prompt above, so the model is
    #     holding/wearing it in every animated clip (no floating cutout). ---

    # --- Clips: shots 1-2 animate the SAME hero with Wan; shot 3 = beauty ---
    clips = []
    used_fallback = []
    for i in (1, 2):
        s = shots[i - 1]
        log(f"🎞️ Shot {i}: animating hero ({s['motion']}) via Wan2.2...")
        try:
            clip = animate_image(
                hero_img, config=cfg, profile=profile,
                clothing=clothing, motion_style=s["motion"], scene=hero_scene,
            )
        except Exception as e:
            # animate_image tries every configured HF account; if all fail
            # (quota, auth, Space/network), degrade this shot to a cinematic
            # Ken Burns zoom instead of killing the whole ad.
            log(f"⚠️ Wan failed for shot {i} ({type(e).__name__}) — "
                "using cinematic zoom for this shot.")
            clip = _ken_burns_clip(
                hero_img,
                f"outputs/story_{run_id}_shot{i}_kb.mp4",
                duration=s["duration"],
                fps=30.0,
                zoom_in=(i == 1),
            )
            if not clip:
                raise RuntimeError(
                    "Wan failed and the cinematic-zoom fallback could not "
                    "be created."
                )
            used_fallback.append(i)
        clips.append(clip)
    if used_fallback:
        log("ℹ️ " + ", ".join(f"shot {i}" for i in used_fallback)
            + " used cinematic zoom (Wan unavailable).")

    if product_img_pil is not None:
        log("🎯 Shot 3: real-product beauty close-up (uploaded product).")
        clip3 = _product_beauty(
            hero_img, product_img_pil, f"outputs/story_{run_id}_shot3.mp4",
            duration=shots[2]["duration"], fps=30.0,
        )
    else:
        log("🎞️ Shot 3: cinematic push-in on the hero (no product uploaded).")
        clip3 = _ken_burns_clip(
            hero_img, f"outputs/story_{run_id}_shot3_kb.mp4",
            duration=shots[2]["duration"], fps=30.0, zoom_in=True,
        )
    if not clip3:
        raise RuntimeError("Could not build the product-reveal clip.")
    clips.append(clip3)

    # --- Voiceover: two beats. The CTA offset is None = "end 0.2s into the
    #     card", resolved frame-accurately inside assemble_storyboard. ---
    log("🔊 Generating voiceover (edge-tts)...")
    vo_items = []
    v1 = tts_voiceover("Step out of the ordinary.",
                       f"outputs/storyboard_vo_{run_id}_1.mp3")
    if v1:
        vo_items.append((v1, 0.7))
    v2 = tts_voiceover(
        f"{brand_name} — own the moment. Find your nearest showroom.",
        f"outputs/storyboard_vo_{run_id}_2.mp3",
    )
    if v2:
        vo_items.append((v2, None))
    if vo_items:
        log(f"🔊 Voiceover: {len(vo_items)} segment(s), CTA lands on the card.")

    # --- Tagline from the real brief + product context ---
    brief_for_tag = " ".join(part for part in (brief, product_desc) if part)
    tagline = generate_tagline(brief_for_tag, profile, brand_name, cfg=cfg)

    # --- Assemble the strict-`target_duration` ad ---
    out_path = f"outputs/storyboard_{run_id}.mp4"
    log("🎬 Assembling 20s timeline (dissolves + brand card + VO)...")
    assemble_storyboard(
        clips, [s["duration"] for s in shots], brand_name, tagline, out_path,
        product_img=product_img_pil, target_duration=target_duration,
        vo_items=vo_items, log=log,
    )
    log(f"✅ Storyboard ad ready: {out_path}")
    return out_path
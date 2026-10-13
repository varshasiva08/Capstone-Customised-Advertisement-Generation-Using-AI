"""
Module 5 - Motion / Animation

Takes the final advertisement image and animates it via a FREE public
Hugging Face Space (Saravutw/WAN2.2_I2V_LIGHTNING_4-8step_custom: a
4-step distilled Wan 2.2 I2V Lightning model) driven through gradio_client.

Pipeline:
  1. Build a demographically-aware motion prompt from the user's
     profile and clothing choice — contextual motion descriptions
     derived from the brand brief rather than a fixed driving video.
  2. Submit the character image + motion prompt to the free Wan Space
     (no local VRAM and no paid credits required).
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

import requests
import yaml
from dotenv import load_dotenv
from PIL import Image
from huggingface_hub import InferenceClient

from profiler.prompt_builder import clean_scene

load_dotenv()

# Free ZeroGPU quota is per HF account, so the Wan Space retries across up to
# three accounts (HF_TOKEN_1/2/3 — each must belong to a DIFFERENT HF account)
# plus a generic HF_TOKEN before giving up. HF_TOKEN stays the primary account
# (first token) used for downloads and the optional paid Inference API path.
def _collect_hf_tokens():
    tokens = []
    for key in ("HF_TOKEN_1", "HF_TOKEN_2", "HF_TOKEN_3", "HF_TOKEN"):
        value = os.getenv(key)
        if value and value not in tokens:
            tokens.append(value)
    return tokens

HF_TOKENS = _collect_hf_tokens()
HF_TOKEN  = HF_TOKENS[0] if HF_TOKENS else None

# Short negative prompt for the fast free Wan Space (used when the config
# does not supply wan.negative_prompt).
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, distorted face, deformed hands, "
    "extra fingers, fused fingers, watermark, text, signature, static, "
    "jump cut, messy background"
)


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

def build_motion_prompt(profile: dict, clothing: str = "", motion_style: str = "sway",
                        scene: str = None) -> str:
    """
    Build a locked-down motion prompt for Wan I2V.

    Two styles:
      - "sway": gentle body sway and slight turn to show product from
                different angles. Best for jewellery, bags, accessories.
      - "walk": walks forward a few steps then stops in a natural pose.
                Best for clothing, sarees, suits.

    Args:
        profile:      Demographic profile dict
        clothing:     Clothing description
        motion_style: "sway" or "walk"
        scene:        Optional scene / background-effect text matching the
                      image being animated. When set, it replaces the
                      hardcoded "studio, seamless background" setting so the
                      motion prompt does not contradict the frame.

    Returns:
        Motion prompt string.
    """
    ethnicity  = profile.get("ethnicity", "woman")
    age        = profile.get("age", "adult")
    clothing_l = clothing.lower() if clothing else ""

    # Setting — a scene replaces the generic studio so Wan2.1 is not told
    # the subject is on a seamless backdrop when the image shows a lobby.
    scene = clean_scene(scene)
    if scene:
        setting = f"standing in {scene}, environment clearly visible"
    else:
        setting = "standing in a studio, seamless background"

    # Base identity anchor — keeps Wan2.1 from drifting
    identity = (
        f"{ethnicity} woman in her {age}, "
        f"{setting}, "
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
# Token-rotation helpers (quota is per HF account)
# ---------------------------------------------------------------------------

def _is_token_scoped(exc: Exception) -> bool:
    """True when the error is scoped to the HF account being used, so another
    account's token may succeed. Covers daily ZeroGPU quota exhaustion, auth
    failures and file-permission denials."""
    m = str(exc).lower()
    words = ("zerogpu", "quota", "unauthorized", "authentication",
             "not authorized", "401", "403", "permission denied",
             "access denied", "forbidden")
    return any(w in m for w in words)


def _short_err(exc: Exception) -> str:
    return str(exc)[:160]


# ---------------------------------------------------------------------------
# Video generation providers
# ---------------------------------------------------------------------------

def _call_wan_space(image: Image.Image, motion_prompt: str, cfg: dict) -> bytes:
    """
    Submit image + motion prompt to the FREE public Hugging Face Space
    'Saravutw/WAN2.2_I2V_LIGHTNING_4-8step_custom' via gradio_client.

    That Space runs a 4-step distilled Wan 2.2 I2V Lightning model, so a
    short clip finishes in a couple of minutes rather than the ~40-60 min
    taken by the full 14B Wan Space. Costs nothing; may sit in a short queue.

    Confirmed live API (gradio_client 1.14 and 2.x), a single synchronous call:
      /generate_video(input_image, last_image, prompt, steps, negative_prompt,
                      duration_seconds, guidance_scale, guidance_scale_2, seed,
                      randomize_seed, quality, scheduler, flow_shift,
                      frame_multiplier, safe_mode, video_component)
                      -> (generated_video, download_video, seed)

    Output resolution follows the input image (no height/width args).

    Returns:
        Raw video bytes (mp4).
    """
    try:
        import inspect
        import httpx
        from gradio_client import Client, handle_file
    except ImportError as e:
        raise RuntimeError(
            "gradio_client (and its httpx dependency) must be importable to "
            "use the free video Space. Install with: "
            f"pip install gradio_client  [import error: {e}]"
        ) from e

    wan_cfg  = cfg.get("wan", {}) or {}
    space_id = wan_cfg.get("space_id",
                           "Saravutw/WAN2.2_I2V_LIGHTNING_4-8step_custom")
    timeout  = int(wan_cfg.get("space_timeout", 900))

    steps       = int(wan_cfg.get("steps", 4))
    duration    = float(wan_cfg.get("duration_seconds", 3.5))
    guidance    = float(wan_cfg.get("guidance_scale", 1.0))
    guidance_2  = float(wan_cfg.get("guidance_scale_2", 1.0))
    quality     = int(wan_cfg.get("quality", 5))
    scheduler   = wan_cfg.get("scheduler", "UniPCMultistep")
    flow_shift  = float(wan_cfg.get("flow_shift", 3.0))
    # frame_multiplier must be an int in the Space's choices [16, 32, 64, 128]
    frame_mult  = int(wan_cfg.get("frame_multiplier", 16))
    safe_mode   = bool(wan_cfg.get("safe_mode", False))
    negative    = wan_cfg.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.convert("RGB").save(tmp.name)
        tmp_path = tmp.name

    try:
        # Free ZeroGPU quota is per HF account, so rotate across the accounts
        # in .env (HF_TOKEN_1/2/3, HF_TOKEN); a final anonymous attempt is
        # appended as a last resort. EVERY account is tried before giving up,
        # so a single bad/expired account (or a quota hiccup) never kills the
        # run — the caller degrades to a cinematic zoom only if all fail.
        attempts = list(HF_TOKENS) if HF_TOKENS else []
        if attempts and None not in attempts:
            attempts.append(None)

        last_err = None
        for idx, token in enumerate(attempts):
            who = "anonymous" if not token else f"token {idx + 1}"
            try:
                print(f"[motion] Attempt {idx + 1}/{len(attempts)} ({who}) -> Space '{space_id}'...")
                # gradio_client 1.x forwards httpx_kwargs to every internal httpx
                # call (config fetch, file upload, SSE upload/download). 2.x keeps
                # the same option. The free Space can sit in a queue and the
                # result download can be slow, so use a generous connect + read
                # timeout instead of the library's short defaults (which caused
                # SSL handshake timeouts on download).
                token_kwargs = {}
                if token:
                    # gradio_client <2.x used `hf_token`; 2.x renamed it to `token`.
                    params = inspect.signature(Client.__init__).parameters
                    token_kwargs["token" if "token" in params else "hf_token"] = token
                client = Client(
                    space_id,
                    verbose=False,
                    # Return the result URL instead of letting gradio_client
                    # perform its own short-timeout download. _read_video()
                    # fetches it with a generous connect/read timeout, which
                    # avoids the SSL handshake timeouts seen on the free
                    # Space's file endpoint.
                    download_files=False,
                    httpx_kwargs={
                        "timeout": httpx.Timeout(
                            connect=60,
                            read=float(timeout),   # wait while the Space queues/runs
                            write=120,
                            pool=60,
                        )
                    },
                    **token_kwargs,
                )

                print(
                    f"[motion] Generating {duration:g}s clip "
                    f"({steps} steps, {scheduler}, free Space)..."
                )
                generated, _download, seed = client.predict(
                    handle_file(tmp_path),   # input_image
                    None,                    # last_image (no explicit end frame)
                    motion_prompt,           # prompt
                    steps,                   # steps
                    negative,                # negative_prompt
                    duration,                # duration_seconds
                    guidance,                # guidance_scale
                    guidance_2,              # guidance_scale_2
                    0,                       # seed (>= 0; ignored when randomize_seed=True)
                    True,                    # randomize_seed
                    quality,                 # quality
                    scheduler,               # scheduler
                    flow_shift,              # flow_shift
                    frame_mult,              # frame_multiplier (int)
                    safe_mode,               # safe_mode
                    True,                    # video_component
                    api_name="/generate_video",
                )
                print(f"[motion] Space returned clip (seed={seed}).")

                result = _extract_video(generated)
                if not result:
                    raise RuntimeError(
                        f"Wan Space returned no video (got {type(generated).__name__})."
                    )
                return _read_video(result, token=token)
            except Exception as e:
                # Try EVERY configured HF account, then give up. A transient
                # hiccup on one account (or a token-scoped quota/auth error)
                # must not kill the run: the caller falls back to a cinematic
                # zoom only after all accounts have genuinely failed.
                last_err = e
                print(
                    f"[motion] {who} failed ({_short_err(e)}) — "
                    "trying the next account..."
                )
        raise RuntimeError(
            f"All {len(attempts)} HF account(s) exhausted for the free Wan "
            f"Space. Last error: {_short_err(last_err)}. "
            "Quota resets daily; or add fresh HF_TOKEN_1/2/3 (one per HF "
            "account) to .env."
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _extract_video(result):
    """Pull the video reference out of a Space result.

    Handles every gradio_client return shape:
      - 1.x dicts: {"video": <ref>, ...} or FileData {"path", "url"}
      - 2.x FileData objects: .data holds the URL (download_files=False)
        or a local path; .name is just the file name

    Prefers an http(s) URL and only uses a local path when the file
    actually exists (with download_files=False the Space's own
    server-side path is NOT usable client-side).
    """
    if not result:
        return None
    first = result[0] if isinstance(result, (list, tuple)) else result
    if first is None:
        return None
    if isinstance(first, str):
        candidates = (first,)
    else:
        wrapped = _file_attr(first, "video")
        obj = wrapped if wrapped is not None else first
        candidates = (
            _file_attr(obj, "url"),
            _file_attr(obj, "data"),
            _file_attr(obj, "path"),
            _file_attr(obj, "name"),
        )
    for c in candidates:
        if isinstance(c, str) and c.startswith("http"):
            return c
        if isinstance(c, str) and os.path.exists(c):
            return c
    return None


def _file_attr(value, name, default=None):
    """Read a file attribute that may be a dict key (gradio_client 1.x) or
    an object attribute (gradio_client 2.x FileData)."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _read_video(video, token=None) -> bytes:
    """Read a local video path or download an http(s) video URL.

    token: the HF account used for the generation — its auth may be required
    to fetch the result file. Downloads without headers when token is None.
    """
    if isinstance(video, str) and video.startswith("http"):
        headers = {"Authorization": f"Bearer {token}"} if token else None
        # Generous connect timeout: the free Space's file endpoint can be slow
        # to complete the TLS handshake right after a generation.
        resp = requests.get(video, headers=headers, timeout=(60, 300))
        if resp.status_code in (401, 403) and headers:
            resp = requests.get(video, timeout=(60, 300))
        resp.raise_for_status()
        return resp.content
    with open(video, "rb") as f:
        return f.read()


def _call_wan_hf(image: Image.Image, motion_prompt: str, cfg: dict) -> bytes:
    """
    Submit image + motion prompt to Wan2.1-I2V-14B-720P via the HuggingFace
    Inference API (fal-ai provider). Requires paid HF Inference credits.

    Args:
        image:         PIL Image - the final advertisement image
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
    provider  = wan_cfg.get("hf_provider", "fal-ai")

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


def _call_wan_api(image: Image.Image, motion_prompt: str, cfg: dict) -> bytes:
    """Dispatch to the configured video provider (wan.provider)."""
    provider = (cfg.get("wan", {}) or {}).get("provider", "space").strip().lower()
    if provider in ("space", "hf-space", "gradio"):
        return _call_wan_space(image, motion_prompt, cfg)
    if provider in ("fal-ai", "hf"):
        return _call_wan_hf(image, motion_prompt, cfg)
    raise RuntimeError(
        f"Unknown wan.provider '{provider}' (expected 'space' or 'fal-ai')"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def animate_image(image: Image.Image, config: dict = None,
                  profile: dict = None, clothing: str = "",
                  motion_style: str = "sway", scene: str = None) -> str:
    """
    Animate the final advertisement image into a short video.

    Args:
        image:        The final advertisement image (PIL Image).
        config:       Loaded config.yaml dict.
        profile:      Demographic profile dict.
        clothing:     Clothing description from the user's brief.
        motion_style: "sway" (gentle turn, best for accessories/jewellery)
                      or "walk" (walks forward then poses, best for clothing).
        scene:        Optional scene / background-effect text used to build
                      the image, forwarded to build_motion_prompt() so the
                      motion prompt matches the frame.

    Returns:
        File path to the generated video, saved under outputs/.
    """
    cfg = _load_config(config)

    motion_prompt = build_motion_prompt(
        profile or {}, clothing or "", motion_style, scene=scene
    )
    print(f"[motion] Style: {motion_style} | Prompt: {motion_prompt}")

    video_bytes = _call_wan_api(image, motion_prompt, cfg)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"animation_{int(time.time())}.mp4")
    with open(out_path, "wb") as f:
        f.write(video_bytes)

    print(f"[motion] Saved -> {out_path}")
    return out_path
import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

load_dotenv()

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

HF_TOKENS = [t for t in (
    os.getenv("HF_TOKEN_1"),
    os.getenv("HF_TOKEN_2"),
    os.getenv("HF_TOKEN_3"),
) if t]

# fallback for when there's no Streamlit session (e.g. quick_test.py)
_fallback_token_idx = 0


def _get_token_idx():
    try:
        if "hf_token_idx" not in st.session_state:
            st.session_state.hf_token_idx = 0
        return st.session_state.hf_token_idx
    except Exception:
        return _fallback_token_idx


def _set_token_idx(idx):
    global _fallback_token_idx
    try:
        st.session_state.hf_token_idx = idx
    except Exception:
        _fallback_token_idx = idx


def generate_image(prompt: str, seed: int, config: dict):
    """Generate one image via the HF Inference API, rotating across
    multiple personal accounts if one is rate-limited, out of credits,
    or hasn't accepted the model's terms yet."""
    if not HF_TOKENS:
        raise RuntimeError("No HF_TOKEN_1/2/3 found in environment (.env)")

    errors = []
    start = _get_token_idx()
    for offset in range(len(HF_TOKENS)):
        idx = (start + offset) % len(HF_TOKENS)
        client = InferenceClient(model=MODEL_ID, token=HF_TOKENS[idx], timeout=300)
        try:
            img = client.text_to_image(prompt, seed=seed)
            _set_token_idx(idx)  # remember the working one
            return img, "hf-api", (1024, 1024), 4
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            errors.append(f"account {idx + 1} -> {status}")
            if status not in (429, 402, 403):
                raise  # don't mask unrelated errors (network, 500, etc.)

    raise RuntimeError(f"All {len(HF_TOKENS)} HF accounts failed: {', '.join(errors)}")

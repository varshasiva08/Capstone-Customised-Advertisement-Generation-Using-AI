"""
Parses a free-text brand/ad brief into a demographic profile
(ethnicity, body_type, age).

Ollama (phi3:mini) is REQUIRED — it is the primary parser because it
handles informal, ambiguous language that keyword rules cannot catch
(e.g. "fat aunty", "thirty-something", "dusky Indian woman in her 40s").

The rule-based keyword parser still runs first and its results are merged
in, but Ollama's output wins on any conflict because it is the more
flexible and accurate parser.

If Ollama is not running, the app raises an informative error at startup
rather than silently falling back to keyword-only parsing.
"""

import json
import re
import requests


# ---------- Availability check ----------

def check_ollama(host: str = "http://localhost:11434") -> None:
    """
    Verify Ollama is reachable. Raises RuntimeError if not.
    Called once at app startup.
    """
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        if not r.ok:
            raise RuntimeError(
                f"Ollama responded with status {r.status_code}. "
                "Make sure `ollama serve` is running."
            )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Ollama at " + host + ".\n"
            "Start it with: ollama serve\n"
            "Then in a separate terminal: ollama pull phi3:mini"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama timed out. Is it still starting up? Wait a moment and retry."
        )


# ---------- Ollama LLM parser (primary) ----------

def parse_with_ollama(text: str, cfg: dict, required_fields: list,
                      model: str = "phi3:mini",
                      host: str = "http://localhost:11434") -> dict:
    """
    Use phi3:mini via Ollama to extract demographic attributes from
    a free-text ad brief.

    Returns a dict with any of {ethnicity, body_type, age} it found.
    Keys with no clear match are omitted (not included as None/null).
    """
    options = {k: cfg["options"][k] for k in required_fields}
    prompt = (
        "Extract demographic attributes for an advertisement model from this brief. "
        f"Allowed values: {json.dumps(options)}. "
        'Reply ONLY with compact JSON like {"ethnicity": "...", "body_type": "...", "age": "..."}. '
        f'Use null for anything not clearly mentioned.\nBrief: "{text}"'
    )
    r = requests.post(
        f"{host}/api/generate",
        timeout=60,
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
    )
    r.raise_for_status()
    data = json.loads(r.json()["response"])
    # Only keep fields that are valid allowed values
    return {
        k: v for k, v in data.items()
        if k in required_fields and v in cfg["options"].get(k, [])
    }


# ---------- Keyword / regex parser (secondary, merges with Ollama) ----------

def parse_with_rules(text: str, cfg: dict) -> dict:
    """
    Fast keyword/regex matching against config.yaml's `keywords` section.
    Used as a first pass — its results are merged with Ollama's output,
    with Ollama winning on conflicts.
    """
    found, low = {}, text.lower()
    for field, mapping in cfg["keywords"].items():
        for value, kws in mapping.items():
            if any(re.search(rf"\b{re.escape(k)}\b", low) for k in kws):
                found[field] = value
                break
    return found


# ---------- Combined parse (called by app.py) ----------

def parse_brief(text: str, current: dict, cfg: dict, required_fields: list,
                model: str = "phi3:mini",
                host: str = "http://localhost:11434") -> dict:
    """
    Parse a user's brief and merge results into the current profile.

    Strategy:
        1. Rule-based keyword matching (fast, deterministic baseline)
        2. Ollama LLM parsing (smarter, wins on conflicts)
        3. Merge both with existing profile — previously confirmed fields
           are never overwritten.

    Args:
        text:            The user's raw input text.
        current:         Existing profile dict (from previous chat turns).
        cfg:             Loaded config.yaml dict.
        required_fields: List of field names to extract.
        model:           Ollama model name.
        host:            Ollama host URL.

    Returns:
        Updated profile dict.
    """
    rule_parsed   = parse_with_rules(text, cfg)
    ollama_parsed = parse_with_ollama(text, cfg, required_fields, model, host)

    # Ollama wins over rules; both merged on top of existing confirmed profile
    merged = {**rule_parsed, **ollama_parsed}
    return {**current, **merged}

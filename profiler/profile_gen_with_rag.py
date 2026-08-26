"""
profile_gen.py — Updated with RAG context
==========================================
Replace your existing adfidelity/profiler/profile_gen.py with this file.

The only change from the original:
  - Accepts an optional `rag_context` string
  - If provided, prepends it to the LLM prompt so it has historical context
  - Everything else is identical to your original
"""

import json
import requests

SYSTEM_PROMPT = """You are a demographic profile generator for advertisement campaigns.
Given a brand brief, generate exactly 6 diverse demographic profiles as a JSON array.
Each profile must have these exact fields:
- ethnicity: one of [South Asian, East Asian, African American]
- body_type: one of [slim, medium, plus-size]
- age: one of [20s, 30s, 40s, 50s]
- skin_tone_description: brief descriptor
- style_notes: 1 sentence about styling

Cover diverse combinations. Reply ONLY with a JSON array, no extra text."""


def generate_profiles(brand_brief: str,
                      model: str = "phi3:mini",
                      host: str = "http://localhost:11434",
                      rag_context: str = "") -> list[dict]:
    """
    Generate demographic profiles for a brand brief.

    Args:
        brand_brief:  The raw brand brief text.
        model:        Ollama model to use.
        host:         Ollama host URL.
        rag_context:  Optional context string from BriefRAG.retrieve().
                      If provided, the LLM uses similar past campaigns
                      as reference when generating profiles.

    Returns:
        List of profile dicts.
    """
    # Build the full prompt — prepend RAG context if available
    if rag_context:
        prompt = f"{SYSTEM_PROMPT}\n\n{rag_context}Brand brief: {brand_brief}"
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nBrand brief: {brand_brief}"

    r = requests.post(
        f"{host}/api/generate",
        timeout=300,
        json={
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
    )
    r.raise_for_status()
    raw = r.json()["response"].strip()

    # Strip markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    profiles = json.loads(raw.strip())
    return profiles if isinstance(profiles, list) else [profiles]

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
                      host: str = "http://localhost:11434") -> list[dict]:
    prompt = f"{SYSTEM_PROMPT}\n\nBrand brief: {brand_brief}"

    r = requests.post(
        f"{host}/api/generate",
        timeout=300,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",   # forces JSON output mode in Ollama
        },
    )
    r.raise_for_status()
    raw = r.json()["response"].strip()

    # Strip markdown fences if the model adds them despite format=json
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    profiles = json.loads(raw.strip())

    # Ollama's json mode sometimes returns {"profiles": [...]} instead of [...]
    if isinstance(profiles, dict):
        profiles = next(v for v in profiles.values() if isinstance(v, list))

    return profiles


if __name__ == "__main__":
    brief = "sportswear brand targeting diverse women across South Asia"
    profiles = generate_profiles(brief)
    for i, p in enumerate(profiles, 1):
        print(f"\nProfile {i}:")
        print(json.dumps(p, indent=2))

"""
Profile suggestions seeded by similar past campaigns (RAG), deterministic.

The retrieval store (rag/brief_rag.py) is unchanged: briefs are embedded with a
small sentence-transformer (an embedding model, not a generative LLM). Past
profiles that scored well for similar briefs are offered first; the remaining
slots are filled by the coverage-maximising sampler in profile_gen.py.
"""
from __future__ import annotations

from profiler.profile_gen import generate_profiles as _stratified
from profiler.text_parser import parse_brief_detailed


def generate_profiles(brand_brief: str, n: int = 6, cfg: dict = None, rag=None,
                      seed: int = 0, **_ignored) -> list:
    """`rag` is an optional rag.brief_rag.BriefRAG; without it this equals profile_gen."""
    picked = []
    if rag is not None:
        try:
            for prof in rag.retrieve_profiles(brand_brief):
                if prof not in picked:
                    picked.append(prof)
        except Exception:                       # noqa: BLE001 - retrieval is best-effort
            picked = []
    held = parse_brief_detailed(brand_brief, cfg).profile()
    picked = [{**p, **held} for p in picked][:n]
    rest = [p for p in _stratified(brand_brief, n, cfg, seed) if p not in picked]
    return (picked + rest)[:n]

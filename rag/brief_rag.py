"""
RAG Module — Brand Brief Context Retrieval
==========================================
Retrieves similar past brand briefs and their successful demographic profiles
from a local ChromaDB vector store, and uses them as context when generating
a new profile via the LLM.

How it fits into the pipeline:
    Brand Brief (new)
        ↓
    [brief_rag.py] ← retrieves top-3 similar past briefs + their profiles
        ↓
    [profile_gen.py] — now generates profile WITH historical context
        ↓
    [prompt_builder.py] → FLUX → image

Why this matters:
    Without RAG, every brief is treated fresh. The LLM has no memory of what
    demographic profiles worked well for similar past campaigns. With RAG,
    the system learns from history — if "ethnic wear for South Asian women"
    previously scored high fidelity, new similar briefs benefit from that.

Setup:
    pip install chromadb sentence-transformers

Where to put this file:
    adfidelity/rag/brief_rag.py

Usage:
    from rag.brief_rag import BriefRAG

    rag = BriefRAG()

    # After a successful run, store the brief + profile + fidelity score
    rag.store(
        brief="Traditional silk saree brand for South Indian women in their 30s",
        profile={"ethnicity": "South Asian", "body_type": "medium", "age": "30s"},
        fidelity_score=8.7
    )

    # Before generating a new profile, retrieve similar past context
    context = rag.retrieve("saree brand for Indian women")
    # context is a string you pass to profile_gen.py
"""

import json
import os

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────

DB_PATH        = os.path.join(os.path.dirname(__file__), "..", "rag_store")
COLLECTION     = "brief_history"
EMBED_MODEL    = "all-MiniLM-L6-v2"   # small, fast, no GPU needed (~80MB)
TOP_K          = 3                     # how many past briefs to retrieve
MIN_FIDELITY   = 5.0                   # only store runs that scored above this

# ── Core class ───────────────────────────────────────────────────────────────

class BriefRAG:
    """
    Stores past brand briefs + demographic profiles + fidelity scores
    in a local ChromaDB vector store, and retrieves similar ones as
    context for new brief processing.
    """

    def __init__(self):
        self._embedder  = SentenceTransformer(EMBED_MODEL)
        self._client    = chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

    # ── Write ────────────────────────────────────────────────────────────────

    def store(self, brief: str, profile: dict, fidelity_score: float) -> None:
        """
        Store a brief + its profile + fidelity score after a successful run.
        Only stores runs above MIN_FIDELITY to keep the store clean.

        Args:
            brief:          The raw brand brief text the user typed.
            profile:        The demographic profile dict that was generated.
            fidelity_score: Overall fidelity score (0–10) from CLIP or LLaVA.
        """
        if fidelity_score < MIN_FIDELITY:
            return   # don't pollute the store with low-quality runs

        embedding = self._embedder.encode(brief).tolist()
        doc_id    = f"brief_{self._col.count() + 1}"

        self._col.add(
            ids        = [doc_id],
            embeddings = [embedding],
            documents  = [brief],
            metadatas  = [{
                "profile":         json.dumps(profile),
                "fidelity_score":  fidelity_score,
                "ethnicity":       profile.get("ethnicity", ""),
                "body_type":       profile.get("body_type", ""),
                "age":             profile.get("age", ""),
            }]
        )

    # ── Read ─────────────────────────────────────────────────────────────────

    def retrieve(self, brief: str) -> str:
        """
        Given a new brand brief, retrieve the top-K most similar past briefs
        and return them as a formatted context string for the LLM prompt.

        Returns empty string if the store has fewer than 2 entries
        (not enough history to be useful).

        Args:
            brief: The raw brand brief text.

        Returns:
            A context string to prepend to the LLM profile-generation prompt.
        """
        if self._col.count() < 2:
            return ""   # store is too empty to be useful yet

        embedding = self._embedder.encode(brief).tolist()
        results   = self._col.query(
            query_embeddings = [embedding],
            n_results        = min(TOP_K, self._col.count()),
            include          = ["documents", "metadatas", "distances"]
        )

        if not results["documents"][0]:
            return ""

        lines = ["=== SIMILAR PAST CAMPAIGNS (use as reference) ==="]
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = round((1 - dist) * 100, 1)
            profile    = json.loads(meta["profile"])
            score      = meta["fidelity_score"]

            lines.append(
                f"\nPast brief ({similarity}% similar, fidelity score {score}/10):\n"
                f"  Brief:   {doc}\n"
                f"  Profile: ethnicity={profile.get('ethnicity')}, "
                f"body_type={profile.get('body_type')}, age={profile.get('age')}"
            )

        lines.append("\n=== Use the above as context. Generate a profile for the NEW brief below. ===\n")
        return "\n".join(lines)

    # ── Info ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Returns total number of briefs stored."""
        return self._col.count()

    def clear(self) -> None:
        """Wipe the store. Useful for testing."""
        self._client.delete_collection(COLLECTION)
        self._col = self._client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rag = BriefRAG()

    print(f"Store currently has {rag.count()} entries.\n")

    # Seed with a few dummy entries
    test_data = [
        {
            "brief":   "Traditional silk saree brand for South Indian women in their 30s",
            "profile": {"ethnicity": "South Asian", "body_type": "medium", "age": "30s"},
            "score":   8.7
        },
        {
            "brief":   "Corporate fashion brand targeting working women in their 40s",
            "profile": {"ethnicity": "South Asian", "body_type": "slim", "age": "40s"},
            "score":   7.9
        },
        {
            "brief":   "Casual streetwear for young South Asian women in their 20s",
            "profile": {"ethnicity": "South Asian", "body_type": "slim", "age": "20s"},
            "score":   8.2
        },
    ]

    print("Storing test entries...")
    for d in test_data:
        rag.store(d["brief"], d["profile"], d["score"])
    print(f"Store now has {rag.count()} entries.\n")

    # Test retrieval
    new_brief = "Ethnic wear for Indian women in their 30s — soft cotton kurta brand"
    print(f"Retrieving context for:\n  '{new_brief}'\n")
    context = rag.retrieve(new_brief)
    print(context if context else "No context retrieved (store may be empty).")

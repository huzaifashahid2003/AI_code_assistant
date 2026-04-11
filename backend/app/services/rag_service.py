"""
rag_service.py — Embedding-based retrieval for code chunks.

Pipeline
--------
1. Bi-encoder (all-MiniLM-L6-v2, ~90 MB): each code chunk is encoded into a
   384-dimensional L2-normalised vector once per file.  The index is cached in
   memory and only rebuilt when the file changes (mtime-based invalidation).
2. Coarse retrieval: top-(k × 3) candidates by cosine similarity (dot product).
3. Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2, ~70 MB):
   reorders candidates by fine-grained relevance.  Falls back silently to the
   bi-encoder ranking when the reranker is unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_EMBED_MODEL = "all-MiniLM-L6-v2"
_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Model helpers (loaded once, cached for the lifetime of the process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_encoder() -> SentenceTransformer:
    """Load and cache the bi-encoder (downloaded once, ~90 MB)."""
    logger.info("Loading bi-encoder model '%s'.", _EMBED_MODEL)
    return SentenceTransformer(_EMBED_MODEL)


@lru_cache(maxsize=1)
def _get_reranker() -> Optional[object]:
    """Load and cache the cross-encoder for reranking (optional, ~70 MB).

    Returns None if the model is unavailable so callers degrade gracefully.
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[attr-defined]
        logger.info("Loading cross-encoder reranker '%s'.", _RERANK_MODEL)
        return CrossEncoder(_RERANK_MODEL)
    except Exception as exc:
        logger.warning("Cross-encoder reranker unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

@dataclass
class VectorStore:
    """L2-normalised dense embeddings alongside the original chunk strings."""

    embeddings: np.ndarray   # shape (n_chunks, 384)
    chunks: List[str]


# ---------------------------------------------------------------------------
# In-memory index cache  —  filename → (mtime, VectorStore)
# ---------------------------------------------------------------------------

_index_cache: Dict[str, Tuple[float, VectorStore]] = {}


def get_or_build_index(filename: str, mtime: float, chunks: List[str]) -> VectorStore:
    """Return a cached VectorStore for *filename* or build and cache a fresh one.

    The cache entry is invalidated whenever *mtime* changes, so a re-uploaded
    file always gets a freshly encoded index.
    """
    cached = _index_cache.get(filename)
    if cached is not None and cached[0] == mtime:
        logger.debug("Embedding cache hit for '%s' (mtime=%.3f).", filename, mtime)
        return cached[1]

    logger.info(
        "Building embedding index for '%s' (%d chunk(s)).", filename, len(chunks)
    )
    store = _build_index(chunks)
    _index_cache[filename] = (mtime, store)
    return store


def _build_index(chunks: List[str]) -> VectorStore:
    """Encode *chunks* with the bi-encoder and return a VectorStore."""
    if not chunks:
        raise ValueError("Cannot create a vector store from an empty chunk list.")
    encoder = _get_encoder()
    embeddings: np.ndarray = encoder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalise → cosine = dot product
        show_progress_bar=False,
    )
    return VectorStore(embeddings=embeddings, chunks=chunks)


# ---------------------------------------------------------------------------
# Retrieval + reranking
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(
    store: VectorStore,
    query: str,
    k: int = 5,
    rerank: bool = True,
) -> List[int]:
    """Return indices of the top-k most relevant chunks for *query*.

    Steps:
      1. Bi-encoder coarse pass: score all chunks by cosine similarity,
         keep top min(k*3, n) candidates.
      2. Cross-encoder reranking: reorder candidates by a fine-grained
         relevance score.  Skipped when the reranker is unavailable.

    Returns plain Python ints for direct indexing into parallel lists.
    """
    n = len(store.chunks)
    k = min(k, n)

    # ── 1. Coarse bi-encoder retrieval ────────────────────────────────────
    encoder = _get_encoder()
    query_emb: np.ndarray = encoder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = store.embeddings.dot(query_emb[0])   # (n_chunks,) cosine similarities

    candidate_k = min(k * 3, n)
    candidate_indices: List[int] = np.argsort(scores)[::-1][:candidate_k].tolist()

    # ── 2. Cross-encoder reranking (optional) ─────────────────────────────
    if rerank and len(candidate_indices) > k:
        reranker = _get_reranker()
        if reranker is not None:
            pairs = [(query, store.chunks[idx]) for idx in candidate_indices]
            rerank_scores = reranker.predict(pairs)
            reranked = sorted(
                zip(candidate_indices, rerank_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return [int(idx) for idx, _ in reranked[:k]]

    return [int(i) for i in candidate_indices[:k]]

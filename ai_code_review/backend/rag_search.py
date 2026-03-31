"""
rag_search.py — FAISS vector store + LLM query pipeline (stub).

This module will:
    1. Embed code chunks using a sentence-transformer model.
    2. Store/load embeddings in a FAISS index.
    3. Retrieve the most relevant chunks for a user query.
    4. Pass retrieved chunks + query to an LLM for a final review response.

Currently contains typed stubs so the rest of the project can import from
this module without errors.  Fill in each TODO section as the project grows.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party imports (installed lazily to avoid hard failures at startup)
# ---------------------------------------------------------------------------
# from sentence_transformers import SentenceTransformer  # TODO: uncomment
# import faiss                                            # TODO: uncomment
# import numpy as np                                     # TODO: uncomment

from code_processing import CodeChunk


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "faiss.index"
TOP_K = 5  # number of nearest neighbours to retrieve


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_chunks(chunks: List[CodeChunk]) -> "np.ndarray":  # type: ignore[name-defined]
    """
    Convert a list of CodeChunks into a 2-D numpy array of embeddings.

    Args:
        chunks: Code chunks produced by ``chunk_code``.

    Returns:
        Float32 numpy array of shape ``(len(chunks), embedding_dim)``.
    """
    # TODO: initialise SentenceTransformer and encode chunk sources
    # model = SentenceTransformer(EMBED_MODEL_NAME)
    # texts = [c.source for c in chunks]
    # return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    raise NotImplementedError("embed_chunks — install sentence-transformers first.")


# ---------------------------------------------------------------------------
# FAISS index management
# ---------------------------------------------------------------------------

def build_index(embeddings: "np.ndarray") -> "faiss.IndexFlatL2":  # type: ignore[name-defined]
    """
    Build a flat L2 FAISS index from a matrix of embeddings.

    Args:
        embeddings: 2-D float32 array (n_chunks × embedding_dim).

    Returns:
        A populated ``faiss.IndexFlatL2`` instance.
    """
    # TODO:
    # import faiss, numpy as np
    # dim = embeddings.shape[1]
    # index = faiss.IndexFlatL2(dim)
    # index.add(embeddings.astype(np.float32))
    # return index
    raise NotImplementedError("build_index — install faiss-cpu first.")


def save_index(index: "faiss.Index", path: Path = INDEX_PATH) -> None:  # type: ignore[name-defined]
    """Persist a FAISS index to disk."""
    # TODO: faiss.write_index(index, str(path))
    raise NotImplementedError("save_index — install faiss-cpu first.")


def load_index(path: Path = INDEX_PATH) -> "faiss.Index":  # type: ignore[name-defined]
    """Load a previously saved FAISS index from disk."""
    # TODO: return faiss.read_index(str(path))
    raise NotImplementedError("load_index — install faiss-cpu first.")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def search(
    query: str,
    index: "faiss.Index",  # type: ignore[name-defined]
    chunks: List[CodeChunk],
    top_k: int = TOP_K,
) -> List[Tuple[CodeChunk, float]]:
    """
    Retrieve the *top_k* most relevant code chunks for a natural-language query.

    Args:
        query:  User's question or review request.
        index:  Populated FAISS index aligned with ``chunks``.
        chunks: Original list of CodeChunks (same order as index rows).
        top_k:  How many results to return.

    Returns:
        List of ``(CodeChunk, distance)`` pairs, closest first.
    """
    # TODO:
    # from sentence_transformers import SentenceTransformer
    # import numpy as np
    # model = SentenceTransformer(EMBED_MODEL_NAME)
    # q_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    # distances, indices = index.search(q_vec, top_k)
    # return [(chunks[i], float(distances[0][j])) for j, i in enumerate(indices[0]) if i != -1]
    raise NotImplementedError("search — install faiss-cpu and sentence-transformers first.")


# ---------------------------------------------------------------------------
# LLM review
# ---------------------------------------------------------------------------

def review_with_llm(query: str, context_chunks: List[CodeChunk], model: str = "gpt-4o-mini") -> str:
    """
    Send retrieved code chunks + user query to an LLM and return review text.

    Args:
        query:          The user's review question.
        context_chunks: Relevant code chunks retrieved from FAISS.
        model:          OpenAI model identifier.

    Returns:
        LLM response string.
    """
    # TODO:
    # from openai import OpenAI
    # client = OpenAI()
    # context = "\n\n---\n\n".join(c.source for c in context_chunks)
    # messages = [
    #     {"role": "system", "content": "You are an expert Python code reviewer."},
    #     {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
    # ]
    # response = client.chat.completions.create(model=model, messages=messages)
    # return response.choices[0].message.content
    raise NotImplementedError("review_with_llm — set OPENAI_API_KEY and install openai first.")

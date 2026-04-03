"""
rag_search.py — FAISS vector store + Google Gemini LLM review pipeline.

Responsibilities:
    1. Embed code-chunk strings using Google Gemini (embedding-001).
    2. Build and query a FAISS flat-L2 index for similarity search.
    3. Generate a structured code review via Google Gemini (gemini-1.5-flash).

Fallback behaviour (no API key / package not installed):
    - Embeddings: deterministic hash-based placeholder (reproducible, not semantic).
    - Review:     structured placeholder listing detected chunks.

This keeps the full pipeline runnable end-to-end without any credentials, while
making it trivial to enable real AI by setting the GEMINI_API_KEY env variable.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional third-party imports
# ---------------------------------------------------------------------------

try:
    import google.generativeai as genai  # type: ignore[import]
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

try:
    import faiss  # type: ignore[import]
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMINI_EMBED_MODEL = "models/embedding-001"
GEMINI_REVIEW_MODEL = "gemini-1.5-flash"

# Gemini embedding-001 always returns 768-dimensional vectors.
GEMINI_EMBED_DIM = 768
# Dimension used by the deterministic placeholder embedder.
PLACEHOLDER_EMBED_DIM = 128

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FAISSStore:
    """
    Bundles a FAISS index with the original chunk texts and embedding dimension.

    Attributes:
        index:  The populated ``faiss.IndexFlatL2`` instance.
        chunks: Source strings in the same row order as the index.
        dim:    Embedding dimension (needed to embed queries consistently).
    """

    index: object       # faiss.IndexFlatL2
    chunks: List[str]
    dim: int


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _embed_with_gemini(texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings using Google Gemini embedding-001.

    Args:
        texts: Strings to embed (typically code chunk sources).

    Returns:
        Float32 numpy array of shape ``(len(texts), 768)``.

    Raises:
        EnvironmentError: If ``GEMINI_API_KEY`` is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it or remove google-generativeai to use the placeholder embedder."
        )
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model=GEMINI_EMBED_MODEL,
        content=texts,
        task_type="retrieval_document",
    )
    # result["embedding"] is a list[list[float]] for multiple texts
    return np.array(result["embedding"], dtype=np.float32)


def _embed_placeholder(texts: List[str], dim: int = PLACEHOLDER_EMBED_DIM) -> np.ndarray:
    """
    Deterministic hash-based fallback embedder.

    Produces the same L2-normalised vector for identical text, making the
    pipeline fully testable without credentials.

    Args:
        texts: Strings to embed.
        dim:   Output embedding dimension.

    Returns:
        Float32 numpy array of shape ``(len(texts), dim)``.
    """
    vectors: List[np.ndarray] = []
    for text in texts:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        vectors.append(vec / norm if norm > 0.0 else vec)
    return np.array(vectors, dtype=np.float32)


def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed strings, preferring Gemini when available; falls back to placeholder.

    Args:
        texts: Strings to embed.

    Returns:
        Float32 embedding matrix of shape ``(len(texts), embed_dim)``.
    """
    if _GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
        return _embed_with_gemini(texts)
    return _embed_placeholder(texts)


def _embed_query(query: str, expected_dim: int) -> np.ndarray:
    """
    Embed a single query string into a (1, dim) matrix for FAISS search.

    Uses ``retrieval_query`` task type for Gemini so the vector is optimised
    for similarity lookup rather than document storage.

    Args:
        query:        The user's query or review question.
        expected_dim: Embedding dim of the FAISS index (must match).

    Returns:
        Float32 array of shape ``(1, expected_dim)``.
    """
    if _GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        result = genai.embed_content(
            model=GEMINI_EMBED_MODEL,
            content=query,
            task_type="retrieval_query",
        )
        return np.array(result["embedding"], dtype=np.float32).reshape(1, -1)
    return _embed_placeholder([query], dim=expected_dim)


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------


def create_faiss_index(chunks: List[str]) -> FAISSStore:
    """
    Embed each chunk and store the result in a FAISS flat-L2 index.

    Args:
        chunks: Source-code strings to embed and index.

    Returns:
        A :class:`FAISSStore` containing the FAISS index, the original chunks,
        and the embedding dimension.

    Raises:
        ImportError: If ``faiss-cpu`` is not installed.
        ValueError:  If ``chunks`` is empty.
    """
    if not _FAISS_AVAILABLE:
        raise ImportError(
            "faiss-cpu is required. Install it with: pip install faiss-cpu"
        )
    if not chunks:
        raise ValueError("Cannot create a FAISS index from an empty chunk list.")

    embeddings = _embed_texts(chunks)       # shape: (n, dim)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return FAISSStore(index=index, chunks=chunks, dim=dim)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve_relevant_chunks(
    store: FAISSStore,
    query: str,
    k: int = 3,
) -> List[str]:
    """
    Retrieve the *k* most relevant code chunks for a natural-language query.

    Args:
        store: A :class:`FAISSStore` built by :func:`create_faiss_index`.
        query: User's question or free-text review request.
        k:     Number of top results to return.  Capped to ``len(store.chunks)``.

    Returns:
        List of chunk source strings ordered from most to least similar.
    """
    k = min(k, len(store.chunks))
    query_vec = _embed_query(query, store.dim)       # shape: (1, dim)
    _, indices = store.index.search(query_vec, k)    # indices shape: (1, k)
    return [store.chunks[i] for i in indices[0] if i != -1]


# ---------------------------------------------------------------------------
# Code review prompt template
# ---------------------------------------------------------------------------

_REVIEW_PROMPT = """\
You are an expert Python code reviewer conducting a professional code review.
Analyse the following Python code chunks and produce a structured report.

For each issue, reference the exact function or class name and the approximate
line numbers provided in the chunk headers (e.g. "In `divide()` (Lines 5-7):").

Structure your entire response using EXACTLY these four sections in this order.
Only include a section if you have findings for it — otherwise omit it.

---

## 1. Code Quality Issues
List issues related to readability, PEP 8 violations, overly complex logic,
missing docstrings, or poor code organisation.

## 2. Performance Improvements
Highlight inefficiencies, unnecessary computations, suboptimal data structures,
or operations that could be vectorised or cached.

## 3. Naming & Style Suggestions
Point out unclear variable names, non-`snake_case` identifiers, magic numbers,
or style inconsistencies that reduce readability.

## 4. Potential Bugs
Identify logic errors, unhandled edge cases (e.g. division by zero, empty
inputs, off-by-one), missing input validation, or other likely runtime failures.

---

For every finding use this format:
- **`FunctionOrClassName` (Lines X-Y):** Brief description of the issue.
  *Suggestion:* Concrete, actionable fix.

If the code is clean in a category, write: *(No issues found.)*

--- CODE ---
{code_context}
------------

Be concise, professional, and constructive. Respond in well-formatted Markdown."""


# ---------------------------------------------------------------------------
# LLM review
# ---------------------------------------------------------------------------


def generate_code_review(chunks: List[str]) -> str:
    """
    Generate an AI-powered code review for the supplied code chunks.

    Uses Google Gemini (``gemini-1.5-flash``) when ``GEMINI_API_KEY`` is set
    and ``google-generativeai`` is installed; otherwise returns a structured
    placeholder review.

    Args:
        chunks: Source-code strings to review (typically retrieved by
                :func:`retrieve_relevant_chunks`).  Each string may include a
                metadata header of the form ``# Type: Name | Lines X-Y``.

    Returns:
        Review text formatted as Markdown.
    """
    if not chunks:
        return "No code chunks provided — nothing to review."

    code_context = "\n\n---\n\n".join(
        f"#### Chunk {i + 1}\n```python\n{chunk.strip()}\n```"
        for i, chunk in enumerate(chunks)
    )
    prompt = _REVIEW_PROMPT.format(code_context=code_context)

    if _GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
        logger.info("Calling Gemini API for code review (%d chunks).", len(chunks))
        try:
            review = _call_gemini(prompt)
            logger.info("Gemini review generated successfully.")
            return review
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini API call failed: %s", exc)
            return (
                "## AI Code Review\n\n"
                f"> **Warning:** The AI service returned an error: `{exc}`\n>\n"
                "> Showing placeholder review instead.\n\n"
                + _placeholder_review(chunks)
            )
    logger.info("Gemini unavailable — returning placeholder review.")
    return _placeholder_review(chunks)


def _call_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return the text response.

    Raises:
        RuntimeError: If the API returns no text content.
    """
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_REVIEW_MODEL)
    response = model.generate_content(prompt)
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return response.text


def _placeholder_review(chunks: List[str]) -> str:
    """
    Return a structured placeholder review when Gemini is unavailable.

    Follows the same four-section structure as the real prompt so the UI
    renders consistently regardless of whether the LLM is available.
    """
    chunk_list = ""
    for i, chunk in enumerate(chunks, 1):
        first_line = chunk.strip().splitlines()[0] if chunk.strip() else "<empty>"
        chunk_list += f"- **Chunk {i}:** `{first_line}`\n"

    return (
        "## AI Code Review (Placeholder Mode)\n\n"
        "> **Note:** Set the `GEMINI_API_KEY` environment variable and install\n"
        "> `google-generativeai` to enable real AI-powered review.\n\n"
        f"**Chunks analysed:** {len(chunks)}\n\n"
        f"{chunk_list}\n"
        "---\n\n"
        "## 1. Code Quality Issues\n\n"
        "- Ensure all public functions and classes have docstrings.\n"
        "- Keep functions short and focused on a single responsibility.\n\n"
        "## 2. Performance Improvements\n\n"
        "- Avoid redundant computations inside loops.\n"
        "- Use built-in functions (`sum`, `max`, `any`) where applicable.\n\n"
        "## 3. Naming & Style Suggestions\n\n"
        "- Use `snake_case` for variables and functions; `PascalCase` for classes.\n"
        "- Replace magic numbers with named constants.\n\n"
        "## 4. Potential Bugs\n\n"
        "- Verify edge cases: empty inputs, `None` values, division by zero.\n"
        "- Add input validation at all public API boundaries.\n"
    )

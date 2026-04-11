from __future__ import annotations
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from dotenv import load_dotenv

# Ensure .env is loaded even when this module is imported before main.py runs
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)


def _get_groq_client():
    """Lazy Groq client — created at call time so key is always current."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq  # type: ignore[import]
        return Groq(api_key=api_key)
    except ImportError:
        return None


GROQ_MODEL = "llama-3.3-70b-versatile"
TFIDF_MAX_FEATURES = 512


# ---------------------------------------------------------------------------
# Pure-numpy TF-IDF vectorizer (no sklearn / sentence-transformers needed)
# ---------------------------------------------------------------------------

class _TFIDFVectorizer:
    """Minimal TF-IDF vectorizer built on numpy + Python builtins only."""

    def __init__(self, max_features: int = TFIDF_MAX_FEATURES) -> None:
        self.max_features = max_features
        self.vocabulary_: dict = {}
        self.idf_: np.ndarray = np.array([], dtype=np.float32)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text.lower())

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        tokenized = [self._tokenize(t) for t in texts]
        n_docs = len(tokenized)

        df: Counter = Counter()
        for tokens in tokenized:
            df.update(set(tokens))

        top_terms = [term for term, _ in df.most_common(self.max_features)]
        self.vocabulary_ = {term: i for i, term in enumerate(top_terms)}
        vocab_size = len(self.vocabulary_)

        self.idf_ = np.zeros(vocab_size, dtype=np.float32)
        for term, idx in self.vocabulary_.items():
            self.idf_[idx] = math.log((1 + n_docs) / (1 + df[term])) + 1.0

        return self._vectorize(tokenized)

    def transform(self, texts: List[str]) -> np.ndarray:
        return self._vectorize([self._tokenize(t) for t in texts])

    def _vectorize(self, tokenized: List[List[str]]) -> np.ndarray:
        vocab_size = len(self.vocabulary_)
        mat = np.zeros((len(tokenized), vocab_size), dtype=np.float32)
        for i, tokens in enumerate(tokenized):
            tf = Counter(tokens)
            for term, count in tf.items():
                if term in self.vocabulary_:
                    mat[i, self.vocabulary_[term]] = (
                        count * self.idf_[self.vocabulary_[term]]
                    )
            norm = float(np.linalg.norm(mat[i]))
            if norm > 0.0:
                mat[i] /= norm
        return mat


# ---------------------------------------------------------------------------
# Vector store (pure numpy — no FAISS dependency)
# ---------------------------------------------------------------------------

@dataclass
class VectorStore:
    matrix: np.ndarray   # shape (n_chunks, vocab_size)
    chunks: List[str]
    vectorizer: Any = field(default=None)


def create_faiss_index(chunks: List[str]) -> "VectorStore":
    """Build a TF-IDF vector store from a list of text chunks."""
    if not chunks:
        raise ValueError("Cannot create a vector store from an empty chunk list.")

    vec = _TFIDFVectorizer(max_features=TFIDF_MAX_FEATURES)
    matrix = vec.fit_transform(chunks)   # shape: (n, vocab_size)
    return VectorStore(matrix=matrix, chunks=chunks, vectorizer=vec)


def retrieve_relevant_chunks(
    store: "VectorStore",
    query: str,
    k: int = 20,
) -> List[str]:
    """Return the top-k chunks most similar to *query* (cosine similarity)."""
    k = min(k, len(store.chunks))
    if store.vectorizer is not None:
        query_vec = store.vectorizer.transform([query])  # (1, vocab_size)
    else:
        query_vec = np.zeros((1, store.matrix.shape[1]), dtype=np.float32)

    # Cosine similarity = dot product of L2-normalised vectors
    # Both corpus rows and query are already L2-normalised by _TFIDFVectorizer
    scores = store.matrix.dot(query_vec[0])   # (n,)
    top_indices = np.argsort(scores)[::-1][:k]
    return [store.chunks[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REVIEW_PROMPT = (
    "You are an expert Python code reviewer conducting a professional code review.\n"
    "Analyse the following Python code chunks and produce a structured report.\n\n"
    "For each issue, reference the exact function or class name and the approximate\n"
    "line numbers provided in the chunk headers.\n\n"
    "Structure your entire response using EXACTLY these four sections in this order.\n"
    "Only include a section if you have findings for it -- otherwise omit it.\n\n"
    "---\n\n"
    "## 1. Code Quality Issues\n"
    "List issues related to readability, PEP 8 violations, overly complex logic,\n"
    "missing docstrings, or poor code organisation.\n\n"
    "## 2. Performance Improvements\n"
    "Highlight inefficiencies, unnecessary computations, suboptimal data structures,\n"
    "or operations that could be vectorised or cached.\n\n"
    "## 3. Naming & Style Suggestions\n"
    "Point out unclear variable names, non-snake_case identifiers, magic numbers,\n"
    "or style inconsistencies that reduce readability.\n\n"
    "## 4. Potential Bugs\n"
    "Identify logic errors, unhandled edge cases, missing input validation, or "
    "other likely runtime failures.\n\n"
    "---\n\n"
    "For every finding use this format:\n"
    "- **`FunctionOrClassName` (Lines X-Y):** Brief description of the issue.\n"
    "  *Suggestion:* Concrete, actionable fix.\n\n"
    "If the code is clean in a category, write: *(No issues found.)*\n\n"
    "--- CODE ---\n"
    "{code_context}\n"
    "------------\n\n"
    "Be concise, professional, and constructive. Respond in well-formatted Markdown."
)

_CORRECT_PROMPT = (
    "You are an expert Python developer. Your task is to fix ALL bugs, errors, and "
    "code quality issues in the Python code below.\n\n"
    "Rules you MUST follow:\n"
    "1. Return ONLY the corrected Python source code -- no explanations, no markdown "
    "fences, no prose.\n"
    "2. Fix every bug, logic error, unhandled edge case, PEP 8 violation, and "
    "naming issue you find.\n"
    "3. Do NOT add new features; only correct existing code.\n"
    "4. Preserve the original structure, function names, and variable names unless "
    "a name itself is the bug.\n\n"
    "--- ORIGINAL CODE ---\n"
    "{original_code}\n"
    "---------------------\n\n"
    "Output the corrected Python code only."
)


def _call_groq(client, prompt: str) -> str:
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = completion.choices[0].message.content
    if not text:
        raise RuntimeError("Groq returned an empty response.")
    return text


def _placeholder_review(chunks: List[str]) -> str:
    chunk_list = "".join(
        f"- **Chunk {i}:** `{chunk.strip().splitlines()[0] if chunk.strip() else '<empty>'}`\n"
        for i, chunk in enumerate(chunks, 1)
    )
    return f"{len(chunks)}\n\n{chunk_list}\n"


def generate_code_review(chunks: List[str]) -> str:
    if not chunks:
        return "No code chunks provided -- nothing to review."

    code_context = "\n\n---\n\n".join(
        f"#### Chunk {i + 1}\n```python\n{chunk.strip()}\n```"
        for i, chunk in enumerate(chunks)
    )
    prompt = _REVIEW_PROMPT.format(code_context=code_context)

    client = _get_groq_client()
    if client is not None:
        logger.info("Calling Groq API for code review (%d chunks).", len(chunks))
        try:
            review = _call_groq(client, prompt)
            logger.info("Groq review generated successfully.")
            return review
        except Exception as exc:
            logger.error("Groq API call failed: %s", exc)
            return (
                "## AI Code Review\n\n"
                f"> **Warning:** The AI service returned an error: `{exc}`\n>\n"
                "> Showing placeholder review instead.\n\n"
                + _placeholder_review(chunks)
            )
    logger.info("Groq unavailable -- returning placeholder review.")
    return _placeholder_review(chunks)


def generate_corrected_code(original_code: str) -> str:
    if not original_code.strip():
        return original_code

    client = _get_groq_client()
    if client is not None:
        prompt = _CORRECT_PROMPT.format(original_code=original_code)
        logger.info("Calling Groq API to generate corrected code.")
        try:
            corrected = _call_groq(client, prompt)
            corrected = corrected.strip()
            if corrected.startswith("```"):
                lines = corrected.splitlines()
                lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                corrected = "\n".join(lines)
            logger.info("Corrected code generated successfully.")
            return corrected
        except Exception as exc:
            logger.error("Groq correction call failed: %s", exc)
            return original_code

    logger.info("Groq unavailable -- returning original code unchanged.")
    return original_code

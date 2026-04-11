"""Write the full rag_search.py content to disk."""
import os

DEST = r"c:\Users\Huzaifa\OneDrive\Desktop\AI_Code_assistant\ai_code_review\backend\rag_search.py"

CONTENT = r'''from __future__ import annotations
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


try:
    import faiss  # type: ignore[import]
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

GROQ_MODEL = "llama-3.3-70b-versatile"
TFIDF_MAX_FEATURES = 512  # FAISS index dimension = min(vocab_size, 512)


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
                    mat[i, self.vocabulary_[term]] = count * self.idf_[self.vocabulary_[term]]
            norm = float(np.linalg.norm(mat[i]))
            if norm > 0.0:
                mat[i] /= norm
        return mat


# ---------------------------------------------------------------------------
# FAISS store
# ---------------------------------------------------------------------------

@dataclass
class FAISSStore:
    index: object                        # faiss.IndexFlatL2
    chunks: List[str]
    dim: int
    vectorizer: Any = field(default=None)  # _TFIDFVectorizer instance


def create_faiss_index(chunks: List[str]) -> "FAISSStore":
    if not _FAISS_AVAILABLE:
        raise ImportError(
            "faiss-cpu is required. Install it with: pip install faiss-cpu"
        )
    if not chunks:
        raise ValueError("Cannot create a FAISS index from an empty chunk list.")

    vec = _TFIDFVectorizer(max_features=TFIDF_MAX_FEATURES)
    embeddings = vec.fit_transform(chunks)   # shape: (n, dim)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return FAISSStore(index=index, chunks=chunks, dim=dim, vectorizer=vec)


def retrieve_relevant_chunks(
    store: "FAISSStore",
    query: str,
    k: int = 20,
) -> List[str]:
    k = min(k, len(store.chunks))
    if store.vectorizer is not None:
        query_vec = store.vectorizer.transform([query])  # shape: (1, dim)
    else:
        query_vec = np.zeros((1, store.dim), dtype=np.float32)
    _, indices = store.index.search(query_vec, k)
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
Only include a section if you have findings for it -- otherwise omit it.

---

## 1. Code Quality Issues
List issues related to readability, PEP 8 violations, overly complex logic,
missing docstrings, or poor code organisation.

## 2. Performance Improvements
Highlight inefficiencies, unnecessary computations, suboptimal data structures,
or operations that could be vectorised or cached.

## 3. Naming & Style Suggestions
Point out unclear variable names, non-snake_case identifiers, magic numbers,
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


def _call_groq(client, prompt: str) -> str:
    """Send a prompt to Groq and return the text response."""
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
    chunk_list = ""
    for i, chunk in enumerate(chunks, 1):
        first_line = chunk.strip().splitlines()[0] if chunk.strip() else "<empty>"
        chunk_list += f"- **Chunk {i}:** `{first_line}`\n"
    return f"{len(chunks)}\n\n{chunk_list}\n"


_CORRECT_PROMPT = """\
You are an expert Python developer. Your task is to fix ALL bugs, errors, and \
code quality issues in the Python code below.

Rules you MUST follow:
1. Return ONLY the corrected Python source code -- no explanations, no markdown \
fences, no prose.
2. Fix every bug, logic error, unhandled edge case, PEP 8 violation, and \
naming issue you find.
3. Do NOT add new features; only correct existing code.
4. Preserve the original structure, function names, and variable names unless \
a name itself is the bug.

--- ORIGINAL CODE ---
{original_code}
---------------------

Output the corrected Python code only."""


def generate_corrected_code(original_code: str) -> str:
    """Return a corrected version of original_code using Groq."""
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
'''

with open(DEST, "w", encoding="utf-8") as f:
    f.write(CONTENT)

size = os.path.getsize(DEST)
print(f"Written {size} bytes to {DEST}")

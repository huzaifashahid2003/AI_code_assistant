"""
rag_service.py — TF-IDF vector store and retrieval.

Pure-numpy implementation; no FAISS or external embedding libraries required.
Retrieval returns chunk indices so callers can index into any parallel list
(source strings, CodeChunk objects) without a fragile string-mapping step.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np

from app.core.config import TFIDF_MAX_FEATURES


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


@dataclass
class VectorStore:
    matrix: np.ndarray    # shape (n_chunks, vocab_size), L2-normalised rows
    chunks: List[str]
    vectorizer: Any = field(default=None)


def build_tfidf_index(chunks: List[str]) -> VectorStore:
    """Fit a TF-IDF vectorizer on *chunks* and return a VectorStore."""
    if not chunks:
        raise ValueError("Cannot create a vector store from an empty chunk list.")
    vec = _TFIDFVectorizer(max_features=TFIDF_MAX_FEATURES)
    matrix = vec.fit_transform(chunks)
    return VectorStore(matrix=matrix, chunks=chunks, vectorizer=vec)


def retrieve_relevant_chunks(store: VectorStore, query: str, k: int = 5) -> List[int]:
    """Return indices of the top-k chunks most similar to *query*.

    Uses cosine similarity (dot product on L2-normalised vectors).
    Returns plain Python ints so callers can directly index into any parallel
    list without a separate string-to-object mapping step.
    """
    k = min(k, len(store.chunks))
    if store.vectorizer is not None:
        query_vec = store.vectorizer.transform([query])   # (1, vocab_size)
    else:
        query_vec = np.zeros((1, store.matrix.shape[1]), dtype=np.float32)

    scores = store.matrix.dot(query_vec[0])               # (n_chunks,)
    top_indices = np.argsort(scores)[::-1][:k]
    return top_indices.tolist()                            # numpy int64 → int

"""Hybrid retrieval over the context notes (Section 6: dense + BM25, metadata pre-filter).

Each note is one chunk -- they are one-to-two sentences, so splitting would only
lose context. Retrieval is: metadata pre-filter (route scope + time proximity)
-> hybrid rank -> top-k. Retrieval only *proposes* candidates; whether a note may
justify a flag is decided by the deterministic checks in verdict.py.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .config import CACHE_DIR, CANDIDATE_SLACK_WEEKS, EMBEDDING_MODEL, RETRIEVAL_ALPHA, RETRIEVAL_TOP_K
from .notes import Note

log = logging.getLogger(__name__)
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "a", "an", "and", "or", "of", "on", "in", "to", "for", "by", "was", "were", "this", "that",
              "is", "are", "be", "as", "at", "with", "from", "but", "not", "no", "about", "after", "again"}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Hit:
    note: Note
    score: float
    dense: float
    keyword: float


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [_tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg_len = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = _tokens(query)
        out = np.zeros(len(self.docs))
        for i, doc in enumerate(self.docs):
            tf = Counter(doc)
            norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len)
            out[i] = sum(self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        return out


class DenseEmbedder:
    """Local open-source embeddings (bge-small via ONNX). Deterministic on CPU, $0."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from fastembed import TextEmbedding  # optional dependency

        # Cache inside the project (default is a temp dir the OS may clear before a demo).
        self.model = TextEmbedding(model_name, cache_dir=str(CACHE_DIR / "fastembed"))
        self.name = model_name

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.embed(texts)))

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(list(self.model.query_embed(text)))[0]


class NoteIndex:
    def __init__(self, notes: list[Note], use_dense: bool = True, alpha: float = RETRIEVAL_ALPHA):
        self.notes = notes
        self.alpha = alpha
        docs = [f"{n.applies_to}. {n.text}" for n in notes]
        self.bm25 = BM25(docs)
        self.embedder: DenseEmbedder | None = None
        self.doc_vectors: np.ndarray | None = None
        if use_dense:
            try:
                self.embedder = DenseEmbedder()
                self.doc_vectors = self.embedder.embed_docs(docs)
            except Exception as exc:  # missing package or model download failure
                log.warning("Dense embeddings unavailable (%s); falling back to BM25-only retrieval", exc)
                self.embedder = None
        self.mode = f"hybrid (alpha={alpha}, {self.embedder.name})" if self.embedder else "bm25-only"

    def search(self, query: str, candidates: list[Note] | None = None, top_k: int = RETRIEVAL_TOP_K) -> list[Hit]:
        pool = candidates if candidates is not None else self.notes
        if not pool:
            return []
        idx = [self.notes.index(n) for n in pool]
        keyword = self.bm25.scores(query)[idx]
        keyword = keyword / keyword.max() if keyword.max() > 0 else keyword
        if self.embedder is not None:
            dense = self.doc_vectors[idx] @ self.embedder.embed_query(query)
            combined = self.alpha * dense + (1 - self.alpha) * keyword
        else:
            dense = np.zeros(len(idx))
            combined = keyword
        hits = [Hit(pool[i], float(combined[i]), float(dense[i]), float(keyword[i])) for i in range(len(pool))]
        # Stable tie-break on note_id keeps ranking identical across runs.//
        hits.sort(key=lambda h: (-round(h.score, 6), h.note.note_id))
        return hits[:top_k]

    def candidates_for(self, route: str, week_of: date, slack_weeks: int = CANDIDATE_SLACK_WEEKS) -> list[Note]:
        """Metadata pre-filter: notes scoped to this route (or All Routes) whose window is near this week."""
        lo = week_of - timedelta(weeks=slack_weeks)
        hi = week_of + timedelta(days=6) + timedelta(weeks=slack_weeks)
        return [n for n in self.notes if n.applies_to_route(route) and n.overlaps(lo, hi)]


def query_for(route: str, week_of: date) -> str:
    return f"{route} freight cost increase surcharge disruption week of {week_of:%d %B %Y}"

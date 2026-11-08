"""
Hybrid retrieval for timestamped OmniSense transcript chunks.

Combines:
- Semantic retrieval: MiniLM + FAISS
- Keyword retrieval: BM25
- Rank fusion: Reciprocal Rank Fusion (RRF)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from omnisense.pipelines.audio import TranscriptChunk
from omnisense.pipelines.search import TranscriptSearchIndex


@dataclass
class HybridSearchHit:
    """A hybrid search result with preserved transcript timestamps."""

    chunk: TranscriptChunk
    score: float
    rank: int
    semantic_rank: int | None = None
    keyword_rank: int | None = None


def _tokenize(text: str) -> list[str]:
    """Simple, case-insensitive tokenizer that retains technical terms."""
    return re.findall(r"[\w.-]+", text.casefold())


class HybridTranscriptSearchIndex:
    """Combine semantic and keyword retrieval using Reciprocal Rank Fusion."""

    def __init__(
        self,
        semantic_index: TranscriptSearchIndex,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        self.semantic_index = semantic_index
        self.rrf_k = rrf_k
        self._chunks: list[TranscriptChunk] = []
        self._bm25: BM25Okapi | None = None
        self._ready = False

    def build(self, chunks: list[TranscriptChunk]) -> None:
        """Build both indexes from the same transcript chunks."""
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        tokenized = [_tokenize(chunk.text) for chunk in chunks]

        if not any(tokenized):
            raise ValueError("Cannot index chunks containing only empty text.")

        # Build semantic search first. Mark this index ready only on success.
        self.semantic_index.build(chunks)

        self._chunks = list(chunks)
        self._bm25 = BM25Okapi(tokenized)
        self._ready = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_semantic_score: float = 0.30,
        candidate_k: int | None = None,
    ) -> list[HybridSearchHit]:
        """Return results ranked by fused semantic and keyword relevance."""
        if not self._ready or self._bm25 is None:
            raise RuntimeError("Index not built. Call build() before search().")

        query = query.strip()

        if not query or top_k <= 0:
            return []

        if candidate_k is None:
            candidate_k = max(top_k * 3, top_k)

        if candidate_k <= 0:
            return []

        candidate_k = min(candidate_k, len(self._chunks))

        semantic_hits = self.semantic_index.search(
            query,
            top_k=candidate_k,
            min_score=min_semantic_score,
        )

        query_tokens = _tokenize(query)
        keyword_scores = (
            self._bm25.get_scores(query_tokens)
            if query_tokens
            else []
        )

        keyword_indices = sorted(
            range(len(self._chunks)),
            key=lambda i: keyword_scores[i],
            reverse=True,
        )

        # Store each segment's position in each ranking.
        semantic_ranks = {
            id(hit.chunk): hit.rank
            for hit in semantic_hits
        }

        keyword_ranks: dict[int, int] = {}

        for rank, idx in enumerate(keyword_indices[:candidate_k], start=1):
            if keyword_scores[idx] > 0:
                keyword_ranks[id(self._chunks[idx])] = rank

        # Reciprocal Rank Fusion avoids adding incomparable raw scores.
        fused_scores: dict[int, float] = {}
        chunks_by_id: dict[int, TranscriptChunk] = {}

        for chunk in self._chunks:
            chunks_by_id[id(chunk)] = chunk

        for chunk_id, rank in semantic_ranks.items():
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        for chunk_id, rank in keyword_ranks.items():
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                -fused_scores[chunk_id],
                semantic_ranks.get(chunk_id, float("inf")),
                keyword_ranks.get(chunk_id, float("inf")),
            ),
        )

        return [
            HybridSearchHit(
                chunk=chunks_by_id[chunk_id],
                score=round(fused_scores[chunk_id], 6),
                rank=rank,
                semantic_rank=semantic_ranks.get(chunk_id),
                keyword_rank=keyword_ranks.get(chunk_id),
            )
            for rank, chunk_id in enumerate(ranked_ids[:top_k], start=1)
        ]

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def segment_count(self) -> int:
        return len(self._chunks)
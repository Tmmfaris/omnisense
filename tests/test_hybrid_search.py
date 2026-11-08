
import pytest

from omnisense.pipelines.audio import TranscriptChunk
from omnisense.pipelines.search import SearchHit
from omnisense.pipelines.hybrid_search import (
    HybridTranscriptSearchIndex,
)


class FakeSemanticIndex:
    def __init__(self):
        self.chunks = []
        self.ready = False

    def build(self, chunks):
        self.chunks = list(chunks)
        self.ready = True

    def search(self, query, top_k=5, min_score=0.30):
        if not self.ready:
            raise RuntimeError("Index not built.")

        if not query.strip():
            return []

        # Simulate a semantic ranking for deterministic tests.
        return [
            SearchHit(chunk=c, score=0.9, rank=i)
            for i, c in enumerate(self.chunks[:top_k], start=1)
        ]


@pytest.fixture
def chunks():
    return [
        TranscriptChunk(
            text="Python is used for machine learning.",
            start=10.0, end=15.0, chunk_id=1,
        ),
        TranscriptChunk(
            text="The weather is sunny today.",
            start=20.0, end=25.0, chunk_id=2,
        ),
        TranscriptChunk(
            text="FAISS performs vector similarity search.",
            start=30.0, end=35.0, chunk_id=3,
        ),
    ]


@pytest.fixture
def index():
    return HybridTranscriptSearchIndex(FakeSemanticIndex())


def test_search_before_build_raises(index):
    with pytest.raises(RuntimeError):
        index.search("Python")


def test_build_empty_raises(index):
    with pytest.raises(ValueError):
        index.build([])


def test_build_sets_ready(index, chunks):
    index.build(chunks)

    assert index.is_ready
    assert index.segment_count == 3


def test_empty_query_returns_no_results(index, chunks):
    index.build(chunks)

    assert index.search("") == []
    assert index.search("   ") == []


def test_keyword_search_finds_exact_term(index, chunks):
    index.build(chunks)

    results = index.search(
        "FAISS",
        top_k=3,
        min_semantic_score=0.0,
    )

    assert results
    assert any("FAISS" in hit.chunk.text for hit in results)


def test_timestamps_are_preserved(index, chunks):
    index.build(chunks)

    results = index.search(
        "machine learning",
        top_k=3,
        min_semantic_score=0.0,
    )

    hit = next(
        hit for hit in results
        if hit.chunk.chunk_id == 1
    )

    assert hit.chunk.start == 10.0
    assert hit.chunk.end == 15.0


def test_result_ranks_are_sequential(index, chunks):
    index.build(chunks)

    results = index.search(
        "Python",
        top_k=3,
        min_semantic_score=0.0,
    )

    assert [hit.rank for hit in results] == list(
        range(1, len(results) + 1)
    )

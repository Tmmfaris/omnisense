
"""Evaluate semantic vs hybrid transcript search."""

import json
from pathlib import Path
from statistics import mean

from omnisense.pipelines.audio import TranscriptChunk
from omnisense.pipelines.search import TranscriptSearchIndex
from omnisense.pipelines.hybrid_search import HybridTranscriptSearchIndex

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "benchmark.json"
TOP_K = 5


def precision_at_k(results, relevant_ids, k=TOP_K):
    returned = [hit.chunk.chunk_id for hit in results[:k]]
    if k <= 0:
        return 0.0
    return sum(i in relevant_ids for i in returned) / k


def recall_at_k(results, relevant_ids, k=TOP_K):
    if not relevant_ids:
        return 0.0
    returned = {hit.chunk.chunk_id for hit in results[:k]}
    return len(returned & relevant_ids) / len(relevant_ids)


def reciprocal_rank(results, relevant_ids, k=TOP_K):
    for rank, hit in enumerate(results[:k], start=1):
        if hit.chunk.chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {DATA_FILE}\n"
            "Create evaluation/benchmark.json first."
        )

    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    chunks = [
        TranscriptChunk(
            text=item["text"],
            start=float(item["start"]),
            end=float(item["end"]),
            chunk_id=int(item["chunk_id"]),
        )
        for item in data["chunks"]
    ]
    queries = data["queries"]

    if not chunks or not queries:
        raise ValueError("Benchmark needs chunks and queries.")

    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Every chunk_id must be unique.")

    available_ids = set(ids)
    for item in queries:
        relevant = set(item["relevant_chunk_ids"])
        if not relevant or not relevant <= available_ids:
            raise ValueError(
                f"Invalid relevant IDs for query: {item['query']}"
            )

    semantic = TranscriptSearchIndex()
    hybrid = HybridTranscriptSearchIndex(semantic)
    hybrid.build(chunks)

    metrics = {
        "Semantic": {"Precision@5": [], "Recall@5": [], "MRR@5": []},
        "Hybrid": {"Precision@5": [], "Recall@5": [], "MRR@5": []},
    }

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_chunk_ids"])

        semantic_results = semantic.search(
            query, top_k=TOP_K, min_score=0.30
        )
        hybrid_results = hybrid.search(
            query, top_k=TOP_K, min_semantic_score=0.30
        )

        for name, results in [
            ("Semantic", semantic_results),
            ("Hybrid", hybrid_results),
        ]:
            metrics[name]["Precision@5"].append(
                precision_at_k(results, relevant)
            )
            metrics[name]["Recall@5"].append(
                recall_at_k(results, relevant)
            )
            metrics[name]["MRR@5"].append(
                reciprocal_rank(results, relevant)
            )

        print(f"\nQuery: {query}")
        print("Semantic IDs:", [h.chunk.chunk_id for h in semantic_results])
        print("Hybrid IDs:  ", [h.chunk.chunk_id for h in hybrid_results])

    print("\n=== Average Evaluation Results ===")
    print(f"{'Method':<12} {'Precision@5':>12} {'Recall@5':>10} {'MRR@5':>8}")

    for name, values in metrics.items():
        print(
            f"{name:<12} "
            f"{mean(values['Precision@5']):>12.3f} "
            f"{mean(values['Recall@5']):>10.3f} "
            f"{mean(values['MRR@5']):>8.3f}"
        )


if __name__ == "__main__":
    main()

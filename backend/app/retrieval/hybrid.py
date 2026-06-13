"""Hybrid retrieval: fuse vector and keyword search with Reciprocal Rank Fusion.

Vector search (ChromaDB) finds chunks by meaning; BM25 finds them by exact words.
Their scores are on incompatible scales, so we can't merge by raw score. Instead
we use Reciprocal Rank Fusion (RRF): each result scores 1/(k + rank) by its
*position* in each list, and a chunk's RRF scores from both lists are summed.
Chunks found by both methods rise to the top; the score scales never need to match.
"""

from app.core.config import settings
from app.ingestion import bm25_index, embedder


def _reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Fuse multiple ranked result lists into one using Reciprocal Rank Fusion.

    Each result earns 1/(k + rank) points from each list it appears in, where
    rank is its 1-based position. Points are summed across lists, so chunks
    ranked well by multiple methods rise to the top. Dedup is by chunk text.
    k=60 (the conventional default) dampens top ranks so lower-ranked results
    still contribute.
    """
    fused: dict[str, dict] = {}

    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            key = hit["text"]
            rrf_score = 1.0 / (k + rank)

            if key not in fused:
                fused[key] = {
                    "text": hit["text"],
                    "metadata": hit["metadata"],
                    "score": rrf_score,
                }
            else:
                fused[key]["score"] += rrf_score

    return sorted(fused.values(), key=lambda h: h["score"], reverse=True)


def search(query: str, top_k: int | None = None) -> list[dict]:
    """Hybrid search: run vector and BM25 retrieval, fuse with RRF.

    Returns the fused top results as {text, metadata, score} dicts — the same
    shape both retrievers use, so downstream consumers (reranker, generation)
    don't care how results were found.
    """
    vector_hits = embedder.search(query, top_k=settings.vector_top_k)
    keyword_hits = bm25_index.search(query, top_k=settings.bm25_top_k)

    fused = _reciprocal_rank_fusion([vector_hits, keyword_hits])

    n_results = top_k or (settings.vector_top_k + settings.bm25_top_k)
    return fused[:n_results]

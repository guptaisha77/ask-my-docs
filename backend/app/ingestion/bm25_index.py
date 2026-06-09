"""Keyword search over chunks using BM25.

BM25 is the classic keyword-ranking algorithm (the kind inside Elasticsearch).
It scores a chunk by how many query words it contains, weighting rare words more
than common ones and correcting for chunk length. It complements vector search,
which matches meaning but misses exact strings like codes and clause numbers.

The index is held in memory and pickled to disk so it survives restarts. Because
BM25's word statistics depend on the whole collection, adding chunks rebuilds the
index — fine at this scale; a production system would use Elasticsearch.
"""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.ingestion.chunker import Chunk

# BM25 needs ALL chunks in memory to compute word statistics, so we hold them
# here. _chunks keeps the original Chunk objects (for returning results);
# _bm25 is the built index. Both are rebuilt together when chunks are added.
_chunks: list[Chunk] = []
_bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens for BM25.

    BM25 works on words, not the embedding model's subword tokens. We lowercase
    so "Payment" and "payment" match, and use a simple word-character regex.
    """
    return re.findall(r"\b\w+\b", text.lower())


def add_chunks(chunks: list[Chunk]) -> int:
    """Add chunks to the BM25 index and persist it to disk.

    Because BM25's word statistics depend on the whole collection, we append the
    new chunks to the in-memory list and rebuild the index over everything, then
    pickle it to disk. Returns the number of chunks now indexed in total.
    """
    global _chunks, _bm25

    if not chunks:
        return len(_chunks)

    _chunks.extend(chunks)

    # Tokenize every chunk's text into a list of words. BM25Okapi takes a list
    # of token-lists (one per document) and computes its word statistics.
    tokenized_corpus = [_tokenize(chunk.text) for chunk in _chunks]
    _bm25 = BM25Okapi(tokenized_corpus)

    _save_index()
    return len(_chunks)


def _index_path() -> Path:
    """Where the BM25 index is stored on disk."""
    return Path(settings.bm25_index_path)


def _save_index() -> None:
    """Pickle the chunks and index to disk so they survive a restart."""
    with open(_index_path(), "wb") as f:
        pickle.dump({"chunks": _chunks, "bm25": _bm25}, f)


def load_index() -> None:
    """Load a previously-saved index from disk, if one exists.

    Called once at startup. If no file exists yet (first run), does nothing and
    the index stays empty until chunks are added.
    """
    global _chunks, _bm25

    path = _index_path()
    if not path.exists():
        return

    with open(path, "rb") as f:
        data = pickle.load(f)
        _chunks = data["chunks"]
        _bm25 = data["bm25"]


def search(query: str, top_k: int | None = None) -> list[dict]:
    """Return the chunks with the best BM25 keyword match for `query`.

    Tokenizes the query, scores every chunk, and returns the highest-scoring ones
    as {text, metadata, score} dicts — the same shape as the vector search, so the
    hybrid retriever can combine both uniformly.
    """
    n_results = top_k or settings.bm25_top_k

    if _bm25 is None or not _chunks:
        return []

    query_tokens = _tokenize(query)
    scores = _bm25.get_scores(query_tokens)

    # Pair each chunk with its score, sort by score descending, take the top n.
    ranked = sorted(
        zip(_chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True
    )

    hits: list[dict] = []
    for chunk, score in ranked[:n_results]:
        hits.append(
            {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": float(score),
            }
        )
    return hits

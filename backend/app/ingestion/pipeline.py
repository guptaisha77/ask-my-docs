"""Ingestion pipeline: one call from a file on disk to both search stores.

extract -> chunk -> embed (ChromaDB) + index (BM25), in a single operation so
the two stores can never drift out of sync.
"""

from pathlib import Path

from app.ingestion import bm25_index, embedder
from app.ingestion.chunker import chunk_text
from app.ingestion.extractor import extract_text


def ingest_file(file_path: str) -> int:
    """Ingest one document end to end. Returns the number of chunks stored."""
    text = extract_text(file_path)

    doc_id = Path(file_path).stem
    metadata = {"source": Path(file_path).name}

    chunks = chunk_text(text, doc_id=doc_id, metadata=metadata)

    embedder.add_chunks(chunks)
    bm25_index.add_chunks(chunks)

    return len(chunks)

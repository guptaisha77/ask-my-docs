"""Embed chunks into ChromaDB and search them by meaning.

ChromaDB is a vector database: it stores each chunk as a 384-dimensional vector
(produced by the all-MiniLM-L6-v2 model) plus the original text and metadata.
We let Chroma run the embedding model for us via an embedding function, so this
module never handles raw vectors directly — it hands Chroma text and gets back
the nearest chunks for a query.
"""

import chromadb
from chromadb.utils import embedding_functions

from app.core.config import settings
from app.ingestion.chunker import Chunk

# The embedding function tells Chroma WHICH model to turn text into vectors with.
# all-MiniLM-L6-v2 runs locally (no API key) and produces 384-dim vectors.
# Built once at import and reused — loading the model is the expensive part.
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=settings.embedding_model
)


# Connect to a ChromaDB database that persists to disk at settings.chroma_persist_dir.
# PersistentClient means vectors survive restarts (vs an in-memory client that
# forgets everything when the process ends). Built once and reused.
_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)


def _get_collection() -> chromadb.Collection:
    """Return the Chroma collection, creating it if it doesn't exist yet.

    A collection is like a table that holds vectors + text + metadata. Binding
    the embedding function here means Chroma uses MiniLM automatically for both
    storing chunks and embedding queries.
    """
    return _client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Chunk]) -> int:
    """Store a list of Chunk objects in ChromaDB.

    Chroma needs three parallel lists: ids, documents (the text), and metadatas.
    We build them from the Chunk objects. Chroma runs the embedding function on
    each document automatically, so we never produce vectors ourselves.
    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    collection = _get_collection()

    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            **chunk.metadata,
        }
        for chunk in chunks
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def search(query: str, top_k: int | None = None) -> list[dict]:
    """Find the chunks most similar in meaning to `query`.

    Chroma embeds the query with the same model, compares it to stored vectors
    by cosine distance, and returns the nearest ones. Returns a list of dicts,
    each with the chunk's text, metadata, and similarity score.
    """
    n_results = top_k or settings.vector_top_k
    collection = _get_collection()

    results = collection.query(query_texts=[query], n_results=n_results)

    hits: list[dict] = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, metadata, distance in zip(documents, metadatas, distances, strict=True):
        hits.append(
            {
                "text": text,
                "metadata": metadata,
                "score": 1 - distance,
            }
        )
    return hits

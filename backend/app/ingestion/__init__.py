"""Document ingestion: turning raw documents into searchable chunks.

This package handles the one-time work performed when a document is uploaded:

    extractor  — pull plain text out of PDFs and text files
    chunker    — split text into ~700-token overlapping chunks
    embedder   — store chunks as vectors in ChromaDB (semantic search)
    bm25_index — store chunks in a BM25 keyword index (keyword search)
    pipeline   — orchestrate the above into one ingestion flow

Each chunk is stored in both the vector store and the keyword index, which is
what enables hybrid retrieval at query time.
"""

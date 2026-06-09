"""Ask My Docs — backend application.

A question-answering system over your own documents, built on retrieval-augmented
generation (RAG). Upload documents, ask questions in plain English, and get
answers that cite the exact source passages they came from.

Packages:

    core       — configuration and settings
    ingestion  — extract, chunk, embed, and index documents
    retrieval  — hybrid search (vector + keyword) and reranking
    graph      — the LangGraph agent that generates cited answers

Retrieval is hybrid (semantic + keyword), results are reranked, and every answer
is validated to cite its sources.
"""

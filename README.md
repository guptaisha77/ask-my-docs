# Ask My Docs

A domain-specific question-answering system over your own documents, built with
**hybrid retrieval** (keyword + semantic search), **cross-encoder reranking**,
**enforced citations**, and a **CI-gated evaluation pipeline**.

Upload a set of documents, ask questions in natural language, and get answers
that cite the exact source passages they came from — with answer quality
measured automatically on every change.

> **Status:** Active development. This README documents the full target
> architecture; see the [build status](#build-status) table for what is live today.

---

## Why this project exists

Most retrieval-augmented generation (RAG) demos stop at "embed documents, do a
vector search, ask an LLM." That falls apart in practice: vector search misses
exact terms (product codes, clause numbers), answers can't be trusted without
sources, and there's no way to know if a change made quality better or worse.

This project addresses all three:

- **Hybrid retrieval** combines BM25 keyword search (catches exact terms) with
  vector search (catches meaning), so neither blind spot sinks an answer.
- **Cross-encoder reranking** re-scores candidates with a model that reads the
  question and passage *together*, lifting the genuinely relevant chunks to the top.
- **Citation enforcement** validates that every answer points back to retrieved
  source passages, so responses are auditable rather than trusted blindly.
- **CI-gated evaluation** runs RAGAS metrics on every pull request and blocks
  merges that regress answer quality below set thresholds.
- **Monitoring & observability** (planned) adds tracing, latency percentiles
  (p50/p95), and cost-per-request — because operating a RAG system is most of the
  real work, and the code is built with the seams to support it from the start.

---

## Architecture

```mermaid
flowchart TD
    subgraph FE["Frontend — React + Vite + TanStack Query"]
        UI["Upload UI"]
        Chat["Chat interface"]
        Cite["Citation panel"]
    end

    subgraph ING["Ingestion pipeline"]
        Extract["Text extractor<br/>pypdf / plaintext"]
        Chunk["Chunker<br/>700 tokens, 100 overlap"]
        Embed["Embedder<br/>MiniLM-L6-v2"]
        BM25idx["BM25 index<br/>rank-bm25"]
    end

    subgraph STORE["Storage"]
        Chroma[("ChromaDB<br/>vector store")]
        Pickle[("BM25 index<br/>on disk")]
    end

    subgraph RET["Retrieval + reranking"]
        Hybrid["Hybrid merger<br/>dedup + fuse"]
        Rerank["Cohere reranker<br/>cross-encoder, top-5"]
    end

    subgraph GEN["Generation — LangGraph agent"]
        Ctx["Build context"]
        LLM["Generate answer<br/>Groq Llama 3.1"]
        Valid["Citation validator<br/>retry if missing"]
    end

    subgraph EVAL["Evaluation + CI"]
        Testset["Eval testset"]
        Ragas["RAGAS metrics"]
        Gate["CI gate<br/>fail below threshold"]
    end

    UI --> Extract
    Extract --> Chunk
    Chunk --> Embed
    Chunk --> BM25idx
    Embed --> Chroma
    BM25idx --> Pickle

    Chat --> Hybrid
    Pickle --> Hybrid
    Chroma --> Hybrid
    Hybrid --> Rerank
    Rerank --> Ctx
    Ctx --> LLM
    LLM --> Valid
    Valid --> Cite

    Testset --> Ragas
    Ragas --> Gate
```

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | React, Vite, Bun, TanStack Query, Biome | Fast tooling, production data-fetching, single lint+format tool |
| Backend | Python, FastAPI | Type-validated API with auto-generated docs |
| Vector store | ChromaDB | Local, zero-infra vector database |
| Embeddings | sentence-transformers (MiniLM-L6-v2) | Runs locally, no API cost |
| Keyword search | rank-bm25 | Exact-term matching to complement vectors |
| Reranking | Cohere Rerank | Cross-encoder relevance scoring |
| LLM / generation | Groq (Llama 3.1) | Fast, low-cost inference via an OpenAI-compatible API |
| Orchestration | LangChain, LangGraph | Explicit, branchable agent graph |
| Evaluation | RAGAS | Faithfulness, relevancy, context precision/recall |
| CI | GitHub Actions | Automated eval gate on every PR |

---

## Build status

| Component | Status |
|---|---|
| Project scaffold + config | ✅ Done |
| Document ingestion — token-aware chunker | ✅ Done |
| Embedding + ChromaDB storage | ✅ Done |
| BM25 keyword index | ✅ Done |
| Text extractor (PDF / plaintext) | ⬜ Planned |
| Hybrid retrieval + Cohere reranking | 🚧 In progress |
| LangGraph agent + citation enforcement | ⬜ Planned |
| React frontend | ⬜ Planned |
| RAGAS evaluation + CI gate | ⬜ Planned |
| Monitoring & observability (tracing, p50/p95 latency, cost-per-request) | ⬜ Planned |

---

## Getting started

### Prerequisites

- Python 3.11+
- [Bun](https://bun.sh) (for the frontend)
- A Groq API key (from [console.groq.com](https://console.groq.com)) and a Cohere API key

### Backend setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp .env.example .env             # then edit .env with your real keys

# Run the API
uvicorn app.main:app --reload --port 8000
```

The interactive API docs are then available at `http://localhost:8000/docs`.

### Frontend setup

```bash
cd frontend
bun install
bun run dev
```

---

## Configuration

All settings live in `backend/app/core/config.py` and are loaded from environment
variables or a `.env` file. Key tunables:

| Setting | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | (none) | Your Groq API key — required to generate answers |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model used for generation |
| `CHUNK_SIZE` | 700 | Target chunk size in tokens |
| `CHUNK_OVERLAP` | 100 | Token overlap between adjacent chunks |
| `BM25_TOP_K` | 10 | Keyword candidates retrieved |
| `VECTOR_TOP_K` | 10 | Semantic candidates retrieved |
| `RERANK_TOP_K` | 5 | Chunks kept after reranking |

Invalid configurations (e.g. `CHUNK_OVERLAP` larger than `CHUNK_SIZE`) are
rejected at startup rather than failing later at runtime.

---

## Project structure

```
ask-my-docs/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI route handlers
│   │   ├── core/         # Config and settings
│   │   ├── ingestion/    # Extract, chunk, embed, index
│   │   ├── retrieval/    # Hybrid search + reranking
│   │   └── graph/        # LangGraph RAG agent
│   ├── eval/             # RAGAS evaluation
│   └── tests/
├── frontend/             # React + Vite app
├── docs/                 # Learning notes
└── .github/workflows/    # CI pipelines
```

## Documentation

Beyond this README, `LEARNING.md` is a living learning log written during the
build — it explains every component and the reasoning behind each decision.

---

## License

MIT
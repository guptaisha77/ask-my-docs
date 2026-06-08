# Ask My Docs — Learning Notes

> A personal learning log for the `ask-my-docs` project. Written so that future-me
> (or anyone new to Python and AI) can revisit any part of the codebase and
> understand not just *what* the code does, but *why* it's built that way.
>
> This document grows as the project grows. Each phase gets its own section.
>

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Key concepts glossary](#2-key-concepts-glossary)
3. [Project structure](#3-project-structure)
4. [Git workflow we use](#4-git-workflow-we-use)
5. [Phase 1 — Tooling and environment](#5-phase-1--tooling-and-environment)
6. [Phase 2 — The config file, line by line](#6-phase-2--the-config-file-line-by-line)
7. [How Pydantic validates settings at startup](#7-how-pydantic-validates-settings-at-startup)
8. [Why we chose Groq](#8-why-we-chose-groq)
9. [Commands cheat sheet](#9-commands-cheat-sheet)
10. [Glossary of "scary" Python syntax](#10-glossary-of-scary-python-syntax)

---

## 1. What this project is

**Ask My Docs** is a question-answering system over your own documents. You upload
a set of documents (PDFs, text files), ask questions in plain English, and get
answers that **cite the exact source passages** they came from.

### The core problem it solves

Imagine you have 50 PDFs and you ask: *"What is the payment notice period in the
supplier contract?"* The system must find the one sentence that answers that —
among potentially 100,000 sentences — and answer with **proof** it didn't make
the answer up.

### Why naive approaches fail (and how we fix each)

| Problem with naive RAG | Our fix |
|---|---|
| Vector search misses exact terms (codes, clause numbers) | **Hybrid retrieval** — also use BM25 keyword search |
| Answers can't be trusted without sources | **Citation enforcement** — every answer points to a source |
| No way to know if a change improved or hurt quality | **CI-gated evaluation** — RAGAS scores block bad merges |

### The five layers

1. **Frontend** (React) — upload documents, ask questions, see citations
2. **Ingestion** — turn documents into searchable pieces
3. **Retrieval + reranking** — find the best pieces for a question
4. **Generation** — an AI writes the answer using those pieces, with citations
5. **Evaluation / CI** — automatically measure answer quality on every change

---

## 2. Key concepts glossary

These are the big ideas the whole project rests on. Read this once; refer back as needed.

### RAG (Retrieval-Augmented Generation)
Instead of asking an AI to answer from memory (where it might make things up),
we first **retrieve** relevant documents, then ask the AI to **generate** an answer
**using only those documents**. "Augmented" = the AI's answer is augmented with
real source material.

### Token
The smallest unit an AI language model reads. Not a word, not a letter — a
*subword piece*. Rough rule: **1 token ≈ 0.75 words ≈ 4 characters**.
Example: "unbelievable" might be 3 tokens (`un` + `believ` + `able`).
Why it matters: AI models have limits measured in tokens, and we size our
document pieces in tokens.

### Chunk
A piece of a document. We split big documents into chunks of ~700 tokens each,
because feeding a whole document to the AI is too much, and feeding single
sentences is too little. 700 tokens ≈ one dense page.

### Chunk overlap
When we split a document, each chunk shares ~100 tokens with the next one.
This stops us losing meaning at the boundary between two chunks. (If a sentence's
context is in the previous chunk, the overlap carries it forward.)

### Embedding / vector
A way to turn text into a list of numbers that captures its *meaning*. Texts with
similar meanings get similar numbers. This lets the computer find "semantically
similar" text even when the exact words differ ("payment" ≈ "invoice").

### Vector search (semantic search)
Searching by *meaning* using embeddings. Good at "what's the refund policy?"
matching text about "money-back guarantees."

### BM25 (keyword search)
The classic search method (used in Google-era search engines). Searches by
*exact words*. Good at finding "Section 4.2" or "API_KEY" — exact strings a
meaning-based search might miss.

### Hybrid retrieval
Using **both** vector search and BM25 together, then combining the results.
Catches both meaning-matches and exact-word-matches.

### Reranking (cross-encoder)
After getting a pile of candidate chunks, a more powerful AI model reads the
question and each chunk *together* and re-scores them, pushing the genuinely
relevant ones to the top. Slower but more accurate than the first search pass —
so we only run it on a small candidate set.

### Citation enforcement
Making sure every claim in the answer points back to a real source passage,
so answers are auditable instead of trusted blindly.

### RAGAS
An evaluation framework that uses AI to score our system's answers on things like
*faithfulness* (did the answer stick to the sources?) and *answer relevancy*
(did it actually address the question?).

### CI (Continuous Integration)
Automation that runs checks every time you propose a code change. Ours runs the
RAGAS evaluation and **blocks the change** if answer quality drops below a threshold.

---

## 3. Project structure

```
ask-my-docs/
├── backend/                      # The Python server (FastAPI)
│   ├── app/
│   │   ├── api/                  # Web endpoints (URLs the frontend calls)
│   │   ├── core/                 # Config and settings  ← we are here
│   │   │   └── config.py         # The control panel for the whole app
│   │   ├── ingestion/            # Extract, chunk, embed, index documents
│   │   ├── retrieval/            # Hybrid search + reranking
│   │   └── graph/                # The LangGraph AI agent
│   ├── eval/                     # RAGAS evaluation
│   ├── tests/                    # Automated tests
│   ├── .env                      # YOUR REAL SECRETS (never committed to git)
│   ├── .env.example              # A template of .env (safe to commit)
│   └── requirements.txt          # List of Python packages we need
├── frontend/                     # The React app (the user interface)
├── .github/workflows/            # CI pipeline definitions
├── README.md                     # Project overview for visitors
└── LEARNING.md                   # This file
```

### Why folders matter in Python
Each folder under `app/` that contains code has an empty `__init__.py` file.
That empty file is what tells Python "this folder is a *package* you can import
code from." Without it, `from app.core.config import settings` wouldn't work.

---

## 4. Git workflow we use

We use a professional branching model so the project history looks like real
team development.

```
main      ← stable, release-ready code only
  └── dev          ← integration branch; features merge here first
        └── feature/config-settings   ← one branch per piece of work
        └── feature/ingestion
        └── docs/readme
```

### The rules
- **Never commit directly to `main`.** It stays clean and releasable.
- **`dev`** is where finished features come together.
- Each new piece of work gets its own **`feature/...`** branch, made *from* `dev`.
- When a feature is done, open a **Pull Request (PR)** from the feature branch
  into `dev`. Even working solo, this builds good habits and a reviewable history.
- Periodically, `dev` is merged into `main` as a release (e.g. `v0.1`).

### Commit message style (Conventional Commits)
Format: `type(scope): short summary`

Common types:
- `feat` — a new feature
- `fix` — a bug fix
- `docs` — documentation only
- `chore` — maintenance (deps, configs)
- `refactor` — restructuring code without changing behaviour
- `test` — adding or fixing tests
- `ci` — CI pipeline changes

Example:
```
feat(config): switch LLM provider from OpenAI to Groq
```

---

## 5. Phase 1 — Tooling and environment

### What each tool is

| Tool | What it is | Why we use it |
|---|---|---|
| **Python** | The backend programming language | AI/ML ecosystem lives here |
| **FastAPI** | A Python web framework | Fast, auto-generates API docs, validates data |
| **Uvicorn** | The server that runs FastAPI | Actually serves the app over HTTP |
| **virtual environment (`.venv`)** | An isolated Python install for this project | Keeps this project's packages separate from others |
| **pydantic** | A data-validation library | Checks our settings are valid |
| **pydantic-settings** | Loads config from files/env vars | Powers our `config.py` |
| **React** | Frontend JavaScript library | Builds the user interface |
| **Vite** | Frontend build tool | Fast dev server and bundler |
| **Bun** | JavaScript runtime + package manager | Faster than npm |
| **TanStack Query** | Data-fetching library for React | Handles loading/error/caching for API calls |
| **Biome** | Linter + formatter in one | Keeps frontend code clean and consistent |
| **ChromaDB** | A vector database | Stores embeddings for semantic search |
| **Groq** | A fast AI model provider | Runs Llama models very fast, free-friendly |

### The virtual environment, explained
A **virtual environment** is a private, isolated copy of Python just for this
project. When you "activate" it, the `python` and `pip` commands in your terminal
use *this project's* Python, not your computer's system-wide one. This means
installing a package here can never break another project.

You can tell it's active because your terminal prompt shows `(.venv)` at the start.

```bash
# Create it (once)
python -m venv .venv
# Activate it (every new terminal)
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
```

---

## 6. Phase 2 — The config file, line by line

> File: `backend/app/core/config.py`
> Purpose: the single **control panel** for the whole app — every secret and
> tunable lives here, validated at startup.

### The mental model
`config.py` is a *form* that gets filled in from three sources, in priority order:

1. **OS environment variables** (used in production, CI, Docker) — highest priority
2. **The `.env` file** (used on your laptop for local development)
3. **The defaults written in the code** (fallback if nothing else is set)

It takes the first value it finds and stops looking. This is why the *same code*
runs on your laptop and on a real server — production just sets environment
variables to override the `.env` file, with no code change.

### The file's docstring
```python
"""Application configuration.
..."""
```
Triple quotes `"""` create a **docstring** — a human note Python ignores when
running. It's the label on the folder.

### The imports
```python
from functools import lru_cache
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
```
`import` = "borrow a tool someone already built." We borrow:
- `lru_cache` — remembers an answer so work isn't repeated (used at the end)
- `Field` — lets us attach rules to a setting
- `field_validator` / `model_validator` — our two "bouncers" that inspect values
- `BaseSettings` — the foundation our control panel is built on
- `SettingsConfigDict` — configures how the panel loads values

### Starting the class
```python
class Settings(BaseSettings):
```
`class` defines our own custom thing called `Settings`. `(BaseSettings)` means
"build on top of the BaseSettings foundation" — that foundation gives us the
magic of auto-reading from `.env`. The colon `:` and indentation show what's
*inside* the class.

### The load configuration
```python
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
```
The *settings for the settings*:
- `env_file=".env"` — also read values from a file called `.env`
- `env_file_encoding="utf-8"` — that file is standard text
- `extra="ignore"` — skip unrecognised lines in `.env` instead of crashing
- `case_sensitive=False` — `GROQ_API_KEY` and `groq_api_key` are treated the same

### The settings themselves

Every setting follows the same pattern:
```
name: type = Field(default=..., optional_rules)
```

**Application metadata**
```python
    app_name: str = "ask-my-docs"
    environment: str = Field(default="development")
```
- `app_name` — a plain text label (`str` = text/string)
- `environment` — text wrapped in `Field` so we can add a rule (a bouncer) later;
  records whether we're in development or production

**LLM provider (Groq)**
```python
    groq_api_key: str = Field(default="")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")
    groq_model: str = Field(default="llama-3.1-8b-instant")
```
- `groq_api_key` — your secret key. Default is `""` (blank) so the app starts
  even before you add the real key (filled from `.env`).
- `groq_base_url` — the *address* requests are sent to. Like a postal address.
- `groq_model` — which AI model to use. `llama-3.1-8b-instant` is fast and free-friendly.

**Reranker (Cohere)**
```python
    cohere_api_key: str = Field(default="")
    cohere_rerank_model: str = Field(default="rerank-english-v3.0")
```
A blank-by-default secret key and the name of the reranking model.

**Vector store (ChromaDB)**
```python
    chroma_persist_dir: str = Field(default="./chroma_db")
    chroma_collection: str = Field(default="ask_my_docs")
```
- `chroma_persist_dir` — folder on disk where the database saves data
  (`./` = "right here next to where I run the app")
- `chroma_collection` — name of the storage area inside the database (like a table name)

**Embedding model**
```python
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
```
The model that turns text into meaning-numbers. Runs locally, no key needed.

**Chunking (now with rules)**
```python
    chunk_size: int = Field(default=700, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
```
- `int` = whole number
- `gt=0` = "greater than 0" (must be positive)
- `ge=0` = "greater than or equal to 0" (can be zero, never negative)
- These rules are checked at startup; break one and the app refuses to start.

**BM25 index path**
```python
    bm25_index_path: str = Field(default="./bm25_index.pkl")
```
A file location where the keyword search saves its index.

**Retrieval knobs**
```python
    bm25_top_k: int = Field(default=10, gt=0)
    vector_top_k: int = Field(default=10, gt=0)
    rerank_top_k: int = Field(default=5, gt=0)
```
Three positive whole numbers controlling how many results each search step fetches.
We fetch ~10+10 candidates, then the reranker trims to the best 5.

**Eval thresholds (decimals with a range)**
```python
    eval_faithfulness_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    eval_answer_relevancy_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    eval_context_precision_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
```
- `float` = decimal number
- `ge=0.0` and `le=1.0` together = "must be between 0 and 1" (`le` = "less than or equal")
- These are quality scores, always a fraction from 0 to 1.

**CORS origins**
```python
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )
```
- `list[str]` = a list of text items (square brackets `[ ]` hold multiple values)
- The web addresses allowed to talk to our API (the frontend's dev addresses)

### The two "bouncers" (validators)

**Field validator — checks ONE setting**
```python
    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, value: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        normalised = value.lower().strip()
        if normalised not in allowed:
            raise ValueError(
                f"environment must be one of {sorted(allowed)}, got '{value}'"
            )
        return normalised
```
A bouncer who checks one fact about one person.
- `@field_validator("environment")` — attaches this checker to the `environment` setting
- `value` — whatever someone tried to set it to
- `value.lower().strip()` — cleans the input (lowercase, no spaces) so `"Production "` → `"production"`
- `if normalised not in allowed:` — if it's not on the guest list...
- `raise ValueError(...)` — ...stop and complain with a clear message
- `return normalised` — otherwise hand back the cleaned value (it *replaces* the original)

**Model validator — checks how settings RELATE**
```python
    @model_validator(mode="after")
    def _validate_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(...)
        max_candidates = self.bm25_top_k + self.vector_top_k
        if self.rerank_top_k > max_candidates:
            raise ValueError(...)
        if self.is_production and not self.has_groq_key:
            raise ValueError("GROQ_API_KEY is required when environment=production")
        return self
```
A bouncer who checks how *two* things fit together (like "kid needs an adult").
- `mode="after"` — runs *after* all individual settings are filled in, so it can
  see the whole object
- `self` — the whole filled-in control panel; reach any setting via `self.xxx`
- First check: overlap must be smaller than chunk size (can't overlap more than the whole chunk)
- Second check: can't ask the reranker for more results than we retrieved
- Third check (**fail fast**): in production, a missing Groq key stops the app at
  startup rather than failing on the first user request. In development it's fine
  to run keyless. This is the difference between dev convenience and prod safety.
- A field validator *cannot* do any of these — it only ever sees one setting at a time.
- Note it's named `_validate_settings` (not `_validate_chunking`) because it now
  checks more than chunking.

### The convenience shortcuts
```python
    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key.strip())

    @property
    def is_production(self) -> bool:
        return self.environment == "production"
```
`@property` lets you ask a question as if the answer were a setting. `bool` = yes/no.
- `has_groq_key` — "do we have a real key?" — asked everywhere instead of checking
  the raw string in many places (single source of truth)
- `is_production` — "are we on the real server?"

### The final lines (OUTSIDE the class — no indentation)
```python
@lru_cache
def get_settings() -> Settings:
    """..."""
    return Settings()


settings = get_settings()
```
- `def get_settings()` — a function whose only job is to build the panel by calling
  `Settings()`. That call is when values get gathered and bouncers run.
- `@lru_cache` — "do this once and remember the answer," so `.env` is read a single
  time no matter how many files ask for settings.
- `settings = get_settings()` — actually runs it and stores the finished panel.
  Other files just do `from app.core.config import settings` to borrow it.

### The takeaway pattern
Most lines are just: `name: type = Field(default=..., maybe_a_rule)`.
Learn to read one and you can read almost the whole file. The only "special"
parts are the two bouncers and the three lines at the very end.

---

## 7. How Pydantic validates settings at startup

When does validation run? When the object is *constructed* — the last line,
`settings = get_settings()`. Because that's at module level, it runs on first
import (when the app boots). If anything is wrong, the import throws and the
**server never starts** — which is the whole point.

The validation runs in **5 stages, in this order**:

| Stage | What happens | Example failure |
|---|---|---|
| 1. Gather sources | Look in env vars, then `.env`, then defaults | (no failure — just collecting) |
| 2. Coerce types | Convert text from env into int/float/list | `CHUNK_SIZE=abc` can't become a number |
| 3. Field constraints | Check `gt`, `ge`, `le` rules | `CHUNK_SIZE=-5` fails `gt=0` |
| 4. Field validators | Run per-field checkers (`@field_validator`) | `ENVIRONMENT=prod` not in allowed set |
| 5. Model validator | Run cross-field checker (`@model_validator`) | `CHUNK_OVERLAP=999` ≥ chunk_size |

### Key insight about Stage 2 (coercion)
**Everything from an environment variable arrives as TEXT.** When you write
`CHUNK_SIZE=700` in `.env`, Pydantic receives the *string* `"700"`, not the number
700. Because the field is typed `int`, Pydantic *converts* it. If conversion fails
(e.g. `"abc"`), it errors right here before any of your logic runs.

### Why the order matters
Each stage assumes the previous passed. By the time the model validator (stage 5)
does `self.bm25_top_k + self.vector_top_k`, both are guaranteed to be valid
positive integers — so the arithmetic is safe.

### Try breaking it on purpose (great way to learn)
```bash
cd backend
CHUNK_SIZE=abc python -c "from app.core.config import settings"     # stage 2 fails
CHUNK_SIZE=-5  python -c "from app.core.config import settings"     # stage 3 fails
ENVIRONMENT=prod python -c "from app.core.config import settings"   # stage 4 fails
CHUNK_OVERLAP=999 python -c "from app.core.config import settings"  # stage 5 fails
```
The last one is the most instructive: `999` is a perfectly valid positive integer,
so it passes stages 2–4. Only the model validator can catch that the *combination*
with `chunk_size=700` is invalid.

---

## 8. Why we chose Groq

**Groq** is a company that runs open-source AI models (like Meta's Llama) on
special hardware that's extremely fast. The clever part: Groq's API is
**OpenAI-compatible**. That means we use the normal OpenAI Python library but point
it at Groq's address with a Groq key.

> Analogy: OpenAI and Groq are two coffee shops that take orders in the *exact same
> format*. You don't learn a new way to order — you just walk into a different door
> and pay with a different card.

### What switching to Groq actually requires
Just three settings:
1. A different API key (`groq_api_key`, starts with `gsk_`, from console.groq.com)
2. A different address (`groq_base_url` = `https://api.groq.com/openai/v1`)
3. A different model name (`groq_model` = `llama-3.1-8b-instant`)

### How it'll be used later (preview)
```python
from openai import OpenAI
from app.core.config import settings

client = OpenAI(
    api_key=settings.groq_api_key,    # the gsk_ key
    base_url=settings.groq_base_url,  # points the library at Groq
)
response = client.chat.completions.create(
    model=settings.groq_model,        # llama-3.1-8b-instant
    messages=[{"role": "user", "content": "Hello"}],
)
```
Note it's the **OpenAI** client — same library — just handed Groq's key and address.
That's the whole trick. The three config fields line up exactly with the three
things this code needs.

### Model options on Groq
- `llama-3.1-8b-instant` — fast, free-friendly (our default, good for portfolio)
- `llama-3.3-70b-versatile` — smarter, slower
- `openai/gpt-oss-120b` — a stronger option Groq added later

---

## 9. Commands cheat sheet

### Environment
```bash
cd backend
python -m venv .venv                 # create virtual env (once)
source .venv/bin/activate            # activate it (every new terminal)
pip install -r requirements.txt      # install all dependencies
pip freeze > requirements.txt        # save exact versions after installing something new
```

### Running and testing config
```bash
# Print some settings to confirm config loads
python -c "from app.core.config import settings; print(settings.app_name, settings.groq_model)"

# Deliberately break a setting to see validation fire
CHUNK_OVERLAP=999 python -c "from app.core.config import settings"
```

### Git
```bash
git checkout dev                          # switch to dev branch
git checkout -b feature/my-thing          # create a new feature branch from current
git add path/to/file.py                   # stage a specific file
git commit -m "feat(scope): summary"      # commit with a conventional message
git push -u origin feature/my-thing       # push the branch to GitHub
```

---

## 10. Glossary of "scary" Python syntax

Quick reference for syntax that looks intimidating but is simple once named.

| Syntax | Name | What it means |
|---|---|---|
| `"""..."""` | docstring | A multi-line note for humans; Python ignores it |
| `# ...` | comment | A single-line note for humans; Python ignores it |
| `name: str` | type hint | "This value should be text" |
| `name: int` | type hint | "This value should be a whole number" |
| `name: float` | type hint | "This value should be a decimal number" |
| `name: bool` | type hint | "This value is yes/no (True/False)" |
| `list[str]` | type hint | "A list of text items" |
| `= Field(default=...)` | default value | The value used if nothing overrides it |
| `gt=0` | constraint | "greater than 0" |
| `ge=0` | constraint | "greater than or equal to 0" |
| `le=1.0` | constraint | "less than or equal to 1.0" |
| `@something` | decorator | A special attachment that adds behaviour to the thing below it |
| `@property` | decorator | Lets a function be accessed like a setting (no parentheses) |
| `class Foo(Bar):` | class definition | Define a custom thing `Foo` built on top of `Bar` |
| `def foo(...):` | function definition | Define a reusable block of code called `foo` |
| `self` | parameter | "this particular object" (the filled-in instance) |
| `cls` | parameter | "the class itself" (used in classmethods) |
| `raise ValueError(...)` | error | Stop and report a problem |
| `return x` | return | Hand a value back out of a function |
| `f"... {x} ..."` | f-string | Text with values slotted in via `{ }` |
| `bool(x)` | conversion | Turn `x` into a yes/no value |
| `.strip()` | string method | Remove spaces from the start and end of text |
| `.lower()` | string method | Make text lowercase |
| `from a import b` | import | Borrow tool `b` from module `a` |

---

## Progress log

| Date | What we did |
|---|---|
| (fill in) | Set up project scaffold, venv, git branching model |
| (fill in) | Wrote and understood `config.py` with Pydantic validation |
| (fill in) | Switched LLM provider to Groq |
| (fill in) | Created README with architecture diagram |
| (fill in) | Created this learning document |

---

*Next up: the ingestion pipeline — extracting text from documents, then chunking
it into ~700-token pieces. This section will be added when we build it.*
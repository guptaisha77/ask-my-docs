# Ask My Docs — Learning Notes

> A personal learning log for the `ask-my-docs` project. Written so that future-me
> (or anyone new to Python and AI) can revisit any part of the codebase and
> understand not just *what* the code does, but *why* it's built that way.
>
> This document grows as the project grows. Each phase gets its own section.

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
11. [Tokens and chunking — the deep concept](#11-tokens-and-chunking--the-deep-concept)
12. [Embeddings and vector search — the deep concept](#12-embeddings-and-vector-search--the-deep-concept)
13. [BM25 keyword search — the deep concept](#13-bm25-keyword-search--the-deep-concept)
14. [Planned: monitoring & observability](#14-planned-monitoring--observability)

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

## 11. Tokens and chunking — the deep concept

> This is the most important concept in the whole project. Understanding it is
> what separates "I followed a RAG tutorial" from "I understand RAG."

### What a token actually is
An AI model doesn't read letters or words — it reads **tokens**: pieces from a
fixed vocabulary of ~100,000 known chunks. A token can be a whole word, a
word-fragment, a single character, or punctuation. A leading space is usually
part of the token.

Real examples (using the cl100k_base tokenizer):

| Text | Tokens | Count |
|---|---|---|
| `cat` | `cat` | 1 |
| `unbelievable` | `un` · `bel` · `iev` · `able` | 4 |
| `ChromaDB` | `Ch` · `roma` · `DB` | 3 |
| `the dog` | `the` · ` dog` | 2 |
| `4.2(b)` | `4` · `.` · `2` · `(` · `b` · `)` | 6 |

Rule of thumb: **1 token ≈ 0.75 words ≈ 4 characters** — but only on average.

#### Diagram: one sentence becoming tokens

```
"Renewal requires 30 days notice."

 ┌─────────┐┌──────────┐┌────┐┌──────┐┌────────┐┌───┐
 │ Renewal ││ requires ││ 30 ││ days ││ notice ││ . │   = 6 tokens
 └─────────┘└──────────┘└────┘└──────┘└────────┘└───┘
   note: the space attaches to the next word ("·requires"),
   and the full stop is its own token.
```

### Why we measure in tokens, not characters
The token count is **unpredictable from the character count**. Dense technical
text (clause numbers, codes, jargon) packs more tokens into the same characters
than plain prose. Example: 1000 characters of legal text ≈ 350 tokens, but 1000
characters of a simple story ≈ 200 tokens.

**This is why we can't split "every 3000 characters"** — we'd get chunks that are
sometimes 400 tokens and sometimes 800. We use the `tiktoken` library to count
tokens *exactly*.

#### Diagram: same characters, different token counts

```
Dense / technical text (clause 4.2(b), GSTIN, codes…)
 ┌────────────────────────────────────────┐
 │ 1000 characters                         │  ≈ 350 tokens   (more tokens)
 └────────────────────────────────────────┘

Plain prose (a simple story)
 ┌────────────────────────────────────────┐
 │ 1000 characters                         │  ≈ 200 tokens   (fewer tokens)
 └────────────────────────────────────────┘

  Same width on the page, very different token counts.
  → fixed-character chunks would be wildly inconsistent in token size.
```

### Why chunk at all (three reasons)
1. **Hard limit:** a whole document is too many tokens to fit in the model's
   context window.
2. **Cost and speed:** you pay per token; sending an entire document for every
   question wastes money and time when 99% is irrelevant.
3. **Focus improves answers:** a single relevant 700-token passage produces a
   better answer than 50,000 tokens where the answer is one buried sentence.
   Less but more relevant context = better answers, not just cheaper ones.

So: chop the document into pieces *once* at upload time, then at question time
retrieve only the few pieces that matter.

### Why ~700 tokens
A size dial, with a tradeoff at each end:
- **Too small** (one sentence): laser-focused but loses context. "This must be
  submitted within 30 days" is useless if you don't know what *this* is.
- **Too large** (whole chapter): full of context but the answer is buried in
  irrelevant text, hurting focus and cost.
- **Sweet spot:** 500–800 tokens ≈ one dense page. We picked **700** as a solid
  middle. It's an empirically reliable default you'd *tune* per document type.

### Why overlap (~100 tokens)
If we cut cleanly at 700 tokens, a cut can land mid-paragraph and orphan context:

> *...renewals are handled by the vendor portal.* **[CUT]** *The portal closes at
> 5pm on the last business day.*

Someone asking "when does the renewal portal close?" gets chunk 2, which never
says *which* portal or that it's about *renewals* — that was in chunk 1.

**Fix:** each chunk shares its last ~100 tokens with the start of the next, so
chunk 2 begins with the tail of chunk 1 and is self-contained. Overlap is a
safety margin that stops meaning falling through the cracks at boundaries.
Tradeoff: more overlap = safer context but more chunks to store.

#### Diagram: the sliding window with overlap

```
The document is one long stream of tokens:
 token 0 ───────────────────────────────────────────────► token N

We slide a 700-token window forward, but step back 100 each time so
windows overlap:

 Chunk 1:  [ tokens 0 ──────────────── 699 ]
 Chunk 2:              [ 600 ──────────────── 1299 ]
 Chunk 3:                          [ 1200 ──────────────── 1899 ]
                        ╰──┬──╯              ╰──┬──╯
                       100 shared           100 shared

 The shared region means a sentence split across a boundary keeps its
 context: the tail of chunk 1 reappears at the start of chunk 2.
```

### Why split on sentences first
If we *only* counted tokens we'd sometimes slice mid-sentence (or mid-word),
producing meaningless fragments. So the chunker splits into **sentences** first,
then groups whole sentences into ~700-token windows. Every chunk ends on a clean
sentence boundary — we never cut mid-thought.

### The complete mental model
**Split into sentences → group sentences into ~700-token windows → overlap each
window with the next by ~100 tokens → never cut mid-sentence.**

#### Diagram: the chunker pipeline

```
  raw document text
        │
        ▼
  ┌──────────────────┐   split on . ! ? so we never cut mid-sentence
  │ split into        │
  │ sentences         │
  └──────────────────┘
        │  [s1, s2, s3, s4, ...]
        ▼
  ┌──────────────────┐   add whole sentences until the next one would
  │ group into        │   push the window over ~700 tokens
  │ 700-token windows │
  └──────────────────┘
        │
        ▼
  ┌──────────────────┐   seed each new window with the last ~100 tokens
  │ apply 100-token   │   of the previous one
  │ overlap           │
  └──────────────────┘
        │
        ▼
  list of Chunk objects  (each: text + token_count + index + metadata)
```

### In one paragraph: why not just split every 3000 characters?
Because token count isn't proportional to character count — dense text has more
tokens per character than prose — so fixed-character chunks would be wildly
inconsistent in token size. We measure tokens directly with tiktoken, split on
sentence boundaries so chunks stay coherent, and overlap them so context isn't
lost where one chunk ends and the next begins.

---

## 12. Embeddings and vector search — the deep concept

> The other half of "how does search-by-meaning work." Tokens + chunking prepare
> the text; embeddings make it *searchable by meaning*.

### The core problem
A computer can compare numbers (`5` vs `7`) and check if two strings are
identical (`"cat" == "cat"`). But it has no built-in way to know that `"dog"` and
`"puppy"` are *related in meaning* while `"dog"` and `"fridge"` are not — they
share no letters that would reveal it.

An **embedding** turns each piece of text into a list of numbers, arranged so that
**texts with similar meanings get similar numbers.** Once meaning is expressed as
numbers, the computer can do math on it: "how related are these two texts?"
becomes "how close are these two lists of numbers?" — answered instantly.

### A toy example: 2 numbers per word
Describe each word with just two numbers — how animal-like (0–1) and how large (0–1):

| Word | animal-ness | largeness |
|---|---|---|
| puppy | 0.9 | 0.2 |
| dog | 0.9 | 0.4 |
| elephant | 0.9 | 0.95 |
| pebble | 0.05 | 0.1 |

Plot them as points (animal-ness across, largeness up):

```
 largeness
   ▲
   │                                   ● elephant (0.9, 0.95)
   │
   │
   │                                   ● dog   (0.9, 0.4)
   │                                   ● puppy (0.9, 0.2)   ← puppy & dog
   │  ● pebble (0.05, 0.1)                                    sit very close
   └─────────────────────────────────────────────────►  animal-ness
```

`puppy` and `dog` land right next to each other (both very animal, both smallish).
`elephant` is directly above `dog` (equally animal, much larger). `pebble` sits
far away in the corner. **Distance between points now encodes how related the
words are.** That is the entire trick.

### The real thing: 384 numbers, not 2
A real model uses **384 numbers** per text (that's what `all-MiniLM-L6-v2`, our
model, produces). You can't picture 384-dimensional space — nobody can — but the
*math* works exactly like the 2D picture: each text is a point, and closeness
still means similar meaning.

The model **learned its 384 "qualities" on its own** by reading huge amounts of
text during training. Nobody hand-picked them; no human fully knows what dimension
#207 means. But collectively the 384 numbers capture meaning well enough that
"payment terms" and "invoice settlement period" land close together — even with
no shared words. That is why semantic search works.

### Measuring closeness: cosine similarity
How does the computer measure closeness between two 384-number lists? The standard
for text is **cosine similarity**: imagine an arrow from the origin to each text's
point, and measure the **angle between the two arrows** (not the distance between
their tips).

Why angle, not distance? It makes the comparison fair regardless of length. A
short sentence and a long paragraph about the same topic have arrows of different
*lengths* but pointing the same *direction* — a small angle. Cosine captures "same
direction = same topic" and ignores length.

```
   short text  ──►
                  ╲  small angle  →  very similar meaning
   long text   ────►

   text A   ──►
              │  90° angle      →  unrelated
   text B     ▼
```

Score runs from -1 to 1: near **1** = almost same direction (similar meaning),
near **0** = unrelated, negative = opposite. For text you mostly see 0 to 1.

### Where ChromaDB fits
**ChromaDB** is a *vector database* — its job is to hold thousands of these
vectors and, given a query vector, quickly find the stored ones at the smallest
angle to it.

- A **normal database** finds *exact* matches ("the row where id = 42").
- A **vector database** finds *nearest* matches ("the 5 vectors closest in
  direction to this one"). Different kind of search → specialised tool.

The convenient part: ChromaDB does the embedding *for* us. We hand it the chunk
text and tell it which embedding model to use; it runs the model, stores the
vector, and keeps the original text + metadata alongside. At query time we hand it
a question, it embeds that too, finds the nearest chunks, and returns their text.
We never juggle the 384 numbers by hand.

### The complete mental model
**Text → embedding model → a point in 384-dimensional space; similar meanings land
close together; ChromaDB stores those points and finds the nearest ones to your
question by comparing angles (cosine similarity).**

### In one paragraph: how can it match with no shared words?
*How can semantic search find 'invoice settlement period' when you searched for
'payment terms', with no shared words?*
Because we don't match words — we match *meaning*. An embedding model maps each
piece of text to a point in a high-dimensional space (384 dims for MiniLM) where
texts about the same concept land close together, regardless of vocabulary. The
model learned those dimensions from huge text corpora. At query time we embed the
question into the same space and return the chunks whose vectors point most nearly
the same direction (highest cosine similarity) — so conceptually-related text is
found even when the exact words differ. This is exactly why we *also* keep BM25
keyword search: it catches the exact-string cases (codes, clause numbers) where
matching meaning isn't enough.

---

## 13. BM25 keyword search — the deep concept

> The keyword half of hybrid retrieval. Embeddings match meaning; BM25 matches
> exact words. Together they cover each other's blind spots.

### Why you need it (vector search has a blind spot)
Vector search finds by *meaning* — great for "when do I pay?" matching "payment is
due." But it has *no meaningful embedding* for exact strings it has never seen: a
clause like "Section 4.2(b)", a product code "SKU-99317", an acronym. BM25 finds
those instantly, because it searches by *exact words*, not meaning.

The two have **complementary blind spots**: vector misses exact strings, BM25
misses paraphrases. Run both and combine — neither's gap can sink an answer. That
complementarity is the entire reason "hybrid retrieval" exists.

### What BM25 is
BM25 ("Best Match 25") is the classic keyword-ranking algorithm behind search
engines like Elasticsearch and Lucene. It scores each chunk against the query with
a number; higher = better. It's smarter than "count matching words" — it balances
three ideas:

1. **Term frequency, with diminishing returns** — more mentions of a query word =
   more relevant, but the 10th mention adds far less than the 2nd. Stops a chunk
   winning just by repeating a word.
2. **Inverse document frequency (rare words matter more)** — common words like
   "the" appear everywhere and carry no signal, so they're down-weighted; rare,
   distinctive words like "payment" are up-weighted. It judges rarity by how many
   chunks across the whole collection contain the word.
3. **Length normalisation** — long chunks naturally contain more words, so they'd
   unfairly match more. BM25 corrects for length so a tight short chunk can beat a
   rambling long one.

So: BM25 scores a chunk high when it contains your *rare, distinctive* query words,
*several times*, without being *padded out* with length.

### How it works mechanically (this shapes the code)
BM25 needs to know how common each word is *across the whole collection* (for the
rarity idea). So it builds an **index**: tokenise every chunk into plain words,
pre-compute the statistics. Two consequences:

- **It works on words, not the 384-number vectors.** Lowercase, split on
  spaces/punctuation. No model, no GPU, no API — pure counting and arithmetic,
  which is why it's fast and free.
- **The index depends on the whole collection.** Because rarity is computed across
  all chunks, adding new chunks changes the statistics, so the index is rebuilt on
  add. Instant at our scale; at massive scale this is exactly why companies use
  Elasticsearch instead.

### Where it lives
No database needed. We use `rank-bm25` (a small pure-Python library), build the
index in memory, and `pickle` it to disk so it survives restarts. Shape: tokenise
chunks → build index → save to disk → load back when searching.

### A real gotcha (learned by running it)
BM25 scores depend on collection statistics, so on a *tiny* corpus the
inverse-document-frequency term can go **negative** — a word in your single
document counts as appearing "everywhere," which the formula treats as
uninformative. It only behaves sensibly with a realistic number of documents.
Good reminder that BM25 is a collection-level algorithm, not per-document.

### Why rank-bm25 and not Elasticsearch
Not cost — Elasticsearch has a free self-hosted tier. It's that Elasticsearch is a
whole separate server to run and manage, while rank-bm25 is a zero-infrastructure
library, keeping the project self-contained (clone-and-run). The tradeoff:
rank-bm25 holds the index in memory and rebuilds on add — fine for thousands of
chunks, breaks at millions. At production scale, move to Elasticsearch/OpenSearch
for incremental indexing and distributed search. Simple option chosen
deliberately, knowing where it stops scaling.

### The complete mental model
**BM25 scores chunks by exact-word overlap, weighting rare query words higher,
rewarding repeated mentions with diminishing returns, and correcting for length.
It complements vector search by catching exact strings meaning-search misses. Built
in memory from all chunks, pickled to disk.**

### In one paragraph: why isn't vector search alone enough?
*Why isn't vector search alone enough — what does BM25 catch that embeddings
miss?*
Vector search matches meaning but has no useful embedding for exact strings it
never saw — clause numbers, product codes, acronyms. BM25 matches exact words, so
it catches precisely those. They have complementary blind spots, so I run both and
combine them; neither gap can sink an answer. BM25 is pure counting — no model, no
cost — which is why it pairs so naturally with semantic search.

---

## 14. Planned: monitoring & observability

> **Status: planned, not built yet.** Documented here because it's the part of
> production AI work most portfolios ignore — and understanding *why* it matters
> is as valuable as the code itself.

### Why this matters
Most RAG projects stop at "it produces answers." But producing answers is maybe
30% of real production AI work. The other ~70% is *operating* the system:
knowing how fast it is, what it costs, and whether its quality is drifting over
time. A system you can't observe is a system you can't trust in production.

This is the difference between "I built a RAG demo" and "I built a RAG system I
could actually run and maintain."

### The five pieces

**1. Tracing**
Follow a single question through every stage — retrieve → rerank → generate —
and see how long each stage took and what it produced. Think of it like a receipt
that itemises where the time went. Tools: OpenTelemetry, or LangSmith (which pairs
naturally with LangGraph — each graph node becomes a traced step almost for free).

**2. Latency percentiles (p50 / p95)**
- **p50** (median): half of requests are faster than this, half slower.
- **p95**: 95% of requests are faster than this; it captures the slow tail.
Why p95 matters more than the average: the average hides bad experiences. "Most
users wait 800ms, but the slowest 5% wait 3+ seconds" is the insight that drives
real optimisation — and the average would never reveal it.

**3. Cost-per-request**
Track tokens sent + received per question, multiply by the model's price. Now you
know what each answer actually costs. Cheap to add because the system already
counts tokens (the chunker uses tiktoken).

**4. Production quality metrics**
The live cousin of the RAGAS CI eval. Sample real answers in production and score
them, so you catch quality *drift* after deployment — not just in CI before it.

**5. Regression gating in CI**
We already plan to gate merges on RAGAS quality scores. Observability extends the
gate: a change that makes the system slower or more expensive — not just lower
quality — also fails the build.

### How this influences the code we write NOW
Observability is hard to bolt on later and easy to design in early. From the
retrieval phase onward, the code is written with **observability seams**:
- functions return consistent shapes that can carry timing + token metadata
- each stage has a clean boundary where a timing/tracing wrapper can slot in
- token counts are already tracked, so cost tracking is "fill in the price"

This means adding observability later is *additive* (fill in the hooks), not a
rewrite. Designing for this from the start is itself a senior engineering habit.

### In one paragraph: why observability matters
Most RAG systems stop at "it produces answers," but ~70% of production AI work is
operating the system — latency, cost, and quality drift. The code is designed with
observability seams from the start, and the roadmap includes tracing, p50/p95
latency, cost-per-request, and CI gates on regressions, not just quality.

---

## Progress log

| Date | What we did |
|---|---|
| (fill in) | Set up project scaffold, venv, git branching model |
| (fill in) | Wrote and understood `config.py` with Pydantic validation |
| (fill in) | Switched LLM provider to Groq |
| (fill in) | Created README with architecture diagram |
| (fill in) | Created this learning document |
| (fill in) | Learned tokens and chunking concepts |
| (fill in) | Built the chunker |
| (fill in) | Learned embeddings and vector search concepts |
| (fill in) | Built the embedder (ChromaDB) |
| (fill in) | Learned BM25 concept and built the BM25 index |

---

*Next up: the hybrid retriever (combines vector + BM25 search), then the text
extractor and pipeline toward a working end-to-end demo.*
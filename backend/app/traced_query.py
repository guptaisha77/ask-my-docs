"""A traced end-to-end query, so the full pipeline appears as one trace in LangSmith.

@traceable turns each function into a span; nesting them produces a trace tree:
  answer_question
    ├── retrieve   (hybrid search)
    └── generate   (Groq call, auto-traced via wrap_openai)
"""

from langsmith import traceable

from app.graph.generator import generate_answer
from app.retrieval.hybrid import search


@traceable(run_type="retriever")
def retrieve(question: str) -> list[dict]:
    return search(question, top_k=5)


@traceable(run_type="chain")
def answer_question(question: str) -> dict:
    """Full traced query: retrieval + generation appear as nested spans."""
    chunks = retrieve(question)
    return generate_answer(question, chunks)

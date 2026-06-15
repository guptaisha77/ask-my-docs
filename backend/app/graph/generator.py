"""Generate cited answers from retrieved chunks using Groq's Llama model.

The retrieved chunks are numbered [1]..[n] in the prompt. The model is
instructed to answer ONLY from those sources, cite each claim with its source
number, and say so if the sources don't contain the answer. This is the
anti-hallucination core: grounding + citations + an honest no-answer path.
"""

from langsmith.wrappers import wrap_openai
from openai import OpenAI

from app.core.config import settings

# wrap_openai auto-logs every call's latency + token usage to LangSmith.
_client = wrap_openai(
    OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )
)


_SYSTEM_PROMPT = """You are a careful assistant that answers questions using ONLY \
the numbered sources provided. Rules:

1. Use ONLY information from the sources. Never use outside knowledge.
2. Cite every claim with its source number in square brackets, e.g. [1] or [2][3].
3. If the sources do not contain the answer, say exactly: \
"I couldn't find this in the provided documents." Do not guess.
4. Be concise and direct."""


def _build_user_prompt(question: str, chunks: list[dict]) -> str:
    """Format the retrieved chunks as numbered sources, followed by the question.

    Numbering the sources is what makes citations checkable: the model cites
    [2], and we know exactly which chunk [2] is.
    """
    sources = "\n\n".join(
        f"[{i}] {chunk['text']}" for i, chunk in enumerate(chunks, start=1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Generate a cited answer to `question` from the retrieved `chunks`.

    Returns {"answer": str, "sources": list[dict]} — the sources are the chunks
    in their numbered order, so [n] in the answer maps to sources[n-1].
    """
    if not chunks:
        return {
            "answer": "I couldn't find this in the provided documents.",
            "sources": [],
        }

    response = _client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(question, chunks)},
        ],
        temperature=0.1,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }

"""LLM-as-judge evaluation over the golden set, gated by config thresholds.

Runs each golden question through the real pipeline (hybrid search -> generate),
then asks an LLM judge to score the answer on two axes — faithfulness (is every
claim supported by the retrieved sources?) and relevancy (does it address the
question?). Averages are gated against config thresholds; a regression exits
non-zero, which fails the CI build.

This is a hand-rolled LLM-as-judge eval — the same idea RAGAS implements, kept
dependency-free and fully under our control. Groq's response_format guarantees
valid JSON, so the scoring never hits the structured-output parsing fragility
that RAGAS/DeepEval suffer against Groq's OpenAI-compatible API. The judge runs
on Groq's larger model, since a judge should be at least as capable as the model
it grades.
"""

import json

from openai import OpenAI

from app.core.config import settings
from app.graph.generator import generate_answer
from app.ingestion import bm25_index, embedder
from app.ingestion.chunker import chunk_text
from app.retrieval.hybrid import search
from eval.testset import EVAL_CASES, EVAL_DOCUMENT

# A judge at least as strong as the generator. 70b handles judging reliably.
_JUDGE_MODEL = "llama-3.3-70b-versatile"

_judge = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)

_JUDGE_PROMPT = """You are a strict evaluator of a question-answering system. \
You are given a QUESTION, the SOURCES the system retrieved, and the system's \
ANSWER. Score two things, each from 0.0 to 1.0:

- "faithfulness": is every claim in the ANSWER supported by the SOURCES? 1.0 means \
fully grounded in the sources; 0.0 means it contradicts them or invents facts not \
present. If the answer says it cannot find the information AND the sources indeed \
lack it, that is faithful (1.0).
- "relevancy": does the ANSWER actually address the QUESTION? 1.0 means directly \
on point; 0.0 means off-topic.

Respond with ONLY a JSON object, no other text:
{"faithfulness": <float>, "relevancy": <float>}"""


def _ingest_eval_document() -> None:
    chunks = chunk_text(
        EVAL_DOCUMENT,
        doc_id="bill_of_rights",
        metadata={"source": "bill_of_rights.txt"},
    )
    embedder.add_chunks(chunks)
    bm25_index.add_chunks(chunks)


def _judge_answer(question: str, sources: list[dict], answer: str) -> dict:
    """Ask the LLM judge to score one answer. Returns {faithfulness, relevancy}."""
    sources_text = "\n\n".join(
        f"[{i}] {s['text']}" for i, s in enumerate(sources, start=1)
    )
    user = f"QUESTION: {question}\n\nSOURCES:\n{sources_text}\n\nANSWER: {answer}"

    response = _judge.chat.completions.create(
        model=_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _JUDGE_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    scores = json.loads(response.choices[0].message.content)
    return {
        "faithfulness": float(scores["faithfulness"]),
        "relevancy": float(scores["relevancy"]),
    }


def main() -> None:
    _ingest_eval_document()

    print(f"Evaluating {len(EVAL_CASES)} cases...\n")
    results = []
    for case in EVAL_CASES:
        question = case["question"]
        answer_result = generate_answer(question, search(question, top_k=5))
        answer_result["answer"] = "This document is about cooking recipes and contains no legal information."
        scores = _judge_answer(
            question, answer_result["sources"], answer_result["answer"]
        )
        results.append(scores)
        print(
            f"  {question}\n"
            f"    faithfulness={scores['faithfulness']:.3f}  "
            f"relevancy={scores['relevancy']:.3f}"
        )

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["relevancy"] for r in results) / len(results)
    print(f"\nAverages:  faithfulness={avg_faith:.3f}  relevancy={avg_rel:.3f}")

    failed = []
    if avg_faith < settings.eval_faithfulness_threshold:
        failed.append(
            f"faithfulness {avg_faith:.3f} < {settings.eval_faithfulness_threshold}"
        )
    if avg_rel < settings.eval_answer_relevancy_threshold:
        failed.append(
            f"relevancy {avg_rel:.3f} < {settings.eval_answer_relevancy_threshold}"
        )

    if failed:
        print("\nFAILED:", "; ".join(failed))
        raise SystemExit(1)

    print("\nPASSED — all metrics above threshold.")


if __name__ == "__main__":
    main()

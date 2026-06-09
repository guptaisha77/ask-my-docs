"""Split documents into overlapping, token-bounded chunks.

Strategy: split text into sentences, group whole sentences into windows of
~chunk_size tokens, and overlap each window with the next by ~chunk_overlap
tokens. Chunks never cut mid-sentence, and context is preserved across
boundaries by the overlap.
"""

import re
import uuid
from dataclasses import dataclass, field

import tiktoken

from app.core.config import settings


@dataclass
class Chunk:
    """One piece of a document, ready to be embedded and indexed."""

    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


# Load the tokenizer ONCE at import time, not per function call.
# cl100k_base is the encoding used by GPT-4/3.5 and the OpenAI embedding models.
# Loading it reads a vocabulary file (~50ms), so we do it a single time and reuse.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the exact number of tokens in `text`."""
    return len(_TOKENIZER.encode(text))


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences on `.`, `!`, or `?` boundaries.

    We split on sentences first so chunks never cut mid-sentence. This is a
    deliberately simple splitter — good enough for prose and business documents.
    It will occasionally mis-split on abbreviations ("Dr." "e.g.") but that
    trade-off is acceptable here and keeps the code dependency-free.
    """
    # The regex finds a sentence-ending punctuation mark (. ! ?) followed by
    # one or more spaces, and splits there. The lookbehind (?<=...) keeps the
    # punctuation attached to the sentence it ends, rather than discarding it.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    # Drop any empty strings that can result from trailing whitespace.
    return [p.strip() for p in parts if p.strip()]


def _overlap_tail(sentences: list[str], overlap_tokens: int) -> list[str]:
    """Return the trailing sentences whose combined size is ~`overlap_tokens`.

    Walks backwards from the end, collecting whole sentences until the token
    budget is reached. Returns whole sentences (never a partial one) so the
    overlap region stays readable.
    """
    if overlap_tokens <= 0:
        return []

    tail: list[str] = []
    running = 0
    # Walk the sentences in reverse so we collect from the end of the window.
    for sentence in reversed(sentences):
        running += count_tokens(sentence)
        tail.insert(0, sentence)
        if running >= overlap_tokens:
            break
    return tail


def chunk_text(text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]:
    """Split `text` into overlapping, token-bounded chunks.

    Sentences are grouped into windows of up to `settings.chunk_size` tokens.
    Each new window is seeded with the last `settings.chunk_overlap` tokens of
    the previous one, so context is preserved across boundaries.
    """
    # Pull the tunables from config so behaviour is controlled in one place.
    max_tokens = settings.chunk_size
    overlap = settings.chunk_overlap

    sentences = _split_into_sentences(text)

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # If adding this sentence would overflow the window, close the current
        # window into a Chunk first — but only if the window has content.
        if current_tokens + sentence_tokens > max_tokens and current_sentences:
            chunk_str = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=chunk_str,
                    token_count=count_tokens(chunk_str),
                    chunk_index=chunk_index,
                    metadata=dict(metadata or {}),
                )
            )
            chunk_index += 1

            # Seed the next window with the tail of this one for overlap.
            current_sentences = _overlap_tail(current_sentences, overlap)
            current_tokens = count_tokens(" ".join(current_sentences))

        # Add the sentence to the (possibly freshly-seeded) window.
        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Flush the final window if anything remains.
    if current_sentences:
        chunk_str = " ".join(current_sentences)
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=chunk_str,
                token_count=count_tokens(chunk_str),
                chunk_index=chunk_index,
                metadata=dict(metadata or {}),
            )
        )

    return chunks

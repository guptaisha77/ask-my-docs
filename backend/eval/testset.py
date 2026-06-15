"""Golden evaluation set over a real public document: the US Bill of Rights.

Source: Project Gutenberg eBook #2 (public domain, permanently hosted at
https://www.gutenberg.org/ebooks/2). The document text lives in
eval/bill_of_rights.txt so anyone can open it and verify these answers.

Small but STABLE — scores are only comparable across changes if the set doesn't
move. Each reference answer is a plainly-stated fact, checkable in seconds against
the amendments themselves.
"""

from pathlib import Path

# The eval document is read from a file checked into the repo.
_DOC_PATH = Path(__file__).parent / "bill_of_rights.txt"
EVAL_DOCUMENT = _DOC_PATH.read_text(encoding="utf-8")

EVAL_CASES = [
    {
        "question": "Which amendment protects freedom of speech and religion?",
        "reference": "The First Amendment protects freedom of religion, speech, the press, assembly, and petition.",
    },
    {
        "question": "What does the Second Amendment protect?",
        "reference": "The Second Amendment protects the right of the people to keep and bear arms.",
    },
    {
        "question": "What does the Third Amendment say about soldiers?",
        "reference": "The Third Amendment prohibits quartering soldiers in any house during peacetime without the owner's consent.",
    },
    {
        "question": "Which amendment protects against unreasonable searches and seizures?",
        "reference": "The Fourth Amendment protects against unreasonable searches and seizures and requires warrants based on probable cause.",
    },
    {
        "question": "What right does the Sixth Amendment guarantee in criminal prosecutions?",
        "reference": "The Sixth Amendment guarantees the right to a speedy and public trial by an impartial jury.",
    },
]

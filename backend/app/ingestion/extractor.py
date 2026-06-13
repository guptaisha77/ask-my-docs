"""Extract plain text from uploaded documents.

Supports .txt and .pdf. PDFs are read with pypdf, page by page. Scanned
(image-only) PDFs yield empty text — OCR is a known, deliberate gap for now.
"""

from pathlib import Path

from pypdf import PdfReader


def extract_text(file_path: str) -> str:
    """Return the plain text content of a .txt or .pdf file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    raise ValueError(f"Unsupported file type: '{suffix}' (use .txt or .pdf)")

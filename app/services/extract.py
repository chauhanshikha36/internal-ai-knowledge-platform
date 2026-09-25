"""Content extraction from uploaded files.

PDFs: use the embedded text layer when present; pages without one (scans, or documents
"printed to PDF" so the text became vector outlines) fall back to Tesseract OCR.
"""

import re
import statistics
from pathlib import Path

from ..config import settings


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _norm(line: str) -> str:
    return re.sub(r"[^a-z]+", "", line.lower())


def _layer_heading(page) -> str | None:
    """Heuristic for text-layer pages: the largest line if clearly bigger than body text."""
    lines: list[tuple[float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if spans:
                lines.append(
                    (
                        max(s["size"] for s in spans),
                        " ".join(s["text"].strip() for s in spans),
                    )
                )
    return _pick_heading(lines)


def _pick_heading(lines: list[tuple[float, str]]) -> str | None:
    if len(lines) < 3:
        return None
    med = statistics.median(size for size, _ in lines)
    size, text = max(lines, key=lambda x: x[0])
    words = text.split()
    alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
    # logos, stats and page numbers are large too, so require a real multi-word phrase
    ok = size >= 1.3 * med and len(words) >= 3 and alpha >= 0.6 and len(text) <= 120
    return text if ok else None


def _ocr_page(page) -> tuple[str, str | None]:
    """OCR one page. Returns (text, heading). Heading = tallest line if clearly larger than body text."""
    import os

    import pytesseract
    from PIL import Image

    default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and os.path.exists(default_win_path):
        pytesseract.pytesseract.tesseract_cmd = default_win_path

    pix = page.get_pixmap(dpi=settings.ocr_dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    data = pytesseract.image_to_data(
        img, lang=settings.ocr_lang, output_type=pytesseract.Output.DICT
    )

    lines: dict[tuple[int, int, int], list[tuple[str, int]]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip() or float(data["conf"][i]) < settings.ocr_min_confidence:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append((word, data["height"][i]))

    out: list[str] = []
    sized: list[tuple[float, str]] = []
    prev_par = None
    for key, words in lines.items():
        text = re.sub(
            r"\bAl\b", "AI", " ".join(w for w, _ in words)
        )  # Tesseract reads "AI" as "Al"
        if sum(c.isalpha() for c in text) < 3:  # decoration noise
            continue
        if prev_par is not None and key[:2] != prev_par:
            out.append("")  # paragraph break
        prev_par = key[:2]
        out.append(text)
        sized.append((statistics.median(h for _, h in words), text))
    return "\n".join(out), _pick_heading(sized)


def _strip_repeated_lines(pages: list[dict]) -> None:
    """Remove running headers/footers (lines that appear on many pages)."""
    if len(pages) < 4:
        return
    counts: dict[str, int] = {}
    for p in pages:
        for n in {_norm(l) for l in p["text"].splitlines() if _norm(l)}:
            counts[n] = counts.get(n, 0) + 1
    threshold = max(3, int(0.4 * len(pages)))
    boilerplate = {n for n, c in counts.items() if c >= threshold}
    for p in pages:
        p["text"] = "\n".join(
            l for l in p["text"].splitlines() if _norm(l) not in boilerplate
        )


def extract_pdf(path: Path) -> list[dict]:
    """Returns [{'page': n, 'text': str, 'heading': str|None, 'ocr': bool}] using real 1-based page indexes."""
    import pymupdf

    pages: list[dict] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if len(text.strip()) >= settings.ocr_min_chars:
                pages.append(
                    {
                        "page": i,
                        "text": text,
                        "heading": _layer_heading(page),
                        "ocr": False,
                    }
                )
            elif settings.ocr_enabled:
                text, heading = _ocr_page(page)
                if text.strip():
                    pages.append(
                        {"page": i, "text": text, "heading": heading, "ocr": True}
                    )
    _strip_repeated_lines(pages)
    last = None
    for p in pages:  # carry the nearest preceding heading forward
        p["heading"] = p["heading"] or last
        last = p["heading"]
    return [p for p in pages if p["text"].strip()]

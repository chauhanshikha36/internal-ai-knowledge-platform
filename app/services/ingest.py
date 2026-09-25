"""Ingestion pipeline: extract > chunk > embed > store. Runs inside the worker."""
import json
import logging
from pathlib import Path

from sqlalchemy import text

from ..config import settings
from ..db import engine
from .chunking import Chunk, chunk_code, chunk_markdown, chunk_prose, recursive_split
from .embeddings import embed_documents, to_pgvector
from .extract import extract_pdf, read_text

log = logging.getLogger("ingest")

EXT_TYPES = {".pdf": "pdf", ".md": "markdown", ".markdown": "markdown", ".txt": "text", ".rst": "text"}
CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs",
    ".rb", ".php", ".kt", ".swift", ".scala", ".sql", ".sh", ".yaml", ".yml", ".json", ".toml",
}
SUPPORTED_EXTS = set(EXT_TYPES) | CODE_EXTS


def detect_file_type(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    if ext in EXT_TYPES:
        return EXT_TYPES[ext]
    return "code" if ext in CODE_EXTS else None


def build_chunks(path: Path, file_type: str) -> list[Chunk]:
    size, overlap = settings.chunk_size, settings.chunk_overlap
    if file_type == "pdf":
        chunks: list[Chunk] = []
        for page in extract_pdf(path):
            for part in recursive_split(page["text"], size, overlap):
                meta = {"page": page["page"]}
                if page["ocr"]:
                    meta["ocr"] = True
                if page["heading"]:
                    meta["heading"] = page["heading"]
                chunks.append(Chunk(part, meta))
        if not chunks:
            raise ValueError("No extractable text in PDF (text layer empty and OCR disabled or found nothing)")
    else:
        source = read_text(path)
        if file_type == "markdown":
            chunks = chunk_markdown(source, size, overlap)
        elif file_type == "code":
            chunks = chunk_code(source, path.suffix.lower(), settings.code_chunk_max_chars)
        else:
            chunks = chunk_prose(source, size, overlap)
        if not chunks:
            raise ValueError("Document is empty")
    for c in chunks:
        c.content = c.content.replace("\x00", "")  # Postgres rejects NUL bytes
    return chunks


def embedding_text(filename: str, chunk: Chunk) -> str:
    """Contextual header improves retrieval: the model sees where the chunk came from."""
    m = chunk.metadata
    label = m.get("symbol") or m.get("heading") or ""
    header = f"{filename} | {label}" if label else filename
    return f"{header}\n{chunk.content}"


def ingest_document(document_id: str) -> None:
    with engine.begin() as conn:
        doc = conn.execute(
            text("SELECT id, filename, file_type, storage_path, deleted_at FROM documents WHERE id = :id FOR UPDATE"),
            {"id": document_id},
        ).first()
        if doc is None or doc.deleted_at is not None:
            log.info("skip ingest %s (missing or deleted)", document_id)
            return
        conn.execute(
            text("UPDATE documents SET status='processing', error=NULL, updated_at=now() WHERE id=:id"),
            {"id": document_id},
        )

    chunks = build_chunks(Path(doc.storage_path), doc.file_type)
    vectors = embed_documents([embedding_text(doc.filename, c) for c in chunks])
    log.info("document %s: %d chunks embedded", document_id, len(chunks))

    rows = [
        {
            "d": document_id,
            "i": i,
            "c": c.content,
            "m": json.dumps(c.metadata),
            "e": to_pgvector(v),
            "model": settings.embedding_model if settings.embedding_backend != "fake" else "fake-hash",
        }
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    with engine.begin() as conn:
        alive = conn.execute(
            text("SELECT 1 FROM documents WHERE id=:id AND deleted_at IS NULL FOR UPDATE"), {"id": document_id}
        ).first()
        if not alive:
            log.info("document %s was deleted during ingest, discarding", document_id)
            return
        # Idempotent: a retry replaces any chunks written by a previous attempt.
        conn.execute(text("DELETE FROM chunks WHERE document_id=:id"), {"id": document_id})
        conn.execute(
            text(
                "INSERT INTO chunks (document_id, chunk_index, content, metadata, embedding, embedding_model) "
                "VALUES (:d, :i, :c, CAST(:m AS jsonb), CAST(:e AS vector), :model)"
            ),
            rows,
        )
        conn.execute(
            text(
                "UPDATE documents SET status='ready', chunk_count=:n, embedding_model=:model, error=NULL, "
                "processed_at=now(), updated_at=now() WHERE id=:id"
            ),
            {"n": len(rows), "model": rows[0]["model"], "id": document_id},
        )

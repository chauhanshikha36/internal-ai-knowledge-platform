"""Internal AI Knowledge Platform: HTTP API."""

import hashlib
import hmac
import json
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Response,
    UploadFile,
)
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from .config import settings
from .db import engine
from .schemas import (
    DeleteResponse,
    DocumentList,
    DocumentOut,
    QueryRequest,
    QueryResponse,
    SearchResult,
    UploadResponse,
)
from .services.embeddings import embed_query, get_model
from .services.ingest import SUPPORTED_EXTS, detect_file_type
from .services.rerank import rerank
from .services.search import search_chunks

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.embedding_backend != "fake":
        get_model()
    yield


app = FastAPI(
    title="Internal AI Knowledge Platform", version="1.0.0", lifespan=lifespan
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and not hmac.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(401, "Invalid or missing X-API-Key")


auth = [Depends(require_api_key)]

DOC_COLUMNS = (
    "id, filename, file_type, size_bytes, status, error, chunk_count, embedding_model, "
    "tags, metadata, created_at, processed_at, deleted_at"
)
LATEST_JOB = (
    "SELECT id, type, status, attempts, last_error FROM jobs WHERE document_id=:id "
    "ORDER BY created_at DESC LIMIT 1"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"database unavailable: {e}")
    return {"status": "ready"}


# --------------------------------------------------------------- documents
@app.post(
    "/documents", response_model=UploadResponse, status_code=202, dependencies=auth
)
def upload_document(
    response: Response,
    file: UploadFile = File(...),
    tags: str = Form("", description="comma separated"),
    metadata: str = Form("{}", description="JSON object"),
):
    """Accept a file and queue it for asynchronous processing (extract, chunk, embed, store)."""
    filename = Path(file.filename or "").name
    file_type = detect_file_type(filename)
    if not filename or file_type is None:
        raise HTTPException(
            415,
            f"Unsupported file type. Supported extensions: {sorted(SUPPORTED_EXTS)}",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        meta = json.loads(metadata or "{}")
        if not isinstance(meta, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "metadata must be a JSON object")
    tag_list = sorted({t.strip() for t in tags.split(",") if t.strip()})

    sha = hashlib.sha256(data).hexdigest()
    doc_id = uuid.uuid4()
    path = Path(settings.upload_dir) / str(doc_id) / filename

    def existing():
        with engine.connect() as conn:
            return conn.execute(
                text(
                    "SELECT id, status FROM documents WHERE sha256=:s AND deleted_at IS NULL"
                ),
                {"s": sha},
            ).first()

    dup = existing()
    if dup:
        response.status_code = 200
        return UploadResponse(
            document_id=dup.id,
            status=dup.status,
            duplicate=True,
            status_url=f"/documents/{dup.id}",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    job_id = uuid.uuid4()
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO documents (id, filename, file_type, content_type, size_bytes, sha256, storage_path, tags, metadata) "
                    "VALUES (:id, :fn, :ft, :ct, :sz, :sha, :sp, CAST(:tags AS text[]), CAST(:meta AS jsonb))"
                ),
                {
                    "id": doc_id,
                    "fn": filename,
                    "ft": file_type,
                    "ct": file.content_type,
                    "sz": len(data),
                    "sha": sha,
                    "sp": str(path),
                    "tags": tag_list,
                    "meta": json.dumps(meta),
                },
            )
            conn.execute(
                text(
                    "INSERT INTO jobs (id, document_id, type, max_attempts) VALUES (:j, :d, 'ingest', :m)"
                ),
                {"j": job_id, "d": doc_id, "m": settings.job_max_attempts},
            )
    except IntegrityError:  # concurrent upload of identical bytes
        shutil.rmtree(path.parent, ignore_errors=True)
        dup = existing()
        if dup:
            response.status_code = 200
            return UploadResponse(
                document_id=dup.id,
                status=dup.status,
                duplicate=True,
                status_url=f"/documents/{dup.id}",
            )
        raise
    except Exception:
        shutil.rmtree(path.parent, ignore_errors=True)
        raise
    return UploadResponse(
        document_id=doc_id,
        job_id=job_id,
        status="pending",
        status_url=f"/documents/{doc_id}",
    )


@app.get("/documents", response_model=DocumentList, dependencies=auth)
def list_documents(
    status: str | None = None,
    include_deleted: bool = False,
    limit: int = 50,
    offset: int = 0,
):
    limit = max(1, min(limit, 200))
    where, params = [], {"limit": limit, "offset": max(0, offset)}
    if not include_deleted:
        where.append("deleted_at IS NULL")
    if status:
        where.append("status = :status")
        params["status"] = status
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT {DOC_COLUMNS} FROM documents {clause} ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
            ),
            params,
        ).fetchall()
    return DocumentList(
        items=[DocumentOut(**r._mapping) for r in rows],
        limit=limit,
        offset=params["offset"],
    )


@app.get("/documents/{document_id}", response_model=DocumentOut, dependencies=auth)
def get_document(document_id: uuid.UUID):
    with engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT {DOC_COLUMNS} FROM documents WHERE id=:id"),
            {"id": document_id},
        ).first()
        if not row:
            raise HTTPException(404, "Document not found")
        job = conn.execute(text(LATEST_JOB), {"id": document_id}).first()
    return DocumentOut(**row._mapping, job=job._mapping if job else None)


@app.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    status_code=202,
    dependencies=auth,
)
def delete_document(document_id: uuid.UUID, mode: str = "soft"):
    """soft: hidden from search immediately, data kept for the retention window, restorable.
    hard: hidden immediately and purged (chunks, embeddings, file, metadata) by the worker right away.
    """
    if mode not in ("soft", "hard"):
        raise HTTPException(422, "mode must be 'soft' or 'hard'")
    delay = 0 if mode == "hard" else settings.soft_delete_retention_hours * 3600
    with engine.begin() as conn:
        doc = conn.execute(
            text("SELECT id, deleted_at FROM documents WHERE id=:id FOR UPDATE"),
            {"id": document_id},
        ).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        if doc.deleted_at is None:
            conn.execute(
                text(
                    "UPDATE documents SET deleted_at=now(), status='deleted', updated_at=now() WHERE id=:id"
                ),
                {"id": document_id},
            )
        # Also cancel a pending ingest so the worker does not resurrect the doc
        conn.execute(
            text(
                "UPDATE jobs SET status='cancelled', finished_at=now() WHERE document_id=:id AND type='ingest' AND status='queued'"
            ),
            {"id": document_id},
        )
        run_after = conn.execute(
            text(
                "INSERT INTO jobs (document_id, type, max_attempts, run_after) "
                "VALUES (:d, 'hard_delete', :m, now() + (:delay * interval '1 second')) "
                "ON CONFLICT (document_id, type) WHERE status IN ('queued', 'running') "
                "DO UPDATE SET run_after = LEAST(jobs.run_after, EXCLUDED.run_after) "
                "RETURNING run_after"
            ),
            {"d": document_id, "m": settings.job_max_attempts, "delay": delay},
        ).scalar_one()
    return DeleteResponse(
        document_id=document_id,
        mode=mode,
        status="deleted",
        purge_scheduled_at=run_after,
    )


@app.post(
    "/documents/{document_id}/retry",
    response_model=UploadResponse,
    status_code=202,
    dependencies=auth,
)
def retry_document(document_id: uuid.UUID):
    """Re-queue ingestion for a document whose processing failed."""
    with engine.begin() as conn:
        doc = conn.execute(
            text(
                "SELECT id, status FROM documents WHERE id=:id AND deleted_at IS NULL FOR UPDATE"
            ),
            {"id": document_id},
        ).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        if doc.status != "failed":
            raise HTTPException(
                409, f"Only failed documents can be retried (status is '{doc.status}')"
            )
        conn.execute(
            text(
                "UPDATE documents SET status='pending', error=NULL, updated_at=now() WHERE id=:id"
            ),
            {"id": document_id},
        )
        job_id = conn.execute(
            text(
                "INSERT INTO jobs (document_id, type, max_attempts) VALUES (:d, 'ingest', :m) RETURNING id"
            ),
            {"d": document_id, "m": settings.job_max_attempts},
        ).scalar_one()
    return UploadResponse(
        document_id=document_id,
        job_id=job_id,
        status="pending",
        status_url=f"/documents/{document_id}",
    )


@app.post(
    "/documents/{document_id}/restore", response_model=DocumentOut, dependencies=auth
)
def restore_document(document_id: uuid.UUID):
    """Undo a soft delete while the purge has not started."""
    with engine.begin() as conn:
        doc = conn.execute(
            text(
                "SELECT id, deleted_at, chunk_count FROM documents WHERE id=:id FOR UPDATE"
            ),
            {"id": document_id},
        ).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        if doc.deleted_at is None:
            raise HTTPException(409, "Document is not deleted")
        cancelled = conn.execute(
            text(
                "UPDATE jobs SET status='cancelled', finished_at=now() "
                "WHERE document_id=:id AND type='hard_delete' AND status='queued' AND run_after > now() RETURNING id"
            ),
            {"id": document_id},
        ).first()
        running = conn.execute(
            text(
                "SELECT 1 FROM jobs WHERE document_id=:id AND type='hard_delete' AND status='running'"
            ),
            {"id": document_id},
        ).first()
        if running or not cancelled:
            raise HTTPException(
                409, "Purge already started, document cannot be restored"
            )
        new_status = "ready" if doc.chunk_count > 0 else "pending"
        conn.execute(
            text(
                "UPDATE documents SET deleted_at=NULL, status=:s, updated_at=now() WHERE id=:id"
            ),
            {"s": new_status, "id": document_id},
        )
        if new_status == "pending":  # ingest never finished: queue it again
            conn.execute(
                text(
                    "INSERT INTO jobs (document_id, type, max_attempts) VALUES (:d, 'ingest', :m) ON CONFLICT DO NOTHING"
                ),
                {"d": document_id, "m": settings.job_max_attempts},
            )
    return get_document(document_id)


# -------------------------------------------------------------------- query
@app.post("/query", response_model=QueryResponse, dependencies=auth)
def query_documents(req: QueryRequest, x_client_id: str | None = Header(default=None)):
    """Semantic search: embed the query, filter by metadata, vector search, optional rerank."""
    t0 = time.perf_counter()
    query = req.query.strip()
    if not query:
        raise HTTPException(422, "query must not be blank")

    t = time.perf_counter()
    qvec, cache_hit = embed_query(query)
    embed_ms = int((time.perf_counter() - t) * 1000)

    do_rerank = settings.rerank_enabled if req.rerank is None else req.rerank
    fetch_k = max(req.top_k * 4, 20) if do_rerank else req.top_k

    t = time.perf_counter()
    rows = search_chunks(qvec, fetch_k, req.filters)
    search_ms = int((time.perf_counter() - t) * 1000)

    rerank_ms = 0
    if do_rerank:
        t = time.perf_counter()
        try:
            rows = rerank(query, rows, req.top_k)
        except Exception as e:
            log.warning("rerank failed, falling back to vector order: %s", e)
            do_rerank = False
            rows = rows[: req.top_k]
        rerank_ms = int((time.perf_counter() - t) * 1000)
    if req.min_score is not None:
        rows = [r for r in rows if r["score"] >= req.min_score]

    results = [SearchResult(rank=i + 1, **r) for i, r in enumerate(rows[: req.top_k])]
    total_ms = int((time.perf_counter() - t0) * 1000)

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO query_logs (client_id, query_text, filters, top_k, reranked, cache_hit, result_count, "
                    "top_score, top_document_id, embed_ms, search_ms, rerank_ms, total_ms) "
                    "VALUES (:c, :q, CAST(:f AS jsonb), :k, :rr, :ch, :n, :ts, :td, :e, :s, :r, :t)"
                ),
                {
                    "c": x_client_id,
                    "q": query,
                    "f": (
                        req.filters.model_dump_json(exclude_none=True)
                        if req.filters
                        else "{}"
                    ),
                    "k": req.top_k,
                    "rr": do_rerank,
                    "ch": cache_hit,
                    "n": len(results),
                    "ts": results[0].score if results else None,
                    "td": results[0].document_id if results else None,
                    "e": embed_ms,
                    "s": search_ms,
                    "r": rerank_ms,
                    "t": total_ms,
                },
            )
    except Exception as e:
        log.warning("query log insert failed: %s", e)

    return QueryResponse(
        query=query,
        results=results,
        reranked=do_rerank,
        cache_hit=cache_hit,
        timings_ms={
            "embed": embed_ms,
            "search": search_ms,
            "rerank": rerank_ms,
            "total": total_ms,
        },
    )

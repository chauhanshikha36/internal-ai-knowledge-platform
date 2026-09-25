"""Vector similarity search with metadata filtering (pgvector, cosine distance)."""
import json

from sqlalchemy import text

from ..db import engine
from ..schemas import QueryFilters
from .embeddings import to_pgvector


def _filter_sql(f: QueryFilters | None, params: dict) -> str:
    if not f:
        return ""
    clauses = []
    if f.document_ids:
        clauses.append("c.document_id = ANY(CAST(:doc_ids AS uuid[]))")
        params["doc_ids"] = [str(x) for x in f.document_ids]
    if f.file_types:
        clauses.append("d.file_type = ANY(CAST(:file_types AS text[]))")
        params["file_types"] = f.file_types
    if f.tags:
        clauses.append("d.tags && CAST(:tags AS text[])")
        params["tags"] = f.tags
    if f.metadata:
        clauses.append("d.metadata @> CAST(:doc_meta AS jsonb)")
        params["doc_meta"] = json.dumps(f.metadata)
    if f.chunk_metadata:
        clauses.append("c.metadata @> CAST(:chunk_meta AS jsonb)")
        params["chunk_meta"] = json.dumps(f.chunk_metadata)
    return "".join(f" AND {c}" for c in clauses)


def search_chunks(qvec: list[float], limit: int, filters: QueryFilters | None) -> list[dict]:
    params = {"q": to_pgvector(qvec), "limit": limit}
    where = _filter_sql(filters, params)
    sql = f"""
        SELECT c.id AS chunk_id, c.document_id, d.filename, d.file_type, c.chunk_index,
               c.content, c.metadata,
               1 - (c.embedding <=> CAST(:q AS vector)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.deleted_at IS NULL AND d.status = 'ready'{where}
        ORDER BY c.embedding <=> CAST(:q AS vector)
        LIMIT :limit
    """
    with engine.begin() as conn:
        conn.execute(text("SELECT set_config('hnsw.ef_search', :ef, true)"), {"ef": str(max(100, limit * 2))})
        try:  # pgvector >= 0.8: keep scanning the index until enough rows pass the filters
            with conn.begin_nested():
                conn.execute(text("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"))
        except Exception:
            pass
        rows = [dict(r._mapping) for r in conn.execute(text(sql), params)]
    rows.sort(key=lambda r: r["score"], reverse=True)  # relaxed_order may be slightly unordered
    for r in rows:
        r["score"] = float(r["score"])
        r["rerank_score"] = None
    return rows

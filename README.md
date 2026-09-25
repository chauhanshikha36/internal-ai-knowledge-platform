# Internal AI Knowledge Platform

A backend for ~100 internal developers to upload documents and code, query them in natural language,
run semantic search with metadata filters, and delete content safely.

- **API**: FastAPI (`/documents`, `/query`, `DELETE /documents/{id}`)
- **Async ingestion**: Postgres-backed job queue and worker (extract, OCR fallback, chunk, embed, store)
- **Vector search**: PostgreSQL 16 + pgvector (HNSW, cosine), optional cross-encoder reranking
- **Embeddings**: local `BAAI/bge-small-en-v1.5` (384 dim, no API key needed)

Full design (architecture diagram, schema, scaling, trade-offs): [docs/DESIGN.md](docs/DESIGN.md)


```bash
docker compose up --build        # db + api (:8000) + worker
# interactive API docs: http://localhost:8000/docs
```

Run the proof of execution for the two task files (already in `samples/`):

```bash
pip install httpx
python scripts/demo.py --with-delete | tee proof/demo_output.txt
```

The demo uploads both files, waits for async processing, runs 8 semantic queries (4 per file), 4 metadata-filter
queries, then demos soft delete, restore and hard delete. Every query prints the top hits with page or symbol and
line numbers, and a PASS/FAIL on whether the top hit came from the expected file.

> **Note on the sample PDF:** `Knowledge_Base_Sample.pdf` has no text layer (it was printed to PDF, so the text is
> vector outlines). Plain extraction returns nothing, so the pipeline falls back to Tesseract OCR per page.
> Ingesting its 22 pages takes about a minute on CPU. This is why upload is asynchronous.

Local development without Docker: `pip install -r requirements.txt`, install Tesseract, start Postgres with
pgvector, load `sql/schema.sql`, then run `uvicorn app.main:app` and `python -m app.worker` with `DATABASE_URL` set.

## API

Optional auth: set `API_KEY` and send `X-API-Key`. Optional `X-Client-Id` is stored in the query log.

### `POST /documents` (multipart, async)

Fields: `file` (pdf, md, txt, or code), `tags` (comma separated, optional), `metadata` (JSON object, optional).

```bash
curl -F "file=@samples/Knowledge_Base_Sample.pdf" -F "tags=knowledge-base" \
     -F 'metadata={"team":"platform"}' localhost:8000/documents
# 202 {"document_id":"...","job_id":"...","status":"pending","duplicate":false,"status_url":"/documents/..."}
```

- Returns `202` immediately. Poll `GET /documents/{id}` until `status` is `ready` (or `failed` with `error`).
- Same bytes uploaded again returns `200` with `duplicate: true` (idempotent, SHA-256).
- `415` unsupported type, `413` over 20 MB, `400` empty file or bad metadata.
- Failed ingestion is retried with exponential backoff (5 attempts). `POST /documents/{id}/retry` re-queues a failed one.

### `POST /query`

```bash
curl -X POST localhost:8000/query -H 'content-type: application/json' -d '{
  "query": "What happens when a proxy reports a failure?",
  "top_k": 5,
  "filters": {"file_types": ["code"]},
  "rerank": false
}'
```

Request: `query`, `top_k` (1-50), `rerank` (overrides server default), `min_score`, and `filters`:

| filter | meaning |
|---|---|
| `document_ids` | restrict to these documents |
| `file_types` | `pdf`, `markdown`, `text`, `code` |
| `tags` | documents having any of these tags |
| `metadata` | document metadata containment (`@>`) |
| `chunk_metadata` | chunk metadata containment, e.g. `{"page": 16}` or `{"class_name": "DecayProxyRotator"}` |

Response: ranked `results` (`score` = cosine similarity, `content`, `filename`, chunk `metadata` such as page,
heading, symbol, start/end line), plus `reranked`, `cache_hit`, and `timings_ms`.

### `DELETE /documents/{id}?mode=soft|hard`

- `soft` (default): hidden from search immediately, data kept 7 days, then purged by the worker. Undo with `POST /documents/{id}/restore`.
- `hard`: hidden immediately, worker purges chunks + embeddings, file, then metadata row right away.
- Returns `202` with `purge_scheduled_at`. See DESIGN.md for the partial-failure handling.

### Other

`GET /documents` (list, paginate, filter by status), `GET /documents/{id}` (status + latest job),
`POST /documents/{id}/retry`, `POST /documents/{id}/restore`, `GET /health`, `GET /ready`.

## Project layout

```
app/main.py               HTTP API
app/worker.py             job worker (SKIP LOCKED queue, retries, stale-job recovery)
app/services/extract.py   PDF text layer + OCR fallback, header/footer stripping
app/services/chunking.py  recursive, markdown, and AST (Python) chunkers
app/services/ingest.py    extract > chunk > embed > store
app/services/search.py    filtered pgvector search
app/services/rerank.py    optional cross-encoder
app/services/deletion.py  idempotent hard delete
sql/schema.sql            full schema and indexes
scripts/demo.py           end-to-end proof of execution
tests/                    chunking unit tests (pytest)
```

## Testing

```bash
pytest -q tests                                  # chunking unit tests
python scripts/demo.py --with-delete             # end to end against a running stack
```

`EMBEDDING_BACKEND=fake` swaps in a deterministic hashing embedder so the pipeline can be tested in CI without
downloading a model. It matches on shared words only, so retrieval quality checks should be run with the real model.

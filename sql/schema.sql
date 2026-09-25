-- Internal AI Knowledge Platform schema (PostgreSQL 16 + pgvector)
-- Loaded automatically by the db container on first start.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- documents
CREATE TABLE documents (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        text        NOT NULL,
    file_type       text        NOT NULL CHECK (file_type IN ('pdf', 'markdown', 'text', 'code')),
    content_type    text,
    size_bytes      bigint      NOT NULL,
    sha256          char(64)    NOT NULL,
    storage_path    text        NOT NULL,
    status          text        NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing', 'ready', 'failed', 'deleted', 'deleting')),
    error           text,
    chunk_count     int         NOT NULL DEFAULT 0,
    embedding_model text,
    tags            text[]      NOT NULL DEFAULT '{}',
    metadata        jsonb       NOT NULL DEFAULT '{}',
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    processed_at    timestamptz,
    deleted_at      timestamptz          -- soft delete marker
);

-- Idempotent uploads: the same bytes cannot be active twice.
CREATE UNIQUE INDEX uq_documents_active_sha ON documents (sha256) WHERE deleted_at IS NULL;
CREATE INDEX ix_documents_status     ON documents (status) WHERE deleted_at IS NULL;
CREATE INDEX ix_documents_created    ON documents (created_at DESC);
CREATE INDEX ix_documents_tags       ON documents USING gin (tags);
CREATE INDEX ix_documents_metadata   ON documents USING gin (metadata jsonb_path_ops);
CREATE INDEX ix_documents_deleted_at ON documents (deleted_at) WHERE deleted_at IS NOT NULL;

-- ------------------------------------------------------------------- chunks
-- One row per chunk. The embedding lives on the row (same lifecycle as the text).
-- embedding_model is stored so a model upgrade can be rolled out and audited.
CREATE TABLE chunks (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id     uuid        NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index     int         NOT NULL,
    content         text        NOT NULL,
    metadata        jsonb       NOT NULL DEFAULT '{}',   -- page, heading, symbol, start_line, ...
    embedding       vector(384) NOT NULL,
    embedding_model text        NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)                    -- also serves document_id lookups
);

CREATE INDEX ix_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX ix_chunks_metadata ON chunks USING gin (metadata jsonb_path_ops);

-- --------------------------------------------------------------------- jobs
-- Postgres-backed queue (SELECT ... FOR UPDATE SKIP LOCKED). No document FK on
-- purpose: job history must survive the hard delete of the document row.
CREATE TABLE jobs (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  uuid        NOT NULL,
    type         text        NOT NULL CHECK (type IN ('ingest', 'hard_delete')),
    status       text        NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    attempts     int         NOT NULL DEFAULT 0,
    max_attempts int         NOT NULL DEFAULT 5,
    last_error   text,
    run_after    timestamptz NOT NULL DEFAULT now(),
    locked_at    timestamptz,
    created_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz
);

CREATE INDEX ix_jobs_claim    ON jobs (run_after) WHERE status IN ('queued', 'running');
CREATE INDEX ix_jobs_document ON jobs (document_id, created_at DESC);
-- At most one active job of each type per document.
CREATE UNIQUE INDEX uq_jobs_active ON jobs (document_id, type) WHERE status IN ('queued', 'running');

-- --------------------------------------------------------------- query_logs
-- Append-only, range partitioned by month so old data can be dropped cheaply.
CREATE TABLE query_logs (
    id              bigserial,
    created_at      timestamptz NOT NULL DEFAULT now(),
    client_id       text,
    query_text      text        NOT NULL,
    filters         jsonb       NOT NULL DEFAULT '{}',
    top_k           int         NOT NULL,
    reranked        boolean     NOT NULL DEFAULT false,
    cache_hit       boolean     NOT NULL DEFAULT false,
    result_count    int         NOT NULL,
    top_score       real,
    top_document_id uuid,
    embed_ms        int,
    search_ms       int,
    rerank_ms       int,
    total_ms        int,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Safety net so inserts never fail if a monthly partition is missing.
CREATE TABLE query_logs_default PARTITION OF query_logs DEFAULT;
CREATE INDEX ix_query_logs_created ON query_logs (created_at DESC);
CREATE INDEX ix_query_logs_client  ON query_logs (client_id, created_at DESC);

-- Helper: SELECT create_query_log_partition('2026-10-01');  (run monthly from cron/pg_cron)
CREATE OR REPLACE FUNCTION create_query_log_partition(month_start date) RETURNS void AS $$
DECLARE
    part_name text := 'query_logs_' || to_char(month_start, 'YYYY_MM');
BEGIN
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF query_logs FOR VALUES FROM (%L) TO (%L)',
        part_name, month_start, (month_start + interval '1 month')::date);
END;
$$ LANGUAGE plpgsql;

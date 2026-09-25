from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://kb:kb@localhost:5432/kb"
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 20
    api_key: str | None = None

    embedding_backend: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embed_batch_size: int = 32
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    query_cache_size: int = 1024

    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # OCR fallback for pages without a text layer (scanned / printed-to-PDF documents)
    ocr_enabled: bool = True
    ocr_dpi: int = 200
    ocr_lang: str = "eng"
    ocr_min_chars: int = 40
    ocr_min_confidence: int = 30

    # Chunking (characters)
    chunk_size: int = 1000
    chunk_overlap: int = 150
    code_chunk_max_chars: int = 1800

    # Jobs / deletion
    job_max_attempts: int = 5
    worker_poll_seconds: float = 1.0
    stale_job_minutes: int = 10
    soft_delete_retention_hours: int = 168  # 7 days before a soft-deleted doc is purged

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

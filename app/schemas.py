import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: uuid.UUID
    job_id: uuid.UUID | None = None
    status: str
    duplicate: bool = False
    status_url: str


class JobInfo(BaseModel):
    id: uuid.UUID
    type: str
    status: str
    attempts: int
    last_error: str | None = None


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    file_type: str
    size_bytes: int
    status: str
    error: str | None = None
    chunk_count: int
    embedding_model: str | None = None
    tags: list[str]
    metadata: dict[str, Any]
    created_at: datetime
    processed_at: datetime | None = None
    deleted_at: datetime | None = None
    job: JobInfo | None = None


class DocumentList(BaseModel):
    items: list[DocumentOut]
    limit: int
    offset: int


class QueryFilters(BaseModel):
    document_ids: list[uuid.UUID] | None = None
    file_types: list[str] | None = Field(default=None, description="pdf | markdown | text | code")
    tags: list[str] | None = Field(default=None, description="match documents having ANY of these tags")
    metadata: dict[str, Any] | None = Field(default=None, description="document metadata containment (@>)")
    chunk_metadata: dict[str, Any] | None = Field(
        default=None, description="chunk metadata containment, e.g. {'page': 3} or {'class_name': 'X'}"
    )


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: QueryFilters | None = None
    rerank: bool | None = Field(default=None, description="override server default")
    min_score: float | None = Field(default=None, ge=-1, le=1)


class SearchResult(BaseModel):
    rank: int
    chunk_id: int
    document_id: uuid.UUID
    filename: str
    file_type: str
    chunk_index: int
    score: float = Field(description="cosine similarity, higher is better")
    rerank_score: float | None = None
    content: str
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    query: str
    results: list[SearchResult]
    reranked: bool
    cache_hit: bool
    timings_ms: dict[str, int]


class DeleteResponse(BaseModel):
    document_id: uuid.UUID
    mode: str
    status: str
    purge_scheduled_at: datetime

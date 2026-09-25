"""Optional cross-encoder reranking of the vector-search candidates."""
import threading

from ..config import settings

_reranker = None
_lock = threading.Lock()


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder

                _reranker = CrossEncoder(settings.rerank_model, device="cpu")
    return _reranker


def rerank(query: str, rows: list[dict], top_k: int) -> list[dict]:
    if not rows:
        return rows
    scores = _get_reranker().predict([(query, r["content"]) for r in rows])
    for r, s in zip(rows, scores):
        r["rerank_score"] = float(s)
    rows.sort(key=lambda r: r["rerank_score"], reverse=True)
    return rows[:top_k]

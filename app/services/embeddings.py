"""Embedding generation. Local sentence-transformers by default."""
import hashlib
import math
import re
import threading
from collections import OrderedDict

from ..config import settings

_model = None
_model_lock = threading.Lock()
_cache: OrderedDict[str, list[float]] = OrderedDict()
_cache_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(settings.embedding_model, device="cpu")
    return _model


_STOP = set("a an the is are was were be of to in on for and or how what which who where when why does do did it its this that with as by at from".split())


def _fake_encode(texts: list[str]) -> list[list[float]]:
    """Deterministic hashed bag-of-words vectors. For tests/CI only, no model download."""
    out = []
    for t in texts:
        v = [0.0] * settings.embedding_dim
        for tok in re.findall(r"[a-z0-9_]+", t.lower()):
            if tok in _STOP:
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % settings.embedding_dim] += 1.0 if (h >> 100) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


def embed_documents(texts: list[str]) -> list[list[float]]:
    if settings.embedding_backend == "fake":
        return _fake_encode(texts)
    vecs = get_model().encode(
        texts,
        batch_size=settings.embed_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.tolist()


def _encode_query(q: str) -> list[float]:
    if settings.embedding_backend == "fake":
        return _fake_encode([q])[0]
    return embed_documents([settings.query_prefix + q])[0]


def embed_query(q: str) -> tuple[list[float], bool]:
    """Returns (vector, cache_hit). Small in-process LRU; use Redis when running many API replicas."""
    with _cache_lock:
        if q in _cache:
            _cache.move_to_end(q)
            return _cache[q], True
    vec = _encode_query(q)
    with _cache_lock:
        _cache[q] = vec
        while len(_cache) > settings.query_cache_size:
            _cache.popitem(last=False)
    return vec, False


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

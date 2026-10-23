"""Semantic retrieval over the policy index.

The index can be rebuilt while a process is using it (re-running the notebook's RAG cell,
or `python -m rag.build_index` while the chat UI runs). Chroma keeps each collection's
vector index in memory per process and doesn't see another process's writes, so:
- the collection handle is cached only while the build stamp (rag/build_index.py) is
  unchanged; when it changes, Chroma's per-process client cache is cleared and reopened;
- a query that hits a collection deleted by `--reset` reopens it once and retries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from chromadb.api.client import SharedSystemClient
from chromadb.errors import NotFoundError

import config
from rag.build_index import get_collection, stamp_path

_cache: dict[str, Any] = {"key": None, "col": None}


def _stamp() -> str:
    try:
        return stamp_path().read_text()
    except OSError:
        return ""


def reset_cache() -> None:
    """Forget the cached collection and Chroma's in-memory state for this process."""
    _cache.update(key=None, col=None)
    SharedSystemClient.clear_system_cache()


def _collection():
    key = (str(Path(config.CHROMA_DIR).resolve()), config.CHROMA_COLLECTION, _stamp())
    if _cache["col"] is None or _cache["key"] != key:
        if _cache["key"] is not None:  # index rebuilt (or moved) since we opened it
            reset_cache()
        col = get_collection()
        if col.count() == 0:
            raise RuntimeError("Policy index is empty. Run `python -m rag.build_index` first.")
        _cache.update(key=key, col=col)
    return _cache["col"]


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Top-k policy chunks for a query: [{id, text, source, section, score}]."""
    try:
        res = _collection().query(query_texts=[query], n_results=k)
    except NotFoundError:  # collection was dropped and recreated by another process
        reset_cache()
        res = _collection().query(query_texts=[query], n_results=k)
    out = []
    for i, doc in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        out.append({
            "id": res["ids"][0][i],
            "text": doc,
            "source": meta["source"],
            "section": meta["section"],
            "score": round(1 - res["distances"][0][i], 3),  # cosine similarity
        })
    return out

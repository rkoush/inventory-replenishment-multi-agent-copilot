"""Semantic retrieval over the policy index."""
from __future__ import annotations

from functools import lru_cache

from rag.build_index import get_collection


@lru_cache(maxsize=1)
def _collection():
    col = get_collection()
    if col.count() == 0:
        raise RuntimeError("Policy index is empty. Run `python -m rag.build_index` first.")
    return col


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Top-k policy chunks for a query: [{id, text, source, section, score}]."""
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

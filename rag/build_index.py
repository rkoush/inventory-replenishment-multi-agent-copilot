"""Build the Chroma vector index over the policy documents in data/policies/.

Chunking: one chunk per markdown `##` section, prefixed with the document title,
so each chunk is a self-contained rule with a clean citation (doc + section).
Embeddings: Chroma's default local model (all-MiniLM-L6-v2, ONNX) - no paid API.

Run:  python -m rag.build_index            (re-running is safe; it upserts and keeps the collection)
      python -m rag.build_index --reset    (drop and recreate the collection)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chromadb  # noqa: E402
from chromadb.utils import embedding_functions  # noqa: E402

import config  # noqa: E402


def chunk_markdown(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", text, flags=re.M)
    title = title_match.group(1).strip() if title_match else path.stem
    parts = re.split(r"^## ", text, flags=re.M)
    chunks = []
    preamble = re.sub(r"^# .+$", "", parts[0], flags=re.M).strip()
    if preamble:
        chunks.append({"section": "Overview", "text": f"{title}\n{preamble}"})
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        chunks.append({"section": heading.strip(), "text": f"{title} - {heading.strip()}\n{body.strip()}"})
    for i, c in enumerate(chunks):
        c.update(source=path.name, id=f"{path.stem}-{i}")
    return chunks


STAMP_NAME = "index_stamp.json"  # written next to the Chroma files after every change


def stamp_path() -> Path:
    return Path(config.CHROMA_DIR) / STAMP_NAME


@lru_cache(maxsize=1)
def _embedding_function():
    return embedding_functions.DefaultEmbeddingFunction()  # loads the ONNX model once per process


def get_collection(reset: bool = False):
    """Open (or create) the policy collection. reset=True drops and recreates it.

    Rebuilds normally keep the same collection (and its id): another process that
    holds a handle to it, e.g. a notebook kernel or the chat UI, keeps working.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:  # collection did not exist yet
            pass
    return client.get_or_create_collection(
        config.CHROMA_COLLECTION, embedding_function=_embedding_function(), metadata={"hnsw:space": "cosine"}
    )


def _fingerprint(chunks: list[dict]) -> str:
    return hashlib.sha256(json.dumps([[c["id"], c["text"], c["source"], c["section"]] for c in chunks]).encode()).hexdigest()


def sync_index(policy_dir: Path = config.POLICY_DIR, reset: bool = False) -> tuple[int, str]:
    """Bring the index in line with the policy docs: (chunk count, 'created' | 'unchanged' | 'updated' | 'rebuilt').

    Ids are deterministic (doc stem + section number), so a rebuild upserts every chunk and
    deletes ids that no longer exist instead of dropping the collection.
    """
    chunks = [c for p in sorted(Path(policy_dir).glob("*.md")) for c in chunk_markdown(p)]
    fp = _fingerprint(chunks)
    col = get_collection(reset=reset)
    ids = [c["id"] for c in chunks]
    existing = set(col.get(include=[])["ids"])
    try:
        stamp = json.loads(stamp_path().read_text())
    except (OSError, ValueError):
        stamp = {}
    if not reset and existing == set(ids) and stamp.get("fingerprint") == fp and stamp.get("collection_id") == str(col.id):
        return len(chunks), "unchanged"
    col.upsert(ids=ids, documents=[c["text"] for c in chunks],
               metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks])
    if stale := sorted(existing - set(ids)):
        col.delete(ids=stale)
    # Readers compare this stamp to decide when to reload (see rag/retriever.py).
    stamp_path().write_text(json.dumps({"fingerprint": fp, "collection_id": str(col.id), "built_at_ns": time.time_ns()}))
    return len(chunks), "rebuilt" if reset else "updated" if existing else "created"


def build_index(policy_dir: Path = config.POLICY_DIR, reset: bool = False) -> int:
    return sync_index(policy_dir, reset)[0]


if __name__ == "__main__":
    reset = "--reset" in sys.argv[1:]  # drop and recreate, e.g. after changing the embedding model
    n, status = sync_index(reset=reset)
    print(f"Indexed {n} policy chunks into {config.CHROMA_DIR} (collection '{config.CHROMA_COLLECTION}', {status})")

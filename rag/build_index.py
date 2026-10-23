"""Build the Chroma vector index over the policy documents in data/policies/.

Chunking: one chunk per markdown `##` section, prefixed with the document title,
so each chunk is a self-contained rule with a clean citation (doc + section).
Embeddings: Chroma's default local model (all-MiniLM-L6-v2, ONNX) - no paid API.

Run:  python -m rag.build_index
"""
from __future__ import annotations

import re
import sys
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


def get_collection(create: bool = False):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    ef = embedding_functions.DefaultEmbeddingFunction()
    if create:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:  # collection did not exist yet
            pass
    return client.get_or_create_collection(
        config.CHROMA_COLLECTION, embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )


def build_index(policy_dir: Path = config.POLICY_DIR) -> int:
    chunks = [c for p in sorted(policy_dir.glob("*.md")) for c in chunk_markdown(p)]
    col = get_collection(create=True)
    col.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
    )
    return len(chunks)


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} policy chunks into {config.CHROMA_DIR} (collection '{config.CHROMA_COLLECTION}')")

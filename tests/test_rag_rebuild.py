"""Rebuilding the policy index while a process holds it open must not break retrieval.

Regression: build_index used to drop and recreate the collection; a retriever that had
already opened it (or Chroma's per-process cache) then failed with
NotFoundError: Collection [...] does not exist.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import config
from rag import retriever
from rag.build_index import get_collection, sync_index

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def own_index(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    yield tmp_path / "chroma"
    retriever.reset_cache()


def _in_subprocess(code: str, chroma_dir: Path) -> str:
    env = {**os.environ, "CHROMA_DIR": str(chroma_dir)}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_rerunning_the_build_keeps_the_collection(own_index):
    assert sync_index()[1] == "created"
    first_id = get_collection().id
    assert retriever.retrieve("safety stock days", k=1)
    assert sync_index()[1] == "unchanged"
    assert get_collection().id == first_id
    assert retriever.retrieve("safety stock days", k=1)


def test_retrieve_survives_rebuilds_in_another_process(own_index, tmp_path):
    sync_index()
    before = retriever.retrieve("safety stock days by category", k=2)

    # 1. plain re-run in a subprocess (the notebook's old `!python -m rag.build_index`)
    _in_subprocess("from rag.build_index import sync_index; print(sync_index())", own_index)
    assert retriever.retrieve("safety stock days by category", k=2) == before

    # 2. --reset in a subprocess drops and recreates the collection (new id)
    _in_subprocess("import sys; sys.argv += ['--reset']; import runpy; runpy.run_module('rag.build_index', run_name='__main__')",
                   own_index)
    assert [h["id"] for h in retriever.retrieve("safety stock days by category", k=2)] == [h["id"] for h in before]

    # 3. a changed policy doc indexed by another process is visible here (no stale in-memory index)
    docs = tmp_path / "policies"
    docs.mkdir()
    for p in Path(config.POLICY_DIR).glob("*.md"):
        (docs / p.name).write_text(p.read_text())
    (docs / "zz_new.md").write_text("# Pallet rule\n## Pallets\nPurple pallets are stacked at most four high.\n")
    _in_subprocess(f"from rag.build_index import sync_index; print(sync_index({str(docs)!r}))", own_index)
    assert retriever.retrieve("how high can purple pallets be stacked", k=1)[0]["source"] == "zz_new.md"

    # 4. collection dropped behind our back without a stamp change: query retries once
    _in_subprocess("import chromadb, config\nc = chromadb.PersistentClient(path=str(config.CHROMA_DIR))\n"
                   "c.delete_collection(config.CHROMA_COLLECTION)\n"
                   "from rag.build_index import sync_index; from pathlib import Path\n"
                   "import json; from rag.build_index import stamp_path\n"
                   "s = stamp_path().read_text(); sync_index(); stamp_path().write_text(s)", own_index)
    assert retriever.retrieve("safety stock days by category", k=2)

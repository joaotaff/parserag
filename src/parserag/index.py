from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from parserag.config import configure_llama_index, index_dir
from parserag.datasets import get_adapter
from parserag.parse import parse_to_nodes

log = logging.getLogger(__name__)


def ingest_dataset(name: str, limit: int | None = None, rebuild: bool = False) -> list[dict]:
    """Index the dataset. Returns parse failures still outstanding."""
    configure_llama_index()
    adapter = get_adapter(name)
    adapter.prepare_pdfs(limit=limit)
    docs = adapter.iter_documents(limit=limit)
    index, indexed = load_or_create_index(name, rebuild=rebuild)
    failures = [] if rebuild else _load_failures(_paths(name)["failures"])
    for doc in docs:
        if not rebuild and doc.doc_id in indexed:
            log.info("Skip already indexed %s", doc.doc_id)
            continue
        try:
            nodes = parse_to_nodes(name, doc.doc_id, doc.pdf_path)
        except Exception as exc:
            log.exception("Parse failed for %s:%s — skipping", name, doc.doc_id)
            failures = record_failure(failures, doc.doc_id, f"{type(exc).__name__}: {exc}")
            _write_failures(name, failures)
            continue
        if not nodes:
            log.error("No text extracted for %s:%s", name, doc.doc_id)
            failures = record_failure(
                failures,
                doc.doc_id,
                "no text extracted; delete its pages.json cache to force a reparse",
            )
            _write_failures(name, failures)
            continue
        index.insert_nodes(nodes)
        indexed.add(doc.doc_id)
        failures = drop_failure(failures, doc.doc_id)
        _persist(name, index, indexed)
        _write_failures(name, failures)
        log.info("Indexed %s (%d nodes)", doc.doc_id, len(nodes))
    _report_failures(name, failures)
    return failures


def record_failure(rows: list[dict], doc_id: str, error: str) -> list[dict]:
    kept = [row for row in rows if row["doc_id"] != doc_id]
    kept.append({"doc_id": doc_id, "error": error})
    return kept


def drop_failure(rows: list[dict], doc_id: str) -> list[dict]:
    return [row for row in rows if row["doc_id"] != doc_id]


def load_index(name: str) -> VectorStoreIndex:
    configure_llama_index()
    paths = _paths(name)
    if not paths["docstore"].exists():
        raise FileNotFoundError(f"No index for {name}. Run ingest first.")
    index, _ = _open_index(name)
    if not index.docstore.docs:
        raise RuntimeError(f"Docstore for {name} has no nodes. Re-run ingest.")
    return index


def load_or_create_index(name: str, rebuild: bool = False) -> tuple[VectorStoreIndex, set[str]]:
    return _open_index(name, rebuild=rebuild)


def _open_index(name: str, rebuild: bool = False) -> tuple[VectorStoreIndex, set[str]]:
    # from_vector_store() discards storage_context, so BM25 would see an empty docstore.
    paths = _paths(name)
    if rebuild:
        _clear_index(paths)
    db = chromadb.PersistentClient(path=str(paths["chroma"]))
    collection = db.get_or_create_collection(name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    if paths["docstore"].exists() and not rebuild:
        docstore = SimpleDocumentStore.from_persist_path(str(paths["docstore"]))
        indexed = _load_manifest(paths["manifest"])
    else:
        docstore = SimpleDocumentStore()
        indexed = set()
    storage = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)
    index = VectorStoreIndex(nodes=[], storage_context=storage, store_nodes_override=True)
    return index, indexed


def _persist(name: str, index: VectorStoreIndex, indexed: set[str]) -> None:
    paths = _paths(name)
    index.docstore.persist(persist_path=str(paths["docstore"]))
    paths["manifest"].write_text(json.dumps(sorted(indexed)), encoding="utf-8")


def _paths(name: str) -> dict[str, Path]:
    root = index_dir(name)
    return {
        "chroma": root / "chroma",
        "docstore": root / "docstore.json",
        "manifest": root / "indexed.json",
        "failures": root / "failures.json",
    }


def _load_manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def _load_failures(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _write_failures(name: str, failures: list[dict]) -> None:
    path = _paths(name)["failures"]
    if failures:
        path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    elif path.exists():
        path.unlink()


def _report_failures(name: str, failures: list[dict]) -> None:
    if not failures:
        log.info("Ingest finished with no outstanding parse failures")
        return
    path = _paths(name)["failures"]
    lines = "\n".join(f"  {row['doc_id']}: {row['error']}" for row in failures)
    log.error("Ingest finished with %d parse failure(s). See %s\n%s", len(failures), path, lines)


def _clear_index(paths: dict[str, Path]) -> None:
    import shutil

    if paths["chroma"].exists():
        shutil.rmtree(paths["chroma"])
    for key in ("docstore", "manifest", "failures"):
        if paths[key].exists():
            paths[key].unlink()

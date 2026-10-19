from __future__ import annotations

import json
import logging
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from parserag.config import CHUNK_OVERLAP, CHUNK_SIZE, llama_cloud_api_key, parsed_dir

log = logging.getLogger(__name__)


def parse_to_nodes(dataset: str, doc_id: str, pdf_path: Path) -> list:
    pages = load_pages(dataset, doc_id, pdf_path)
    documents = [
        Document(
            text=page["text"],
            metadata={"dataset": dataset, "doc_id": doc_id, "page": page["page"]},
        )
        for page in pages
        if page["text"].strip()
    ]
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    for node in nodes:
        node.metadata.setdefault("dataset", dataset)
        node.metadata.setdefault("doc_id", doc_id)
        node.excluded_embed_metadata_keys = ["dataset"]
    return nodes


def load_pages(dataset: str, doc_id: str, pdf_path: Path) -> list[dict]:
    cache = parsed_dir(dataset, doc_id) / "pages.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    log.info("LlamaParse %s:%s", dataset, doc_id)
    pages = _parse_pdf(pdf_path)
    cache.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    return pages


def _parse_pdf(pdf_path: Path) -> list[dict]:
    # JSON carries an explicit page number per page. The download includes layout
    # data, so a long filing can sit on this call for several minutes.
    log.info("Downloading LlamaParse JSON for %s", pdf_path.name)
    raw = _llama_parse().get_json_result(str(pdf_path))
    pages = _pages_from_json(raw)
    if not pages:
        raise RuntimeError(f"LlamaParse returned no pages for {pdf_path}")
    return pages


def _llama_parse():
    from llama_cloud_services import LlamaParse

    return LlamaParse(
        api_key=llama_cloud_api_key(),
        result_type="markdown",
        ignore_errors=False,
        verbose=False,
    )


def _pages_from_json(raw) -> list[dict]:
    job = raw[0] if isinstance(raw, list) and raw else raw
    out: list[dict] = []
    for i, page in enumerate(_extract_pages(job)):
        num, text = _page_fields(page, i)
        out.append({"page": int(num), "text": text or ""})
    if out and min(item["page"] for item in out) == 0:
        for item in out:
            item["page"] += 1
    return out


def _extract_pages(job) -> list:
    if job is None:
        return []
    if isinstance(job, dict):
        return job.get("pages") or []
    if hasattr(job, "pages") and job.pages is not None:
        return list(job.pages)
    if hasattr(job, "model_dump"):
        dumped = job.model_dump()
        return dumped.get("pages") or []
    raise ValueError(f"Unexpected LlamaParse result: {type(job)}")


def _page_fields(page, index: int) -> tuple[int, str]:
    if isinstance(page, dict):
        num = page.get("page", page.get("page_number", index + 1))
        text = page.get("md") or page.get("markdown") or page.get("text") or ""
        return int(num), text
    num = getattr(page, "page", None) or getattr(page, "page_number", index + 1)
    text = getattr(page, "md", None) or getattr(page, "markdown", None) or getattr(page, "text", "") or ""
    return int(num), text

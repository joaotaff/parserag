from __future__ import annotations

import logging
from pathlib import Path

from parserag.config import pdf_dir
from parserag.datasets.base import DatasetAdapter, GoldPage, Question, SourceDoc

log = logging.getLogger(__name__)

HF_DATASET = "jordyvl/DUDE_loader"
# Paths, not in-memory PDF bytes. Binaries are extracted into the HF cache.
HF_CONFIG = "Amazon_original"
SPLIT = "val"


class DudeAdapter(DatasetAdapter):
    name = "dude"

    def __init__(self) -> None:
        self._split = None

    def prepare_pdfs(self, limit: int | None = None) -> None:
        dest = pdf_dir(self.name)
        written: set[str] = set()
        for row in self._hf():
            doc_id = str(row["docId"])
            if doc_id in written:
                continue
            path = dest / f"{doc_id}.pdf"
            if not path.exists() or path.stat().st_size == 0:
                _write_pdf(path, row["document"])
            written.add(doc_id)
            if limit and len(written) >= limit:
                break

    def iter_documents(self, limit: int | None = None) -> list[SourceDoc]:
        dest = pdf_dir(self.name)
        docs: list[SourceDoc] = []
        seen: set[str] = set()
        for row in self._hf():
            doc_id = str(row["docId"])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            docs.append(SourceDoc(doc_id=doc_id, pdf_path=dest / f"{doc_id}.pdf"))
            if limit and len(docs) >= limit:
                break
        return docs

    def iter_questions(self, limit: int | None = None) -> list[Question]:
        questions: list[Question] = []
        for row in self._hf():
            doc_id = str(row["docId"])
            questions.append(
                Question(
                    question_id=str(row["questionId"]),
                    question=row["question"],
                    gold_pages=_gold_pages(doc_id, row),
                )
            )
            if limit and len(questions) >= limit:
                break
        return questions

    def _hf(self):
        if self._split is None:
            from datasets import load_dataset

            ds = load_dataset(HF_DATASET, HF_CONFIG, trust_remote_code=True)
            if SPLIT in ds:
                split = ds[SPLIT]
            else:
                first = ds[list(ds.keys())[0]]
                split = first.filter(
                    lambda row: str(row.get("data_split", "")).lower() in {SPLIT, "validation"}
                )
            self._split = split
        return self._split


def _write_pdf(path: Path, blob) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(blob, (bytes, bytearray, memoryview)):
        path.write_bytes(bytes(blob))
        return
    if isinstance(blob, str):
        src = Path(blob)
        if src.exists():
            path.write_bytes(src.read_bytes())
            return
        raise FileNotFoundError(f"DUDE PDF path missing: {blob}")
    raise TypeError(f"Cannot write PDF from {type(blob)}")


def _gold_pages(doc_id: str, row: dict) -> list[GoldPage]:
    if str(row.get("answer_type", "")).lower() in {"not-answerable", "none", "unanswerable"}:
        return []
    pages = _bbox_pages(row.get("answers_page_bounding_boxes"))
    if not pages:
        return []
    if min(pages) == 0:
        pages = [p + 1 for p in pages]
    seen: set[int] = set()
    gold: list[GoldPage] = []
    for page in pages:
        if page in seen:
            continue
        seen.add(page)
        gold.append(GoldPage(doc_id=doc_id, page=page))
    return gold


def _bbox_pages(raw) -> list[int]:
    pages: list[int] = []

    def walk(value) -> None:
        if isinstance(value, dict) and "page" in value and value["page"] is not None:
            pages.append(int(value["page"]))
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(raw)
    return pages

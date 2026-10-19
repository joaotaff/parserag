from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from parserag.config import pdf_dir
from parserag.datasets.base import DatasetAdapter, GoldPage, Question, SourceDoc

log = logging.getLogger(__name__)

HF_DATASET = "PatronusAI/financebench"
GITHUB_PDF = "https://github.com/patronus-ai/financebench/raw/main/pdfs/{doc_id}.pdf"


class FinanceBenchAdapter(DatasetAdapter):
    name = "financebench"

    def prepare_pdfs(self, limit: int | None = None) -> None:
        for doc in self.iter_documents(limit=limit):
            if doc.pdf_path.exists() and doc.pdf_path.stat().st_size > 0:
                continue
            url = GITHUB_PDF.format(doc_id=doc.doc_id)
            log.info("Downloading %s", url)
            _download(url, doc.pdf_path)

    def iter_documents(self, limit: int | None = None) -> list[SourceDoc]:
        dest = pdf_dir(self.name)
        docs = [SourceDoc(doc_id=doc_id, pdf_path=dest / f"{doc_id}.pdf") for doc_id in self._doc_ids()]
        return docs[:limit] if limit else docs

    def iter_questions(self, limit: int | None = None) -> list[Question]:
        rows = self._rows()
        questions: list[Question] = []
        for row in rows:
            gold: list[GoldPage] = []
            seen: set[tuple[str, int]] = set()
            for ev in row.get("evidence") or []:
                doc_id = ev.get("evidence_doc_name") or row["doc_name"]
                # FinanceBench JSONL evidence_page_num is 0-indexed.
                page = int(ev["evidence_page_num"]) + 1
                key = (doc_id, page)
                if key in seen:
                    continue
                seen.add(key)
                gold.append(GoldPage(doc_id=doc_id, page=page))
            questions.append(
                Question(
                    question_id=str(row["financebench_id"]),
                    question=row["question"],
                    gold_pages=gold,
                    gold_answers=[str(row["answer"])],
                    domain=str(row.get("question_type") or "unknown"),
                    metadata={
                        "company": row.get("company"),
                        "doc_name": row.get("doc_name"),
                        "question_reasoning": row.get("question_reasoning"),
                    },
                )
            )
            if limit and len(questions) >= limit:
                break
        return questions

    def _doc_ids(self) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for row in self._rows():
            for doc_id in [row.get("doc_name"), *[e.get("evidence_doc_name") for e in row.get("evidence") or []]]:
                if doc_id and doc_id not in seen:
                    seen.add(doc_id)
                    ids.append(doc_id)
        return ids

    def _rows(self) -> list[dict]:
        if not hasattr(self, "_cached_rows"):
            from datasets import load_dataset

            ds = load_dataset(HF_DATASET, split="train")
            self._cached_rows = [dict(row) for row in ds]
        return self._cached_rows


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "parserag"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as f:
        f.write(resp.read())

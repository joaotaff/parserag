# Datasets

Adapters live in `src/parserag/datasets/`. Add a new dataset by implementing `DatasetAdapter` (`prepare_pdfs`, `iter_documents`, `iter_questions`) and registering it in `ADAPTERS`.

Internal page numbers are **1-indexed** on every `GoldPage` and every indexed node.

## FinanceBench

- Questions: Hugging Face `PatronusAI/financebench` (150 open-source items).
- PDFs: `https://github.com/patronus-ai/financebench/raw/main/pdfs/{doc_id}.pdf` into `data/pdfs/financebench/`.
- Gold pages: each evidence row's `evidence_doc_name` + `evidence_page_num`. The JSONL field is **0-indexed**; the adapter adds 1.

Unique PDFs are the union of `doc_name` and every `evidence_doc_name`.

## DUDE (validation)

- Hugging Face `jordyvl/DUDE_loader`, config `Amazon_original`, split `val` (~749 PDFs, ~6.3k questions).
- PDFs are copied from the loader's extracted paths into `data/pdfs/dude/`.
- Gold pages: unique `page` values in `answers_page_bounding_boxes` for that `docId`.
- `not-answerable` (and any question with no bbox pages) get empty `gold_pages`. They are still sent to the LLM, but **Recall@k ignores them**.

If a DUDE row's bbox pages include `0`, they are treated as 0-indexed and shifted by +1. Otherwise pages are left as-is (DocVQA-style 1-indexed).

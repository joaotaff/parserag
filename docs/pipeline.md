# Pipeline

ParseRag is a hybrid RAG loop: LlamaParse turns PDFs into markdown pages, Chroma stores dense embeddings, BM25 stores the same nodes as text, Reciprocal Rank Fusion (RRF) merges the two ranked lists, and `gpt-4o-mini` writes a JSON answer. Evaluation scores retrieval and rule-based answer quality.

```
PDFs -> LlamaParse (cached) -> page nodes
                              -> Chroma (dense)
                              -> SimpleDocumentStore (BM25)
Question -> Chroma top-20 + BM25 top-20 -> RRF top-8 -> LLM JSON -> JSONL
JSONL -> Recall@k on gold (doc_id, page)
```

How a PDF becomes cached markdown and page nodes is in `docs/parsing.md`.

## Commands

From the repo root, with `OPENAI_API_KEY` and `LLAMA_CLOUD_API_KEY` in `.env`:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m parserag ingest --dataset financebench
.venv\Scripts\python -m parserag run --dataset financebench
.venv\Scripts\python -m parserag eval --dataset financebench
.venv\Scripts\python tests/test_retrieval.py
```

On Unix, use `.venv/bin/python` instead of `.venv\Scripts\python`.

Run writes timestamped files under `results/<dataset>/run/`: a JSONL checkpoint updated after every question and a PageIndex-style JSON snapshot containing `run_config` and `results`. Eval scores the latest snapshot and writes a timestamp-matched `.metrics.json` under `results/<dataset>/eval/`.

`--limit N` caps unique PDFs on ingest and questions on run (smoke tests). `--rebuild` wipes the Chroma collection and docstore for that dataset.

DUDE validation is ~749 PDFs / ~6.3k questions. Ingest it the same way (`--dataset dude`) only after FinanceBench has proven the loop; LlamaParse credits and time add up.

## Defaults

| Setting | Value |
| --- | --- |
| Embeddings | `text-embedding-3-small` |
| LLM | `gpt-4o-mini` |
| Per-retriever k | 20 |
| Fused top-k | 8 |
| RRF k | 60 |
| Chunk size / overlap | 1024 / 128 |
| Query rewriting | off (`num_queries=1`) |

Datasets use **separate** indexes. Each question retrieves over **all documents in that dataset**, not the gold PDF only.

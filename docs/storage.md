# Parse cache and indexes

## LlamaParse cache

`data/parsed/{dataset}/{doc_id}/pages.json` stores `[{page, text}, ...]` markdown pages.

- A cache hit never calls LlamaParse.
- A parse failure is logged and that PDF is skipped; ingest continues.
- Re-running ingest after a partial DUDE run only pays for uncached files.

LlamaParse is requested as markdown, one page per document. Empty pages are dropped. Pages longer than 1024 tokens are split with `SentenceSplitter` (overlap 128); `dataset`, `doc_id`, and `page` are copied onto every chunk.

## Chroma + BM25

Each dataset gets `data/indexes/{dataset}/`:

- `chroma/` — `PersistentClient` collection named after the dataset (dense vectors only).
- `docstore.json` — LlamaIndex `SimpleDocumentStore` with the same node text BM25 needs.
- `indexed.json` — doc ids already embedded, so ingest can resume.

Chroma does **not** store raw text. Building BM25 from the vector index alone would yield an empty corpus. The docstore is the source of truth for `BM25Retriever`.

On ingest, nodes are inserted incrementally and the docstore/manifest are persisted after each PDF.

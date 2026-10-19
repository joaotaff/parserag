# Parse cache and indexes

## LlamaParse cache

`data/parsed/{dataset}/{doc_id}/pages.json` stores `[{page, text}, ...]` markdown pages.

- A cache hit never calls LlamaParse.
- A parse that raises is not cached. Ingest records it in `data/indexes/{dataset}/failures.json`, continues, and exits with status 1. The next ingest retries that PDF.
- A parse that returns no text is cached and listed in the same file. Delete that doc's `pages.json` to force a reparse.
- The command exits 0 only when `failures.json` is absent. Re-running ingest after a partial DUDE run only pays for uncached files.

LlamaParse is requested as a JSON result. Each page's markdown is stored with the page number from that result (shifted to 1-indexed if the result is 0-indexed). The JSON download includes layout data, so a long filing can take several minutes with no further log lines. Empty pages are dropped. Pages longer than 1024 tokens are split with `SentenceSplitter` (overlap 128); `dataset`, `doc_id`, and `page` are copied onto every chunk.

## Chroma + BM25

Each dataset gets `data/indexes/{dataset}/`:

- `chroma/` — `PersistentClient` collection named after the dataset (dense vectors only).
- `docstore.json` — LlamaIndex `SimpleDocumentStore` with the same node text BM25 needs.
- `indexed.json` — doc ids already embedded, so ingest can resume.
- `failures.json` — parses that did not make it into the index. Written after each failure, so it is still there if ingest is interrupted. Removed when the list is empty.

Chroma does **not** store raw text. Building BM25 from the vector index alone would yield an empty corpus. The docstore is the source of truth for `BM25Retriever`.

On ingest, nodes are inserted incrementally and the docstore/manifest are persisted after each PDF.

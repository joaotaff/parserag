# Evaluation

The output convention follows PageIndex while retaining a crash-safe checkpoint:

- `results/financebench/run/financebench_YYYYMMDD_HHMMSS.jsonl` is flushed after every question.
- `results/financebench/run/financebench_YYYYMMDD_HHMMSS.json` is written when the run completes as `{run_config, results}`.
- `results/financebench/eval/financebench_YYYYMMDD_HHMMSS.metrics.json` contains the matching evaluation.

## Per-question result

Each run record contains:

```json
{
  "dataset": "financebench",
  "question_id": "...",
  "question": "...",
  "answer": {"answer": "...", "citations": [{"doc_id": "...", "page": 1}]},
  "retrieved": [{"doc_id": "...", "page": 12, "score": 0.03, "text": "..."}],
  "gold_pages": [{"doc_id": "...", "page": 12}],
  "k": 8
}
```

The actual record also includes gold answers, domain/type metadata, a joined retrieved context, retrieved document/node/page IDs, latency, retrieval count, and `pipeline_error`. `retrieved` is the RRF-fused list sent to the LLM (`fused_k`, default 8).

## Recall@k

`python -m parserag eval --dataset financebench`

By default eval selects the latest completed JSON snapshot. `--pred` can score a particular snapshot or an interrupted JSONL checkpoint. `--out` overrides the metrics path.

A question **counts** only if `gold_pages` is non-empty.

- **recall_at_k** (page): hit if any gold `(doc_id, page)` appears in the fused top-k.
- **doc_recall_at_k**: hit if any gold `doc_id` appears, ignoring page.
- **precision/recall/nDCG@3,5,10**: binary relevance over unique `(doc_id, page)` results.
- **EM/F1/numeric_match**: rule-based answer metrics against dataset gold answers.

Metrics are reported overall, per dataset domain/type, and per question. `--k` changes the primary Recall@k; ranked metrics remain at 3, 5, and 10. An LLM judge is deliberately not included.

RRF itself is `score(d) = Σ_r 1 / (60 + rank_r(d))`, matching LlamaIndex `QueryFusionRetriever` with `mode="reciprocal_rerank"` and `num_queries=1` (no query rewriting).

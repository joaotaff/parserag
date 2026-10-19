# Evaluation

v1 scores **retrieval**, not answer quality. The JSON `answer` field is written for a later judge / ANLS pass and is ignored here.

## Prediction JSONL

One object per question, written by `python -m parserag run`:

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

`retrieved` is the RRF-fused list sent to the LLM (`fused_k`, default 8).

## Recall@k

`python -m parserag eval --pred results/financebench.jsonl`

A question **counts** only if `gold_pages` is non-empty.

- **recall_at_k** (page): hit if any gold `(doc_id, page)` appears in the fused top-k.
- **doc_recall_at_k**: hit if any gold `doc_id` appears, ignoring page.

`--k` truncates `retrieved` before scoring. If omitted, k is the length of the first record's retrieved list.

RRF itself is `score(d) = Σ_r 1 / (60 + rank_r(d))`, matching LlamaIndex `QueryFusionRetriever` with `mode="reciprocal_rerank"` and `num_queries=1` (no query rewriting).

from __future__ import annotations

import json
from pathlib import Path


def score_predictions(records: list[dict], k: int | None = None) -> dict:
    """Page-level Recall@k. Questions with no gold pages are excluded, not misses."""
    n = 0
    page_hits = 0
    doc_hits = 0
    for record in records:
        gold = [(g["doc_id"], int(g["page"])) for g in record.get("gold_pages") or [] if g.get("page") is not None]
        if not gold:
            continue
        n += 1
        retrieved = record.get("retrieved") or []
        if k is not None:
            retrieved = retrieved[:k]
        got = [(r.get("doc_id"), r.get("page")) for r in retrieved]
        got_pages = {(d, int(p)) for d, p in got if d is not None and p is not None}
        got_docs = {d for d, _ in got_pages}
        gold_pages = set(gold)
        gold_docs = {d for d, _ in gold_pages}
        if gold_pages & got_pages:
            page_hits += 1
        if gold_docs & got_docs:
            doc_hits += 1
    return {
        "n": n,
        "k": k if k is not None else (records[0].get("k") if records else None),
        "recall_at_k": (page_hits / n) if n else 0.0,
        "doc_recall_at_k": (doc_hits / n) if n else 0.0,
        "page_hits": page_hits,
        "doc_hits": doc_hits,
    }


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def eval_file(path: Path, k: int | None = None) -> dict:
    records = load_jsonl(path)
    if k is None and records:
        k = len(records[0].get("retrieved") or [])
    result = score_predictions(records, k=k)
    result["pred"] = str(path)
    return result

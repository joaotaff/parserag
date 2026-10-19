from __future__ import annotations

import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_KS = (3, 5, 10)


def score_predictions(
    records: list[dict],
    k: int | None = None,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:
    """Page retrieval plus answer metrics, with dataset-domain breakdowns."""
    primary_k = k or (records[0].get("k") if records else None) or 8
    overall = _aggregate(records, primary_k, ks)
    domains: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        domains[str(record.get("domain") or "unknown")].append(record)
    return {
        "overall": overall,
        "per_domain": {
            domain: _aggregate(domain_records, primary_k, ks)
            for domain, domain_records in sorted(domains.items())
        },
        "per_question": [_per_question(record, primary_k) for record in records],
    }


def _aggregate(records: list[dict], primary_k: int, ks: tuple[int, ...]) -> dict:
    retrieval_records = [record for record in records if record.get("gold_pages")]
    answer_records = [record for record in records if record.get("gold_answers")]
    page_hits = sum(_page_hit(record, primary_k) for record in retrieval_records)
    doc_hits = sum(_doc_hit(record, primary_k) for record in retrieval_records)
    ems, f1s, numeric = zip(
        *[_answer_scores(record) for record in answer_records],
        strict=False,
    ) if answer_records else ((), (), ())
    failures = sum(bool(record.get("pipeline_error")) for record in records)
    latencies = [float(record["question_latency_ms"]) for record in records if record.get("question_latency_ms") is not None]
    retrieval_metrics = _retrieval_at_ks(retrieval_records, ks)
    return {
        "question_count": len(records),
        "num_valid_answer": len(answer_records),
        "em": _mean(ems),
        "f1": _mean(f1s),
        "numeric_match": _mean(numeric),
        "query_latency_ms": _mean(latencies),
        "retrieval_num": _mean([len(record.get("retrieved") or []) for record in records]),
        "pipeline_failures": failures,
        "pipeline_error_rate": failures / len(records) if records else 0.0,
        "retrieval_queries_scored": len(retrieval_records),
        "k": primary_k,
        "recall_at_k": page_hits / len(retrieval_records) if retrieval_records else 0.0,
        "doc_recall_at_k": doc_hits / len(retrieval_records) if retrieval_records else 0.0,
        "page_hits": page_hits,
        "doc_hits": doc_hits,
        "retrieval_metrics": retrieval_metrics,
    }


def _retrieval_at_ks(records: list[dict], ks: tuple[int, ...]) -> dict:
    metrics: dict[str, float | int | list[int] | str] = {
        "qrel_method": "gold_doc_id_and_page",
        "ks": list(ks),
        "queries_scored": len(records),
    }
    for k in ks:
        precision, recall, ndcg = [], [], []
        for record in records:
            gold = _gold_pages(record)
            ranked = _ranked_pages(record)[:k]
            relevant = [1 if page in gold else 0 for page in ranked]
            precision.append(sum(relevant) / k)
            recall.append(len(set(ranked) & gold) / len(gold))
            dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevant))
            idcg = sum(1 / math.log2(rank + 2) for rank in range(min(len(gold), k)))
            ndcg.append(dcg / idcg if idcg else 0.0)
        metrics[f"precision@{k}"] = _mean(precision)
        metrics[f"recall@{k}"] = _mean(recall)
        metrics[f"ndcg@{k}"] = _mean(ndcg)
    return metrics


def _per_question(record: dict, k: int) -> dict:
    em, f1, numeric = _answer_scores(record)
    has_gold_pages = bool(record.get("gold_pages"))
    return {
        "question_id": record.get("question_id"),
        "domain": record.get("domain") or "unknown",
        "em": em,
        "f1": f1,
        "numeric_match": numeric,
        "retrieval_scored": has_gold_pages,
        "recall_at_k": _page_hit(record, k) if has_gold_pages else None,
        "doc_recall_at_k": _doc_hit(record, k) if has_gold_pages else None,
        "question_latency_ms": record.get("question_latency_ms"),
        "retrieval_num": len(record.get("retrieved") or []),
        "pipeline_error": record.get("pipeline_error"),
    }


def _page_hit(record: dict, k: int) -> int:
    return int(bool(set(_ranked_pages(record)[:k]) & _gold_pages(record)))


def _doc_hit(record: dict, k: int) -> int:
    gold_docs = {doc_id for doc_id, _ in _gold_pages(record)}
    return int(any(doc_id in gold_docs for doc_id, _ in _ranked_pages(record)[:k]))


def _gold_pages(record: dict) -> set[tuple[str, int]]:
    return {
        (str(page["doc_id"]), int(page["page"]))
        for page in record.get("gold_pages") or []
        if page.get("doc_id") is not None and page.get("page") is not None
    }


def _ranked_pages(record: dict) -> list[tuple[str, int]]:
    ranked, seen = [], set()
    for item in record.get("retrieved") or []:
        if item.get("doc_id") is None or item.get("page") is None:
            continue
        page = (str(item["doc_id"]), int(item["page"]))
        if page not in seen:
            seen.add(page)
            ranked.append(page)
    return ranked


def _answer_scores(record: dict) -> tuple[float, float, float]:
    answer = record.get("answer") or {}
    prediction = str(answer.get("answer", "") if isinstance(answer, dict) else answer)
    golds = [str(gold) for gold in record.get("gold_answers") or []]
    em = max((_normalize(prediction) == _normalize(gold) for gold in golds), default=False)
    f1 = max((_token_f1(prediction, gold) for gold in golds), default=0.0)
    numeric = max((_numeric_match(prediction, gold) for gold in golds), default=0.0)
    return float(em), f1, numeric


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(char if char not in string.punctuation else " " for char in text)
    return " ".join(token for token in text.split() if token not in {"a", "an", "the"})


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens, gold_tokens = _normalize(prediction).split(), _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred_tokens), common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _numeric_match(prediction: str, gold: str) -> float:
    pattern = r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?"
    clean = lambda value: value.replace("$", "").replace(",", "").replace("%", "")
    pred_numbers = [clean(value) for value in re.findall(pattern, prediction)]
    gold_numbers = [clean(value) for value in re.findall(pattern, gold)]
    return float(bool(gold_numbers) and pred_numbers == gold_numbers)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_run(path: Path) -> tuple[dict, list[dict]]:
    if path.suffix == ".jsonl":
        return {}, load_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return {}, payload
    return payload.get("run_config") or {}, payload.get("results") or []


def eval_file(path: Path, k: int | None = None) -> dict:
    run_config, records = load_run(path)
    result = score_predictions(records, k=k)
    result["run_config"] = run_config | {"predictions_path": str(path)}
    return result

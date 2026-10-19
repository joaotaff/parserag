from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from parserag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    FUSED_K,
    LLM_MODEL,
    RETRIEVER_K,
    RRF_K,
    configure_llama_index,
    eval_path,
    latest_run_path,
    run_paths,
)
from parserag.datasets import get_adapter
from parserag.eval_retrieval import eval_file
from parserag.generate import generate_answer
from parserag.index import ingest_dataset, load_index
from parserag.retrieve import build_retriever, retrieved_records

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="parserag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Parse PDFs and build Chroma + BM25 indexes")
    ingest.add_argument("--dataset", required=True, choices=["financebench", "dude"])
    ingest.add_argument("--limit", type=int, default=None, help="Max unique PDFs")
    ingest.add_argument("--rebuild", action="store_true")

    run = sub.add_parser("run", help="Retrieve + generate checkpoint JSONL and snapshot JSON")
    run.add_argument("--dataset", required=True, choices=["financebench", "dude"])
    run.add_argument("--out", default=None, help="Snapshot JSON (a sibling JSONL checkpoint is also written)")
    run.add_argument("--limit", type=int, default=None, help="Max questions")

    ev = sub.add_parser("eval", help="Score a run JSON/JSONL")
    ev.add_argument("--dataset", required=True, choices=["financebench", "dude"])
    ev.add_argument("--pred", default=None, help="Default: latest timestamped run JSON")
    ev.add_argument("--k", type=int, default=None)
    ev.add_argument("--out", default=None, help="Default: matching timestamped .metrics.json")

    args = parser.parse_args(argv)
    if args.cmd == "ingest":
        failures = ingest_dataset(args.dataset, limit=args.limit, rebuild=args.rebuild)
        if failures:
            raise SystemExit(1)
    elif args.cmd == "run":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.out:
            snapshot = Path(args.out)
            snapshot = snapshot if snapshot.suffix == ".json" else snapshot.with_suffix(".json")
            checkpoint = snapshot.with_suffix(".jsonl")
        else:
            checkpoint, snapshot = run_paths(args.dataset, timestamp)
        if cmd_run(args.dataset, checkpoint, snapshot, timestamp, limit=args.limit):
            raise SystemExit(1)
    elif args.cmd == "eval":
        pred = Path(args.pred) if args.pred else latest_run_path(args.dataset)
        out = Path(args.out) if args.out else eval_path(args.dataset, pred)
        result = eval_file(pred, k=args.k)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        log.info("Wrote %s", out)
        print(json.dumps(result, indent=2))


def cmd_run(
    dataset: str,
    checkpoint: Path,
    snapshot: Path,
    timestamp: str,
    limit: int | None = None,
) -> int:
    """Write crash-safe JSONL during the run and a PageIndex-style JSON snapshot."""
    started = time.perf_counter()
    configure_llama_index()
    adapter = get_adapter(dataset)
    index = load_index(dataset)
    retriever = build_retriever(index)
    questions = adapter.iter_questions(limit=limit)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    results: list[dict] = []
    with checkpoint.open("w", encoding="utf-8") as handle:
        for question in questions:
            question_started = time.perf_counter()
            pipeline_error = None
            nodes = []
            try:
                nodes = retriever.retrieve(question.question)
                answer = generate_answer(question.question, nodes)
            except Exception as exc:
                failed += 1
                log.exception("Pipeline failed for %s", question.question_id)
                pipeline_error = f"{type(exc).__name__}: {exc}"
                answer = {"answer": "", "citations": [], "error": f"{type(exc).__name__}: {exc}"}
            retrieved = retrieved_records(nodes)
            latency_ms = round((time.perf_counter() - question_started) * 1000, 1)
            record = {
                "dataset": dataset,
                "question_id": question.question_id,
                "question": question.question,
                "answer": answer,
                "predicted_answer": answer.get("answer", ""),
                "gold_answers": question.gold_answers,
                "gold_pages": [{"doc_id": g.doc_id, "page": g.page} for g in question.gold_pages],
                "domain": question.domain,
                "metadata": question.metadata,
                "retrieved": retrieved,
                "retrieved_context": "\n\n".join(item["text"] for item in retrieved),
                "retrieved_doc_ids": list(dict.fromkeys(item["doc_id"] for item in retrieved)),
                "retrieved_node_ids": [
                    {"doc_id": item["doc_id"], "node_id": item["node_id"], "page": item["page"]}
                    for item in retrieved
                ],
                "question_latency_ms": latency_ms,
                "retrieval_num": len(retrieved),
                "pipeline_error": pipeline_error,
                "k": FUSED_K,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            results.append(record)
            log.info("Wrote %s", question.question_id)
    total_time = round(time.perf_counter() - started, 4)
    run_config = {
        "dataset": dataset,
        "output_path": str(snapshot),
        "checkpoint_path": str(checkpoint),
        "timestamp": timestamp,
        "model": LLM_MODEL,
        "embedding_model": EMBED_MODEL,
        "index_dir": f"data/indexes/{dataset}",
        "limit": limit,
        "question_count": len(questions),
        "retriever_k": RETRIEVER_K,
        "fused_k": FUSED_K,
        "rrf_k": RRF_K,
        "num_queries": 1,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_time_sec": total_time,
        "avg_retrieval_num": (
            sum(record["retrieval_num"] for record in results) / len(results) if results else 0.0
        ),
        "pipeline_failures": failed,
    }
    snapshot.write_text(
        json.dumps({"run_config": run_config, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote snapshot %s", snapshot)
    if failed:
        log.error("Run finished with %d pipeline failure(s); inspect pipeline_error in %s", failed, snapshot)
    return failed

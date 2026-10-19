from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from parserag.config import FUSED_K, configure_llama_index
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

    run = sub.add_parser("run", help="Retrieve + generate JSONL predictions")
    run.add_argument("--dataset", required=True, choices=["financebench", "dude"])
    run.add_argument("--out", required=True)
    run.add_argument("--limit", type=int, default=None, help="Max questions")

    ev = sub.add_parser("eval", help="Score retrieval Recall@k from a prediction JSONL")
    ev.add_argument("--pred", required=True)
    ev.add_argument("--k", type=int, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "ingest":
        failures = ingest_dataset(args.dataset, limit=args.limit, rebuild=args.rebuild)
        if failures:
            raise SystemExit(1)
    elif args.cmd == "run":
        cmd_run(args.dataset, Path(args.out), limit=args.limit)
    elif args.cmd == "eval":
        result = eval_file(Path(args.pred), k=args.k)
        print(json.dumps(result, indent=2))


def cmd_run(dataset: str, out: Path, limit: int | None = None) -> None:
    configure_llama_index()
    adapter = get_adapter(dataset)
    index = load_index(dataset)
    retriever = build_retriever(index)
    questions = adapter.iter_questions(limit=limit)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for question in questions:
            nodes = retriever.retrieve(question.question)
            answer = generate_answer(question.question, nodes)
            record = {
                "dataset": dataset,
                "question_id": question.question_id,
                "question": question.question,
                "answer": answer,
                "retrieved": retrieved_records(nodes),
                "gold_pages": [{"doc_id": g.doc_id, "page": g.page} for g in question.gold_pages],
                "k": FUSED_K,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            log.info("Wrote %s", question.question_id)

"""RRF ranking + Recall@k oracle. Run: python tests/test_retrieval.py"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from parserag import config
from parserag.eval_retrieval import score_predictions
from parserag.index import drop_failure, record_failure
from parserag.retrieve import rrf_fuse


def test_rrf_shared_doc_ranks_first() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["a", "c", "d"]], k=60, top_n=4)
    order = [doc_id for doc_id, _ in fused]
    scores = dict(fused)
    assert order[0] == "a"
    assert scores["a"] == 1 / 61 + 1 / 61
    assert scores["c"] == 1 / 63 + 1 / 62
    assert scores["a"] > scores["c"] > scores["b"]
    assert "d" in scores


def test_recall_at_k_page_and_doc() -> None:
    records = [
        {"gold_pages": [], "retrieved": [{"doc_id": "x", "page": 1}]},
        {
            "gold_pages": [{"doc_id": "d", "page": 2}],
            "retrieved": [{"doc_id": "d", "page": 2}, {"doc_id": "other", "page": 1}],
        },
        {
            "gold_pages": [{"doc_id": "d", "page": 3}],
            "retrieved": [{"doc_id": "d", "page": 9}],
        },
        {
            "gold_pages": [{"doc_id": "miss", "page": 1}],
            "retrieved": [{"doc_id": "nope", "page": 1}, {"doc_id": "miss", "page": 1}],
        },
    ]
    result = score_predictions(records, k=8)
    overall = result["overall"]
    assert overall["retrieval_queries_scored"] == 3
    assert overall["page_hits"] == 2
    assert overall["doc_hits"] == 3
    assert overall["recall_at_k"] == 2 / 3
    assert overall["doc_recall_at_k"] == 1.0

    truncated = score_predictions(records, k=1)
    assert truncated["overall"]["page_hits"] == 1
    assert truncated["overall"]["doc_hits"] == 2


def test_answer_and_ranked_retrieval_metrics() -> None:
    records = [
        {
            "question_id": "q1",
            "domain": "numeric",
            "gold_answers": ["$1,234 million"],
            "answer": {"answer": "$1,234 million"},
            "gold_pages": [{"doc_id": "d", "page": 2}],
            "retrieved": [
                {"doc_id": "d", "page": 2},
                {"doc_id": "d", "page": 2},
                {"doc_id": "x", "page": 1},
            ],
        }
    ]
    result = score_predictions(records)
    assert result["overall"]["em"] == 1.0
    assert result["overall"]["numeric_match"] == 1.0
    assert result["overall"]["retrieval_metrics"]["recall@3"] == 1.0
    assert result["per_domain"]["numeric"]["question_count"] == 1


def test_retrieval_only_records_skip_answer_metrics() -> None:
    records = [
        {
            "question_id": "q1",
            "domain": "numeric",
            "gold_answers": ["42"],
            "answer_generation_enabled": False,
            "gold_pages": [{"doc_id": "d", "page": 2}],
            "retrieved": [{"doc_id": "d", "page": 2}],
        }
    ]
    result = score_predictions(records)
    assert result["overall"]["num_valid_answer"] == 0
    assert result["overall"]["retrieval_queries_scored"] == 1
    assert result["overall"]["recall_at_k"] == 1.0
    assert result["per_question"][0]["em"] is None
    assert result["per_question"][0]["f1"] is None
    assert result["per_question"][0]["numeric_match"] is None


def test_failure_list_replaces_and_drops() -> None:
    rows = record_failure([], "a", "RuntimeError: boom")
    rows = record_failure(rows, "a", "RuntimeError: again")
    rows = record_failure(rows, "b", "no text extracted")
    assert rows == [
        {"doc_id": "a", "error": "RuntimeError: again"},
        {"doc_id": "b", "error": "no text extracted"},
    ]
    assert drop_failure(rows, "a") == [{"doc_id": "b", "error": "no text extracted"}]


def test_named_tier_sends_parse_version() -> None:
    from parserag.parse import _llama_parse

    assert _llama_parse(None).version is None
    assert _llama_parse("agentic").version == "latest"


def test_parse_tier_paths_and_latest_run_are_isolated() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_parsed, old_index, old_results = config.PARSED_DIR, config.INDEX_DIR, config.RESULTS_DIR
        try:
            config.PARSED_DIR = root / "parsed"
            config.INDEX_DIR = root / "indexes"
            config.RESULTS_DIR = root / "results"

            assert config.parsed_dir("financebench", "doc") == root / "parsed/financebench/doc"
            assert config.parsed_dir("financebench", "doc", "agentic") == (
                root / "parsed/financebench/agentic/doc"
            )
            assert config.index_dir("financebench") == root / "indexes/financebench"
            assert config.index_dir("financebench", "agentic") == (
                root / "indexes/financebench/agentic"
            )

            run_dir = root / "results/financebench/run"
            run_dir.mkdir(parents=True)
            default_run = run_dir / "financebench_1.json"
            agentic_run = run_dir / "financebench_2.json"
            default_run.write_text(json.dumps({"run_config": {}, "results": []}))
            agentic_run.write_text(
                json.dumps({"run_config": {"parse_tier": "agentic"}, "results": []})
            )
            assert config.latest_run_path("financebench") == default_run
            assert config.latest_run_path("financebench", "agentic") == agentic_run
        finally:
            config.PARSED_DIR, config.INDEX_DIR, config.RESULTS_DIR = (
                old_parsed,
                old_index,
                old_results,
            )


if __name__ == "__main__":
    test_rrf_shared_doc_ranks_first()
    test_recall_at_k_page_and_doc()
    test_answer_and_ranked_retrieval_metrics()
    test_retrieval_only_records_skip_answer_metrics()
    test_failure_list_replaces_and_drops()
    test_named_tier_sends_parse_version()
    test_parse_tier_paths_and_latest_run_are_isolated()
    print("ok")

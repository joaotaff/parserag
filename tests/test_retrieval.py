"""RRF ranking + Recall@k oracle. Run: python tests/test_retrieval.py"""

from parserag.eval_retrieval import score_predictions
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
    assert result["n"] == 3
    assert result["page_hits"] == 2
    assert result["doc_hits"] == 3
    assert result["recall_at_k"] == 2 / 3
    assert result["doc_recall_at_k"] == 1.0

    truncated = score_predictions(records, k=1)
    assert truncated["page_hits"] == 1
    assert truncated["doc_hits"] == 2


if __name__ == "__main__":
    test_rrf_shared_doc_ranks_first()
    test_recall_at_k_page_and_doc()
    print("ok")

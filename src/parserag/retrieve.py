from __future__ import annotations

from collections import defaultdict

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

from parserag.config import FUSED_K, RETRIEVER_K, RRF_K


def build_retriever(index: VectorStoreIndex) -> QueryFusionRetriever:
    vector = index.as_retriever(similarity_top_k=RETRIEVER_K)
    bm25 = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=RETRIEVER_K)
    return QueryFusionRetriever(
        [vector, bm25],
        similarity_top_k=FUSED_K,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )


def rrf_fuse(
    rankings: list[list[str]],
    k: int = RRF_K,
    top_n: int = FUSED_K,
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. Same formula QueryFusionRetriever uses (k=60)."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]


def retrieved_records(nodes: list[NodeWithScore]) -> list[dict]:
    records = []
    for node in nodes:
        meta = node.node.metadata or {}
        records.append(
            {
                "doc_id": meta.get("doc_id"),
                "page": meta.get("page"),
                "score": node.score,
                "text": node.node.get_content(),
            }
        )
    return records

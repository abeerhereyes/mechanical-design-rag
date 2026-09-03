"""
Phase 2b: Hybrid retrieval via Reciprocal Rank Fusion (RRF).

RRF combines two ranked lists using only rank position, not raw scores:
    RRF_score(doc) = sum over lists L containing doc of  1 / (k_const + rank_L(doc))
k_const (commonly 60) dampens the influence of very top ranks so one list
can't totally dominate just because its top hit has an extreme raw score.

WHY RRF AND NOT A WEIGHTED SUM OF RAW SCORES:
Cosine distances from a vector store and BM25 scores live on completely
different, un-normalized scales (BM25 is unbounded and corpus-size
dependent; cosine distance is bounded [0,2]). A weighted sum requires
carefully-tuned normalization/calibration that's brittle across corpora and
query types. RRF sidesteps this entirely by fusing on rank, not score, so it
needs zero calibration and is robust by construction. This is the standard
answer to the interview gotcha "why not just weight and add the two scores?"
"""
from collections import defaultdict
from typing import Optional


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k_const: int = 60, top_k: int = 5):
    """ranked_lists: each is a list of result dicts (already sorted best-first) with an 'id' key."""
    scores = defaultdict(float)
    payload = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item["id"]] += 1.0 / (k_const + rank + 1)
            payload[item["id"]] = item
    fused_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {**payload[i], "rrf_score": scores[i]} for i in fused_ids
    ]


def hybrid_search(
    dense_collection,
    sparse_index,
    query: str,
    k: int = 5,
    fetch_k: int = 10,
    course_id: Optional[str] = None,
):
    from src.dense_retriever import dense_search

    dense_results = dense_search(
        dense_collection, query, k=fetch_k, course_id=course_id
    )
    sparse_results = sparse_index.search(query, k=fetch_k, course_id=course_id)
    return reciprocal_rank_fusion([dense_results, sparse_results], top_k=k)


if __name__ == "__main__":
    import os
    from src.chunker import load_corpus
    from src.dense_retriever import build_dense_index
    from src.sparse_retriever import SparseIndex

    corpus_dir = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
    persist_dir = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
    chunks = load_corpus(corpus_dir)
    dense_col = build_dense_index(chunks, persist_dir)
    sparse_idx = SparseIndex(chunks)

    results = hybrid_search(dense_col, sparse_idx, "preload on a reused bolt connection", k=5)
    for r in results:
        print(f"{r['id']}  rrf={r['rrf_score']:.4f}  {r['metadata']['title']}")

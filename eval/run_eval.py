"""
Phase 5: Evaluation harness.

Compares three retrieval configurations on the same 25-query labeled set:
  1. Dense only (bi-encoder vector search)
  2. Hybrid (dense + BM25 fused via RRF)
  3. Hybrid + re-ranked (cross-encoder-style reordering of the RRF candidates)

METRICS (computed directly from retrieved-id vs gold-id overlap, no LLM needed):
  - Precision@k: of the top-k retrieved chunks, what fraction are relevant?
    (This is a stand-in for RAGAS's "context precision" metric.)
  - Recall@k: of all relevant chunks, what fraction appear in the top-k?
    (Stand-in for RAGAS's "context recall".)
  - MRR (Mean Reciprocal Rank): how high up does the FIRST relevant chunk land?
    Captures "did we at least get one good source near the top", which matters
    because the LLM generation step weighs earlier context more heavily.

WHY THIS IS A LEGITIMATE EVAL EVEN WITHOUT AN LLM JUDGE:
Context precision/recall are upstream of generation -- if retrieval doesn't
surface the right chunk, no amount of good generation/faithfulness can fix
the answer. Measuring retrieval quality in isolation, with a labeled gold set,
is a standard and defensible eval layer on its own, and it's what actually
moves when you add hybrid search + reranking (the two Phase 2/3 investments)
-- which is why it's the quotable number for this project. Answer-level
faithfulness/relevance (RAGAS) requires wiring in a live generation LLM; see
eval/README_EVAL_NOTES.md for exactly how that plugs into this same harness.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chunker import load_corpus
from src.dense_retriever import build_dense_index, dense_search
from src.sparse_retriever import SparseIndex
from src.hybrid_retriever import hybrid_search
from src.reranker import LexicalCrossEncoder
from eval.labeled_queries import LABELED_QUERIES


def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    hits = sum(1 for i in relevant_ids if i in top_k)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate(name, retrieve_fn, k=3):
    precisions, recalls, rrs = [], [], []
    for item in LABELED_QUERIES:
        retrieved = retrieve_fn(item["query"])
        retrieved_ids = [r["id"] for r in retrieved]
        precisions.append(precision_at_k(retrieved_ids, item["relevant"], k))
        recalls.append(recall_at_k(retrieved_ids, item["relevant"], k))
        rrs.append(reciprocal_rank(retrieved_ids, item["relevant"]))
    n = len(LABELED_QUERIES)
    return {
        "config": name,
        f"precision@{k}": sum(precisions) / n,
        f"recall@{k}": sum(recalls) / n,
        "MRR": sum(rrs) / n,
    }


def main():
    base = os.path.dirname(__file__)
    corpus_dir = os.path.join(base, "..", "data", "corpus")
    persist_dir = os.path.join(base, "..", "data", "chroma")
    chunks = load_corpus(corpus_dir)
    dense_col = build_dense_index(chunks, persist_dir)
    sparse_idx = SparseIndex(chunks)
    reranker = LexicalCrossEncoder()

    k = 3

    def dense_only(q):
        return dense_search(dense_col, q, k=k)

    def hybrid_only(q):
        return hybrid_search(dense_col, sparse_idx, q, k=k, fetch_k=10)

    def hybrid_reranked(q):
        candidates = hybrid_search(dense_col, sparse_idx, q, k=10, fetch_k=10)
        return reranker.rerank(q, candidates, top_k=k)

    results = [
        evaluate("1. Dense only", dense_only, k=k),
        evaluate("2. Hybrid (dense+BM25 RRF)", hybrid_only, k=k),
        evaluate("3. Hybrid + reranked", hybrid_reranked, k=k),
    ]

    print(f"\n{'Config':<28}{'Precision@'+str(k):<14}{'Recall@'+str(k):<12}{'MRR':<8}")
    print("-" * 62)
    for r in results:
        print(f"{r['config']:<28}{r[f'precision@{k}']:<14.3f}{r[f'recall@{k}']:<12.3f}{r['MRR']:<8.3f}")

    base_mrr = results[0]["MRR"]
    final_mrr = results[-1]["MRR"]
    if base_mrr > 0:
        lift = (final_mrr - base_mrr) / base_mrr * 100
        print(f"\nMRR lift, dense-only -> hybrid+reranked: {lift:+.1f}%")
    base_p = results[0][f"precision@{k}"]
    final_p = results[-1][f"precision@{k}"]
    if base_p > 0:
        lift_p = (final_p - base_p) / base_p * 100
        print(f"Precision@{k} lift, dense-only -> hybrid+reranked: {lift_p:+.1f}%")

    return results


if __name__ == "__main__":
    main()

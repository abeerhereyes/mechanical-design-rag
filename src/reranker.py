"""
Phase 3: Re-ranking.

WHY RE-RANK AT ALL (interview framing):
Retrieval (dense/sparse/RRF) is optimized for *recall at low cost* over the
whole corpus -- it has to score every chunk, so it uses cheap independent
embeddings (bi-encoder: query and doc embedded separately, compared by
cosine/BM25). A cross-encoder reranker instead takes the (query, doc) PAIR
jointly through one model, so it can model term interactions the bi-encoder
architecturally cannot see. It's too expensive to run over the whole corpus,
so the standard pattern is: cheap retrieval fetches a candidate set (top ~10-20),
then the expensive-but-accurate reranker reorders just that small set to
pick the true top-k for the LLM's context window.

SANDBOX NOTE: real cross-encoder weights (e.g.
cross-encoder/ms-marco-MiniLM-L-6-v2) require a HuggingFace download this
sandbox's network allowlist blocks. `LexicalCrossEncoder` below is a
deterministic stand-in that scores each (query, doc) pair jointly using
exact/near term-overlap features -- it captures the *pattern* a
cross-encoder exploits (joint query-doc interaction, not independent
embedding), just with hand-built features instead of a learned model. Swap
in `sentence_transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
on a machine with internet access -- again, a one-line swap behind the same
`.rerank()` interface.
"""
import re
from collections import Counter


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class LexicalCrossEncoder:
    """Deterministic joint query-doc scorer standing in for a real cross-encoder."""

    def score(self, query: str, doc: str) -> float:
        q_tokens = _tokens(query)
        d_tokens = _tokens(doc)
        q_set, d_counter = set(q_tokens), Counter(d_tokens)
        if not q_set:
            return 0.0
        overlap = sum(1 for t in q_set if t in d_counter)
        coverage = overlap / len(q_set)
        # reward query terms appearing close together / repeated (proxy for joint relevance)
        density = sum(d_counter[t] for t in q_set) / max(len(d_tokens), 1)
        # bigram overlap as a crude proxy for phrase-level (not just bag-of-words) match
        q_bigrams = set(zip(q_tokens, q_tokens[1:]))
        d_bigrams = set(zip(d_tokens, d_tokens[1:]))
        bigram_overlap = len(q_bigrams & d_bigrams) / max(len(q_bigrams), 1) if q_bigrams else 0.0
        return 0.55 * coverage + 0.25 * bigram_overlap + 0.20 * min(density * 20, 1.0)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5):
        scored = [
            {**c, "rerank_score": self.score(query, c["text"])} for c in candidates
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        unique = []
        seen = set()
        for item in scored:
            canonical_id = item.get("metadata", {}).get("canonical_id", item["id"])
            if canonical_id in seen:
                continue
            seen.add(canonical_id)
            unique.append(item)
            if len(unique) == top_k:
                break
        return unique


if __name__ == "__main__":
    import os
    from src.chunker import load_corpus
    from src.dense_retriever import build_dense_index
    from src.sparse_retriever import SparseIndex
    from src.hybrid_retriever import hybrid_search

    corpus_dir = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
    persist_dir = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
    chunks = load_corpus(corpus_dir)
    dense_col = build_dense_index(chunks, persist_dir)
    sparse_idx = SparseIndex(chunks)

    query = "how does buckling affect the design of a compression spring"
    candidates = hybrid_search(dense_col, sparse_idx, query, k=10, fetch_k=10)
    reranked = LexicalCrossEncoder().rerank(query, candidates, top_k=5)
    for r in reranked:
        print(f"{r['id']}  rerank={r['rerank_score']:.3f}  rrf={r['rrf_score']:.4f}  {r['metadata']['title']}")

"""
Phase 2a: Sparse (BM25) retrieval.

Why BM25 alongside dense: dense embeddings are good at *topical* similarity
but weak on exact terminology matches (part numbers, symbol names like "Kw",
"At", grade designations like "SAE Grade 5", "E70xx"). Mechanical design
questions are full of exactly this kind of precise vocabulary, so a pure
embedding-only system will systematically under-retrieve on symbol/spec
lookups. BM25 is a strong, cheap complement precisely because it rewards
exact/rare-term overlap, which is where dense embeddings are weakest.
INTERVIEW GOTCHA: "why not just use a bigger/better embedding model instead
of bothering with BM25?" -> even strong embedding models are trained to
generalize semantically, which trades away exact lexical sensitivity; BM25
costs almost nothing to run and directly patches that blind spot rather than
hoping a bigger model fixes it implicitly.
"""
import re
from rank_bm25 import BM25Okapi
from src.chunker import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class SparseIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, k: int = 5):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {"id": self.chunks[i].id, "text": self.chunks[i].text,
             "metadata": self.chunks[i].metadata, "score": float(scores[i])}
            for i in ranked
        ]


if __name__ == "__main__":
    import os
    from src.chunker import load_corpus

    corpus_dir = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
    chunks = load_corpus(corpus_dir)
    idx = SparseIndex(chunks)
    for r in idx.search("preload on a reused bolt connection", k=3):
        print(f"{r['id']}  score={r['score']:.3f}  {r['metadata']['title']}")

"""
Phase 1: Dense (vector) retrieval via ChromaDB.

SANDBOX NOTE: This dev sandbox has no route to HuggingFace/S3 to download
real embedding weights (network allowlist blocks it). So `LocalHashEmbedder`
below is a deterministic, dependency-free stand-in: TF-IDF-weighted hashed
character n-grams projected into a fixed 384-dim vector. It captures lexical/
sub-word overlap, not learned semantic similarity -- e.g. it won't know
"preload" and "clamping force" are related unless they co-occur in the corpus.
On a real machine (with internet access), replace `LocalHashEmbedder` with
`chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
model_name="all-MiniLM-L6-v2")` -- one-line swap, same interface, this is
exactly the kind of "what would you change for production" answer worth
having ready in an interview.
"""
import chromadb
from chromadb.utils import embedding_functions
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
from src.chunker import Chunk, load_corpus
import os
import hashlib
import json
from typing import Optional

COLLECTION_NAME = "course_content_v1"
EMBED_DIM = 384


class LocalHashEmbedder(embedding_functions.EmbeddingFunction):
    """Deterministic local stand-in for a sentence embedding model (see module docstring)."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self._vec = HashingVectorizer(
            n_features=dim, analyzer="char_wb", ngram_range=(3, 5), norm="l2", alternate_sign=False
        )

    def __call__(self, input):
        mat = self._vec.transform(input)
        return mat.toarray().astype(np.float32).tolist()

    @staticmethod
    def name():
        return "local-hash-char-ngram-384"

    def get_config(self):
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config):
        return LocalHashEmbedder(dim=int(config.get("dim", EMBED_DIM)))


def chunks_content_hash(chunks: list[Chunk]) -> str:
    payload = [
        {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata}
        for chunk in sorted(chunks, key=lambda item: item.id)
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_dense_index(chunks: list[Chunk], persist_dir: str):
    client = chromadb.PersistentClient(path=persist_dir)
    ef = LocalHashEmbedder()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"content_hash": chunks_content_hash(chunks)},
    )
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    return collection


def get_collection(persist_dir: str):
    client = chromadb.PersistentClient(path=persist_dir)
    ef = LocalHashEmbedder()
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def ensure_dense_index(chunks: list[Chunk], persist_dir: str):
    """Load the current index, rebuilding it when approved content changed."""
    try:
        collection = get_collection(persist_dir)
        metadata = collection.metadata or {}
        if (
            collection.count() == len(chunks)
            and metadata.get("content_hash") == chunks_content_hash(chunks)
        ):
            return collection
    except Exception:
        pass
    return build_dense_index(chunks, persist_dir)


def dense_search(collection, query: str, k: int = 5, course_id: Optional[str] = None):
    count = collection.count()
    if course_id:
        matching = collection.get(where={"course_id": course_id}, include=[])
        count = len(matching["ids"])
    if count == 0:
        return []
    kwargs = {"where": {"course_id": course_id}} if course_id else {}
    res = collection.query(query_texts=[query], n_results=min(k, count), **kwargs)
    out = []
    for i in range(len(res["ids"][0])):
        out.append(
            {
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            }
        )
    return out


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    corpus_dir = os.path.join(base, "..", "data", "corpus")
    persist_dir = os.path.join(base, "..", "data", "chroma")
    chunks = load_corpus(corpus_dir)
    col = build_dense_index(chunks, persist_dir)
    print("Index built with", col.count(), "chunks")
    results = dense_search(col, "how do I compute preload on a reused bolt", k=3)
    for r in results:
        print(f"{r['id']}  dist={r['distance']:.3f}  {r['metadata']['title']}")

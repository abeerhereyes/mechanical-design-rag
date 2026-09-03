from src.chunker import Chunk
from src.dense_retriever import build_dense_index, dense_search, ensure_dense_index
from src.hybrid_retriever import reciprocal_rank_fusion
from src.reranker import LexicalCrossEncoder
from src.sparse_retriever import SparseIndex


def _chunk(chunk_id, text, course_id, canonical_id=None):
    metadata = {
        "course_id": course_id,
        "source": f"{course_id}.pdf",
        "section": "1",
        "page": 1,
    }
    if canonical_id:
        metadata["canonical_id"] = canonical_id
    return Chunk(
        id=chunk_id,
        text=text,
        source=metadata["source"],
        section="1",
        page=1,
        title=text,
        metadata=metadata,
    )


def test_sparse_search_never_leaks_between_courses():
    index = SparseIndex(
        [
            _chunk("aero::1", "lift circulation cylinder", "aerodynamics"),
            _chunk("qrm::1", "control chart quality limits", "qrm"),
        ]
    )
    results = index.search("lift control chart", k=5, course_id="qrm")
    assert [item["id"] for item in results] == ["qrm::1"]


def test_reranker_collapses_duplicate_content_nodes():
    candidates = [
        {
            "id": "qrm::deck-a::2",
            "text": "producer risk acceptance sampling",
            "metadata": {"canonical_id": "qrm::acceptance-risk"},
        },
        {
            "id": "qrm::deck-b::11",
            "text": "producer risk acceptance sampling",
            "metadata": {"canonical_id": "qrm::acceptance-risk"},
        },
        {
            "id": "qrm::deck-c::4",
            "text": "consumer risk beta",
            "metadata": {"canonical_id": "qrm::consumer-risk"},
        },
    ]
    results = LexicalCrossEncoder().rerank(
        "producer and consumer risk", candidates, top_k=3
    )
    canonical_ids = [item["metadata"]["canonical_id"] for item in results]
    assert canonical_ids.count("qrm::acceptance-risk") == 1


def test_rrf_retains_course_metadata():
    result = reciprocal_rank_fusion(
        [
            [
                {
                    "id": "aero::1",
                    "text": "lift",
                    "metadata": {"course_id": "aerodynamics"},
                }
            ],
            [
                {
                    "id": "aero::1",
                    "text": "lift",
                    "metadata": {"course_id": "aerodynamics"},
                }
            ],
        ]
    )
    assert result[0]["metadata"]["course_id"] == "aerodynamics"


def test_dense_index_filters_courses_and_reuses_matching_content(tmp_path):
    chunks = [
        _chunk("aero::1", "circulation creates lift", "aerodynamics"),
        _chunk("qrm::1", "acceptance sampling plan", "qrm"),
    ]
    collection = build_dense_index(chunks, str(tmp_path))
    results = dense_search(
        collection, "lift sampling", k=5, course_id="aerodynamics"
    )
    assert [item["id"] for item in results] == ["aero::1"]
    assert ensure_dense_index(chunks, str(tmp_path)).count() == 2

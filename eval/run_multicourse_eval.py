"""Evaluate course isolation and source/page retrieval for PDF courses."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.multicourse_labeled_queries import LABELED_QUERIES
from src.agent import RetrievalEngine


def _relevant(result: dict, item: dict) -> bool:
    metadata = result["metadata"]
    return (
        metadata.get("source") == item["source"]
        and int(metadata.get("page", 0)) in item["pages"]
    )


def evaluate(top_k: int = 3) -> dict:
    engine = RetrievalEngine()
    by_course = defaultdict(lambda: {"queries": 0, "hits": 0, "rr": 0.0, "leaks": 0})
    for item in LABELED_QUERIES:
        results = engine.search(item["query"], item["course_id"], top_k=top_k)
        metrics = by_course[item["course_id"]]
        metrics["queries"] += 1
        metrics["leaks"] += sum(
            result["metadata"].get("course_id") != item["course_id"]
            for result in results
        )
        for rank, result in enumerate(results, start=1):
            if _relevant(result, item):
                metrics["hits"] += 1
                metrics["rr"] += 1.0 / rank
                break

    output = {}
    for course_id, metrics in by_course.items():
        count = metrics["queries"]
        output[course_id] = {
            "queries": count,
            f"hit@{top_k}": metrics["hits"] / count,
            "MRR": metrics["rr"] / count,
            "course_leaks": metrics["leaks"],
        }
    return output


def main() -> dict:
    results = evaluate()
    print(f"{'Course':<20}{'Queries':<10}{'Hit@3':<10}{'MRR':<10}{'Leaks':<8}")
    print("-" * 58)
    for course_id, metrics in sorted(results.items()):
        print(
            f"{course_id:<20}{metrics['queries']:<10}"
            f"{metrics['hit@3']:<10.3f}{metrics['MRR']:<10.3f}"
            f"{metrics['course_leaks']:<8}"
        )
    return results


if __name__ == "__main__":
    main()

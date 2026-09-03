import json

from src.llm import OllamaClient


def test_generated_citations_are_limited_to_retrieved_ids(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(
        client,
        "_chat",
        lambda payload: {
            "message": {
                "content": json.dumps(
                    {
                        "course_answer": "The course gives the stated relation.",
                        "general_answer": "Other texts may use another convention.",
                        "citation_ids": ["aero::p50", "invented::citation"],
                        "insufficient_course_context": False,
                    }
                )
            }
        },
    )
    chunks = [
        {
            "id": "aero::p50",
            "text": "Potential flow around a cylinder.",
            "metadata": {
                "source": "Aerodynamic_M-1.pdf",
                "section": "Cylinder flow",
                "page": 50,
            },
        }
    ]
    answer = client.generate_structured("Explain cylinder flow", chunks)
    assert answer.citation_ids == ["aero::p50"]
    assert answer.general_answer.startswith("Other texts")


def test_insufficient_context_is_preserved(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(
        client,
        "_chat",
        lambda payload: {
            "message": {
                "content": json.dumps(
                    {
                        "course_answer": "The supplied notes do not cover this.",
                        "general_answer": None,
                        "citation_ids": [],
                        "insufficient_course_context": True,
                    }
                )
            }
        },
    )
    answer = client.generate_structured("Unsupported topic", [])
    assert answer.insufficient_course_context is True
    assert answer.citation_ids == []

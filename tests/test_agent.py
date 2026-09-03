import pytest

from src.agent import MechanicalDesignAgent, classify_intent, extract_params


class FakeRetriever:
    def search(self, query, top_k=3):
        return [
            {
                "id": "springs.md::S3",
                "text": "Springs with L0/D > 4 should be checked for buckling.",
                "metadata": {
                    "source": "Springs Module",
                    "section": "S3",
                    "page": 3,
                },
            }
        ]


class FakeLlm:
    def generate(self, query, chunks):
        return "Check buckling when L0/D exceeds 4 [1]."


def test_labeled_spring_parameters_are_extracted():
    state = classify_intent(
        {
            "query": (
                "What is the spring rate for G=79300 MPa, wire 3 mm, "
                "mean D 25 mm, 8 active coils?"
            )
        }
    )
    result = extract_params(state)
    assert result["intent"] == "calculate"
    assert result["params"]["shear_modulus_mpa"] == 79300
    assert result["params"]["wire_diameter_mm"] == 3
    assert result["params"]["mean_diameter_mm"] == 25
    assert result["params"]["active_coils"] == 8


def test_missing_parameters_produce_clarification():
    agent = MechanicalDesignAgent(retriever=FakeRetriever(), llm=FakeLlm())
    result = agent.ask("What is the spring rate for my compression spring?")
    assert result["intent"] == "clarify"
    assert "Wire shear modulus" in result["answer"]
    assert result["missing_params"]


def test_calculation_executes_real_formula():
    agent = MechanicalDesignAgent(retriever=FakeRetriever(), llm=FakeLlm())
    result = agent.ask(
        "Spring rate with G=79300, wire 3, mean D 25, 8 active coils"
    )
    assert result["intent"] == "calculate"
    assert "6.423" in result["answer"]
    assert result["citations"][0]["section"] == "S1"


def test_retrieval_uses_llm_and_returns_structured_citation():
    agent = MechanicalDesignAgent(retriever=FakeRetriever(), llm=FakeLlm())
    result = agent.ask("When should I check a spring for buckling?")
    assert result["intent"] == "retrieve"
    assert result["answer"].endswith("[1].")
    assert result["citations"][0]["id"] == "springs.md::S3"


def test_formula_question_routes_to_retrieval_not_calculation():
    state = classify_intent(
        {"query": "What is the formula for the spring rate of a compression spring?"}
    )
    assert state["intent"] == "retrieve"


def test_empty_query_is_rejected():
    agent = MechanicalDesignAgent(retriever=FakeRetriever(), llm=FakeLlm())
    with pytest.raises(ValueError):
        agent.ask(" ")

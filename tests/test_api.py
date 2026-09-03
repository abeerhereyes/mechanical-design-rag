from fastapi.testclient import TestClient

import src.api as api


class FakeAgent:
    def ask(self, query, params=None):
        return {
            "answer": "Calculated result: 6.423 N/mm.",
            "intent": "calculate",
            "calc_type": "spring_rate",
            "params": params or {},
            "missing_params": [],
            "citations": [
                {
                    "id": "springs.md::S1",
                    "source": "springs.md",
                    "section": "S1",
                    "page": 1,
                }
            ],
        }


def test_health_endpoint():
    assert TestClient(api.app).get("/health").json() == {"status": "ok"}


def test_ask_endpoint(monkeypatch):
    monkeypatch.setattr(api, "get_agent", lambda: FakeAgent())
    response = TestClient(api.app).post("/ask", json={"query": "spring rate"})
    assert response.status_code == 200
    body = response.json()
    assert body["calculation"] == "spring_rate"
    assert body["citations"][0]["section"] == "S1"


def test_empty_question_is_rejected():
    response = TestClient(api.app).post("/ask", json={"query": ""})
    assert response.status_code == 422

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


class FakeMultiCourseAgent:
    def __init__(self):
        self.request = None

    def ask(
        self,
        query,
        params=None,
        course_id="mechanical-design",
        include_general=True,
    ):
        self.request = {
            "query": query,
            "params": params,
            "course_id": course_id,
            "include_general": include_general,
        }
        return {
            "answer": "Combined answer",
            "course_answer": "The notes say use the Wahl factor [1].",
            "general_answer": "Design handbooks provide additional context.",
            "intent": "retrieve",
            "params": params,
            "citations": [
                {
                    "id": "springs.md::S2",
                    "source": "springs.md",
                    "section": "S2",
                    "page": 2,
                }
            ],
            "warnings": ["Check material limits."],
            "assumptions": ["Round wire."],
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


def test_multicourse_fields_are_forwarded_and_answers_are_separated(monkeypatch):
    agent = FakeMultiCourseAgent()
    monkeypatch.setattr(api, "get_agent", lambda: agent)
    response = TestClient(api.app).post(
        "/ask",
        json={
            "query": "Explain spring stress",
            "course_id": "machine-elements",
            "include_general": False,
            "params": {"force_n": 100},
        },
    )
    assert response.status_code == 200
    assert agent.request["course_id"] == "machine-elements"
    assert agent.request["include_general"] is False
    body = response.json()
    assert body["course_answer"].startswith("The notes")
    assert body["general_answer"].startswith("Design handbooks")
    assert body["warnings"] == ["Check material limits."]
    assert body["citations"][0]["preview_url"] == "/sources/springs.md/pages/2"


def test_metadata_and_calculation_schema_endpoints(monkeypatch):
    monkeypatch.setattr(api, "get_agent", lambda: FakeAgent())
    client = TestClient(api.app)
    courses = client.get("/courses")
    status = client.get("/ingestion/status")
    calculations = client.get("/calculations")
    schema = client.get("/calculations/mechanical.helical_spring_rate/schema")

    assert courses.status_code == 200
    assert courses.json()["courses"][0]["id"] == "mechanical-design"
    assert status.json()["documents"] >= 4
    assert calculations.json()["calculations"]
    assert schema.json()["parameters"][0]["name"] == "shear_modulus_mpa"


def test_exact_source_page_preview():
    response = TestClient(api.app).get("/sources/springs.md/pages/3")
    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 3
    assert body["title"] == "Buckling and Slenderness"
    assert "L0/D > 4" in body["content"]
    assert "Fatigue Life" not in body["content"]


def test_pdf_source_page_preview_links_exact_document_page():
    client = TestClient(api.app)
    response = client.get("/sources/Aerodynamic_M-1.pdf/pages/50")
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "pdf"
    assert body["document_url"].endswith(
        "/aerodynamics/Aerodynamic_M-1.pdf#page=50"
    )
    assert client.get(
        "/source-files/aerodynamics/Aerodynamic_M-1.pdf"
    ).status_code == 200


def test_web_chat_is_served_without_a_node_build():
    client = TestClient(api.app)
    page = client.get("/")
    script = client.get("/static/app.js")
    assert page.status_code == 200
    assert "From your course notes" in page.text
    assert "General knowledge" in page.text
    assert script.status_code == 200
    assert "localStorage" in script.text

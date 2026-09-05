"""
Automated tests for COPF-231's /ask endpoint, running against the REAL
RavinAnswerService (backend.service), built from the actual current
policy corpus at startup.

IMPORTANT: these tests require:
  - Real internet access (the app fetches live policy pages from
    policies.latrobe.edu.au on startup)
  - A local Ollama server running (see backend.llm.ollama_provider)

They could not be run in the sandbox this file was written in (no
network access to policies.latrobe.edu.au there) - run and verify these
locally before treating them as passing evidence.

Uses TestClient as a context manager so the lifespan (service startup)
actually runs before requests are made.
"""

from fastapi.testclient import TestClient

from backend.api.main import app


def test_health_check_reports_service_ready():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service_ready"] is True


def test_supported_question_returns_grounded_answer():
    # "Admissions Policy" (policy_id 169) is one of the 6 real current
    # policies in CURRENT_POLICY_LINKS - a direct question about it
    # should be answerable.
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is the current admissions policy?"})
        assert response.status_code == 200
        data = response.json()
        assert data["behavior"] in ("direct_answer", "grounded_overview")
        assert data["grounded"] is True
        assert len(data["sources"]) >= 1
        assert "policy_id" in data["sources"][0]


def test_unsupported_question_returns_no_grounded_answer():
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is the best pizza topping?"})
        assert response.status_code == 200
        data = response.json()
        assert data["behavior"] == "no_grounded_answer"
        assert data["grounded"] is False
        assert data["sources"] == []


def test_ambiguous_question_returns_clarify():
    # Deliberately vague, to exercise the CLARIFY routing path.
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "policy"})
        assert response.status_code == 200
        data = response.json()
        # Ambiguous input may route to clarify OR no_grounded_answer
        # depending on the classifier - both are valid non-fabricated outcomes.
        assert data["behavior"] in ("clarify", "no_grounded_answer")
        assert data["grounded"] is False


def test_empty_question_is_rejected():
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "invalid_request"
        assert "traceback" not in str(data).lower()


def test_missing_question_field_is_rejected():
    with TestClient(app) as client:
        response = client.post("/ask", json={})
        assert response.status_code == 422


def test_wrong_data_type_is_rejected():
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": 12345})
        assert response.status_code == 422


def test_excessively_long_question_is_rejected():
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "a" * 1000})
        assert response.status_code == 422

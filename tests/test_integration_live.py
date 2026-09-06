"""
Live integration tests for COPF-231 - run against the REAL
RavinAnswerService, built from the actual current policy corpus at
startup.

IMPORTANT: these require real internet access (live policy pages from
policies.latrobe.edu.au) and a local Ollama server running.

For fast, no-network tests safe for every commit, see
tests/test_api_contract.py. Per Chris's team handoff deck (slide 11):
run targeted tests while developing, but run the full 725+ test repo
suite before considering integration complete:
    .\\.venv\\Scripts\\python.exe -m pytest -q
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


def test_focused_supported_question_returns_grounded_answer():
    """Known-supported case, per the team handoff deck's three required
    behaviour types (slide 11): known-supported -> grounded answer."""
    with TestClient(app) as client:
        response = client.post(
            "/api/questions",
            json={"question": "What is the current admissions policy?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["behavior"] in ("direct_answer", "grounded_overview")
        assert data["grounded"] is True
        assert len(data["sources"]) >= 1


def test_clear_unsupported_question_returns_grounded_false():
    """Clear-but-unsupported case, per the deck's second required
    behaviour type: clear unsupported -> grounded false.

    NOTE: per Chris's team handoff deck slide 12, a clear question
    returning grounded=false is not automatically a bug - it is only a
    genuine backend gap if the correct policy is known to be indexed and
    RAVIN still fails to retrieve/recognise it. Do not weaken this
    assertion or "fix" it by lowering thresholds without confirming with
    Chris first that it's a real gap."""
    with TestClient(app) as client:
        response = client.post(
            "/api/questions",
            json={"question": "What's the best pizza topping?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["grounded"] is False
        assert data["sources"] == []


def test_ambiguous_question_returns_clarify():
    """Ambiguous case, per the deck's third required behaviour type:
    ambiguous -> clarification."""
    with TestClient(app) as client:
        response = client.post("/api/questions", json={"question": "policy"})
        assert response.status_code == 200
        data = response.json()
        assert data["behavior"] in ("clarify", "no_grounded_answer")
        assert data["grounded"] is False

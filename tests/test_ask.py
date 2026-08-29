"""
Automated tests for COPF-231's /ask endpoint, running against the REAL
backend.core pipeline (build_grounded_response), not a mock.
"""

from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_supported_question_returns_grounded_answer():
    response = client.post("/ask", json={"question": "What is the policy on assignment extensions?"})
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    assert len(data["sources"]) >= 1
    assert "policy_id" in data["sources"][0]


def test_unsupported_question_returns_insufficient_evidence():
    response = client.post("/ask", json={"question": "What's the weather like today?"})
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert data["sources"] == []


def test_empty_question_is_rejected():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "invalid_request"
    assert "traceback" not in str(data).lower()


def test_missing_question_field_is_rejected():
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_wrong_data_type_is_rejected():
    response = client.post("/ask", json={"question": 12345})
    assert response.status_code == 422


def test_excessively_long_question_is_rejected():
    response = client.post("/ask", json={"question": "a" * 1000})
    assert response.status_code == 422


def test_additional_fixture_topics_are_reachable():
    """Confirm the extra fixtures (HDR, conduct, privacy, employment) are
    actually usable, not just Chris's original 2."""
    cases = [
        "What happens if a student breaches the code of conduct?",
        "How is my personal data used by the university?",
        "What are the conditions for casual employment?",
    ]
    for question in cases:
        response = client.post("/ask", json={"question": question})
        assert response.status_code == 200
        data = response.json()
        assert data["grounded"] is True, f"Expected a match for: {question}"

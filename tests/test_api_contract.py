"""
Fast API contract tests for COPF-231.

These test request validation, error handling, and response mapping
using a FAKE service (no real RavinAnswerService, no network calls, no
Ollama required). They run in milliseconds and are safe for CI.

For tests against the REAL RavinAnswerService (live policy corpus,
Ollama), see tests/test_integration_live.py - per Chris's team handoff
deck (slide 11): "Run your API / UI / security tests while developing.
Before integration is considered complete, run the entire repo."

IMPORTANT: TestClient(app) is NOT used as a context manager here
deliberately - this means the FastAPI lifespan (which builds the real
service) does NOT run. Instead, app.state.ravin_service is set directly
to a fake before each test.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.api.main import app


@dataclass
class _FakeBehavior:
    value: str


@dataclass
class _FakeSource:
    policy_id: str
    title: str
    heading: str
    url: str


@dataclass
class _FakeResult:
    behavior: _FakeBehavior
    grounded: bool
    answer: str
    sources: tuple = field(default_factory=tuple)


class _FakeService:
    """Stands in for RavinAnswerService. Returns a fixed grounded answer
    for any question, so these tests only exercise the API layer
    (validation, routing, JSON mapping) - not real retrieval/generation."""

    def answer(self, question: str):
        return _FakeResult(
            behavior=_FakeBehavior("direct_answer"),
            grounded=True,
            answer="This is a fake grounded answer for testing.",
            sources=(
                _FakeSource(
                    policy_id="169",
                    title="Admissions Policy",
                    heading="Section 1",
                    url="https://policies.latrobe.edu.au/document/view.php?id=169",
                ),
            ),
        )


def _client_with_fake_service():
    """Build a TestClient with app.state.ravin_service pre-set to a
    fake, WITHOUT running the real lifespan (no `with` block)."""
    app.state.ravin_service = _FakeService()
    return TestClient(app)


def test_health_check():
    client = _client_with_fake_service()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service_ready": True}


def test_valid_question_returns_mapped_response():
    client = _client_with_fake_service()
    response = client.post("/api/questions", json={"question": "What is the admissions policy?"})
    assert response.status_code == 200
    data = response.json()
    assert data["behavior"] == "direct_answer"
    assert data["grounded"] is True
    assert data["answer"] == "This is a fake grounded answer for testing."
    assert data["sources"][0]["policy_id"] == "169"


def test_empty_question_is_rejected():
    client = _client_with_fake_service()
    response = client.post("/api/questions", json={"question": ""})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "invalid_request"
    assert "traceback" not in str(data).lower()


def test_missing_question_field_is_rejected():
    client = _client_with_fake_service()
    response = client.post("/api/questions", json={})
    assert response.status_code == 422


def test_wrong_data_type_is_rejected():
    client = _client_with_fake_service()
    response = client.post("/api/questions", json={"question": 12345})
    assert response.status_code == 422


def test_excessively_long_question_is_rejected():
    client = _client_with_fake_service()
    response = client.post("/api/questions", json={"question": "a" * 1000})
    assert response.status_code == 422


def test_backend_valueerror_is_reported_as_internal_error_not_bad_input():
    """
    Regression test for the exact class of bug found in COPF-240: a
    ValueError raised from deep inside the real service (e.g. "Generated
    claim cannot contain only evidence markers") must be reported as a
    500 internal error, NOT a 422 invalid-request - the user's input was
    fine; the backend failed.
    """

    class _BrokenService:
        def answer(self, question: str):
            raise ValueError("Generated claim cannot contain only evidence markers.")

    app.state.ravin_service = _BrokenService()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/questions", json={"question": "What is the admissions policy?"})
    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "internal_error"
    assert "traceback" not in str(data).lower()
    assert "evidence markers" not in str(data).lower()  # no internal detail leaked

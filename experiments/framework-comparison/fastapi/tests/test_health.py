from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "ravin",
    }

def test_valid_question():
    response = client.post(
        "/questions/validate",
        json={
            "question": "What is the special consideration policy?",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "question": "What is the special consideration policy?",
    }

def test_empty_question_is_rejected():
    response = client.post(
        "/questions/validate",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422

def test_whitespace_question_is_rejected():
    response = client.post(
        "/questions/validate",
        json={
            "question": "   ",
        },
    )

    assert response.status_code == 422
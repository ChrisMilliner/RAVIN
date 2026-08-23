from flask import Flask, jsonify, request

app = Flask(__name__)

@app.get("/health")
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "service": "ravin",
        }
    )

@app.post("/questions/validate")
def validate_question():
    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return jsonify(
            {
                "detail": "A JSON request body is required.",
            }
        ), 422

    question = body.get("question")

    if not isinstance(question, str) or not question.strip():
        return jsonify(
            {
                "detail": "Question must contain non-whitespace text.",
            }
        ), 422

    return jsonify(
        {
            "valid": True,
            "question": question,
        }
    )
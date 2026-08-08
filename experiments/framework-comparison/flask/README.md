# RAVIN Flask Framework Proof of Concept

## Purpose

This experiment evaluates Flask as a candidate application/API framework for RAVIN under Jira issue COPF-217.

It is a technical proof of concept only and does not represent a final RAVIN architecture or technology-selection decision.

## Environment

- Python: 3.12.2
- Flask: 3.1.3
- Werkzeug: 3.1.8
- pytest: 9.1.1

## Capability Under Test

The application exposes:

`GET /health`

Expected HTTP status:

`200 OK`

Expected response:

```json
{
  "status": "healthy",
  "service": "ravin"
}
```

## Setup

Create a Python 3.12 virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m flask --app app.main run --host 127.0.0.1 --port 8000
```

Health endpoint:

`http://127.0.0.1:8000/health`

## Test

```powershell
python -m pytest -v
```

## Evaluation Result
|Check|Result|
|---|---|
|Local Python 3.12 environment|PASS|
|Dependency installation|PASS|
|Application startup|PASS|
|GET /health reachable|PASS|
|HTTP 200 returned|PASS|
|Expected JSON contract returned|PASS|
|Automated health test|PASS|
|Paid/external service required|NO|
|Whitespace-only question rejection|PASS|
|Missing JSON body rejection|PASS|

The experiment also evaluates request validation through:

`POST /questions/validate`

Valid request:

```json
{
  "question": "What is the special consideration policy?"
}
```

Expected successful response:

```json
{
  "valid": true,
  "question": "What is the special consideration policy?"
}
```

Empty, whitespace-only or missing request content is rejected with:

`HTTP 422`



## Technical Observations
Flask provided a compact implementation of the required health endpoint.

Automated testing was performed using Flask's built-in test client and pytest, without requiring a separately installed HTTP client library.

Unlike the FastAPI proof of concept, this minimal Flask implementation did not automatically expose an interactive API documentation interface.

## Limitations

This experiment evaluates only a minimal application/API capability.

It does not evaluate:

- policy ingestion;
- policy retrieval;
- RAG functionality;
- embeddings;
- vector indexing;
- LLM integration;
- grounding validation;
- production deployment;
- full application performance;
- frontend integration.

Flask has not been selected as RAVIN's final application/API framework by this experiment.
# RAVIN FastAPI Framework Proof of Concept

## Purpose

This experiment evaluates FastAPI as a candidate application/API framework for RAVIN under Jira issue COPF-217.

It is an experimental implementation only and does not represent a final architecture or technology-selection decision.

## Runtime

Python 3.12.2

## Capability Under Test

The application exposes:

`GET /health`

Expected response:

```json
{
  "status": "healthy",
  "service": "ravin"
}
```

Expected status:

`HTTP 200`

## Setup

Create and activate a Python 3.12 virtual environment.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health endpoint:
`http://127.0.0.1:8000/health`

Interactive API documentation:
`http://127.0.0.1:8000/docs`

## Test
```powershell
python -m pytest -v
```

## Evaluation Result

|Check|Result|
|---|---|
|Python 3.12 environment|PASS|
|Dependency installation|PASS|
|Application startup|PASS|
|GET /health reachable|PASS|
|HTTP 200 returned|PASS|
|Expected JSON contract returned|PASS|
|Automated health test|PASS|
|Interactive API documentation|AVAILABLE|
|Paid/external service required|NO|

## Dependency Observation

The initial FastAPI test used HTTPX and passed successfully but produced a Starlette deprecation warning recommending HTTPX2.

HTTPX was removed and replaced with HTTPX2 2.9.1.

The same automated test was rerun after the dependency change and passed without warnings.

This demonstrates that the dependency change did not regress the tested API behaviour.

## Limitations

This proof of concept evaluates only a minimal API capability.

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

FastAPI has not been selected as RAVIN's final application framework by this experiment.
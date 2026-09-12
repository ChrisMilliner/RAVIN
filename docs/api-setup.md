# API Setup (backend/api)

The API layer has its own dependency file, separate from the core RAVIN
backend. A clean environment needs **both** installed:

```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

## Running the fast test suite (default, no external dependencies)

```bash
pytest
```

This excludes integration tests automatically (see `pytest.ini`) - no
internet access or local Ollama server required, runs in under a
second.

## Running the live integration tests

Requires:
- Real internet access (fetches live policy pages from policies.latrobe.edu.au)
- A local Ollama server running

```bash
pytest -m integration
```

## Running the API locally

```bash
uvicorn backend.api.main:app --reload
```

Startup will take noticeably longer than a typical FastAPI app, since it
acquires the current policy corpus from live pages before the server is
ready to accept requests. Once running:

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

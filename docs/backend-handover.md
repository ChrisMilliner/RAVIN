> **Implementation scope:** RAVIN backend workstream
>
> **Backend checkpoint:** `5c8e2ce`
>
> **Verification baseline:** 725 passing tests
>
> **Purpose:** Source material for later project-wide documentation

# RAVIN Backend Handover

## 1. Purpose

This document explains how another developer or application layer can consume the implemented RAVIN backend.

The primary application boundary is `RavinAnswerService`.

## 2. Backend Entry Point

Application code should construct the backend through the shared bootstrap rather than rebuilding the RAG pipeline manually.

```python
from backend.service.bootstrap import (
    create_current_policy_ravin_service,
)

service = create_current_policy_ravin_service()

result = service.answer(
    "What happens when a student is not making satisfactory academic progress?"
)
```

## 3. Service Lifecycle

```text
application startup
-> create_current_policy_ravin_service()
-> retain service instance

user request
-> service.answer(question)

next user request
-> reuse same service
```

Do not construct a new service for every question.

Service creation can involve policy acquisition, ingestion, model loading, corpus embedding, and retrieval index construction.

## 4. Application-Facing Result

`service.answer()` returns an `IntegratedAnswerResult`.

The result includes:

- `behavior`
- `answer`
- `grounded`
- `sources`

Each grounded source exposes:

- `policy_id`
- `title`
- `heading`
- `url`

Application adapters should map this result into their own transport format rather than exposing internal retrieval objects.

## 5. Deterministic Behaviors

Application code should expect:

- `direct_answer`
- `grounded_overview`
- `clarify`
- `no_grounded_answer`

The backend selects these behaviors.

The API or UI should not independently reclassify the response.

## 6. Clarification

A clarification response represents an ambiguous question.

It should not be treated as equivalent to a failed evidence search.

A clarified user question should be submitted as a new question through the normal backend pipeline.

## 7. No Grounded Answer

A no-grounded-answer result represents a clear question for which the backend did not establish sufficient evidence.

The consuming application should preserve this failure rather than asking a language model to invent an alternative response.

## 8. CLI Reference Adapter

`scripts/run_ravin.py` is the implemented local adapter.

It demonstrates the correct high-level pattern:

```text
build service once
-> submit questions
-> print IntegratedAnswerResult
```

The CLI should not itself be invoked from a web API.

Both adapters should consume the same service.

## 9. Example FastAPI Integration

### Status

The following is a handover/integration example.

It demonstrates how the implemented backend should be consumed by a FastAPI application.

It does not claim that this FastAPI module is part of the completed backend workstream.

### Example

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel

from backend.service.answer_service import RavinAnswerService
from backend.service.bootstrap import (
    create_current_policy_ravin_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ravin_service = (
        create_current_policy_ravin_service()
    )

    yield


app = FastAPI(
    lifespan=lifespan
)


class QuestionRequest(BaseModel):
    question: str


@app.post("/api/questions")
def answer_question(
    payload: QuestionRequest,
    request: Request,
):
    service: RavinAnswerService = (
        request.app.state.ravin_service
    )

    result = service.answer(
        payload.question
    )

    return {
        "answer": result.answer,
        "grounded": result.grounded,
        "sources": [
            {
                "policy_id": source.policy_id,
                "title": source.title,
                "heading": source.heading,
                "url": source.url,
            }
            for source in result.sources
        ],
    }
```

## 10. Why Startup Construction Matters

```text
FastAPI startup
-> policy acquisition
-> ingestion
-> model loading
-> embedding/index build
-> RavinAnswerService ready
```

Individual HTTP requests then perform:

```text
request validation
-> service.answer(question)
-> map result to JSON
```

The API must not re-download policies, reload models, rebuild embeddings, or rebuild the retrieval index for every request.

## 11. Agreed HTTP Contract

Project integration baseline:

```text
POST /api/questions
Content-Type: application/json
```

Example request:

```json
{
  "question": "What does the policy say about special consideration?"
}
```

Example grounded response shape:

```json
{
  "answer": "Based on the available policy...",
  "grounded": true,
  "sources": [
    {
      "policy_id": "123",
      "title": "Policy Name",
      "heading": "Relevant Section",
      "url": "https://..."
    }
  ]
}
```

Example no-evidence response shape:

```json
{
  "answer": "I could not find sufficient evidence in the available policy sources.",
  "grounded": false,
  "sources": []
}
```

This section records the integration contract/pattern and should be updated against the implemented API before it is treated as final API documentation.

## 12. Provider Replacement

An integrating developer should not instantiate model providers inside request handlers.

Provider configuration belongs to runtime startup and composition.

## 13. Backend Verification

At backend checkpoint `5c8e2ce`:

```text
725 tests passed
82/82 modules documented
291/291 public definitions documented
171/171 public callable parameters typed
171/171 public callable returns typed
```

## 14. Integration Work Outside This Handover

This backend handover does not claim completion of:

- final FastAPI routes
- frontend integration
- final deployment environment
- authentication/authorization
- project-wide security configuration
- final production monitoring

Those should be documented from the relevant implemented project work.

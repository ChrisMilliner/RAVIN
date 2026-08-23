# RAVIN Grounded Response Reference Slice

## Jira

COPF-218 - Build Framework-Neutral Grounded Response Reference Slice

## Purpose

This implementation provides the first executable framework-neutral reference slice for RAVIN's evidence-before-answer workflow.

It demonstrates the core behavioural requirement that RAVIN must not return a policy answer unless sufficient supporting evidence has first been established.

The implementation is deliberately deterministic and uses synthetic policy fixtures.

It is not the final RAG implementation.

## Current Flow

```text
Question
   ↓
Question validation
   ↓
Deterministic lexical retrieval
   ↓
Retrieved candidate evidence
   ↓
Evidence sufficiency assessment
   ↓
┌──────────────────────────┐
│                          │
Sufficient             Insufficient
│                          │
Reference answer        Controlled refusal
│                          │
Source provenance       No policy citation
```

## Components

### Domain Models

`backend/core/models.py`

Defines framework-neutral representations for:

- policy documents;
- retrieved evidence;
- evidence assessments;
- evidence-sufficiency outcomes;
- response outcomes;
- grounded responses.

These models are deliberately independent of FastAPI, Flask, vector databases and language models.

### Representative Policy Fixtures

`backend/core/fixtures.py`

Provides synthetic policy material for deterministic local testing.

The fixtures are not real La Trobe University policy content.

Each fixture preserves representative:

- policy ID;
- policy title;
- source location;
- current/status information;
- policy text.

The `.invalid` source domain is deliberately used so fixture URLs cannot be confused with real policy locations.

### Retrieval

`backend/core/retrieval.py`

Implements a deterministic lexical keyword-overlap reference retriever.

The current retriever:

1. normalises text to lowercase;
2. extracts meaningful tokens;
3. removes defined stop words;
4. compares question tokens with policy title and text tokens;
5. calculates a simple overlap score;
6. returns candidate evidence ordered by score.

The current relevance score represents token overlap only.

It is not:

- a probability;
- calibrated confidence;
- semantic similarity;
- a final RAG retrieval score.

The retriever is deliberately simple so later semantic, vector or hybrid retrieval implementations can be compared against a stable baseline.

### Evidence Sufficiency

`backend/core/evidence.py`

Separates candidate retrieval from the decision about whether RAVIN is permitted to return a supported policy response.

A retrieval result does not automatically permit an answer.

Candidate evidence must satisfy the configured evidence threshold before it becomes supporting evidence.

If sufficient evidence is not established, the response workflow fails closed.

### Configuration

`backend/core/config.py`

Contains configurable values used by the deterministic reference implementation.

The current default evidence threshold is:

```text
0.5
```

This value is specific to the lexical reference retriever.

It must not be interpreted as 50% confidence.

The value must also not be automatically transferred to a future semantic, embedding, vector or hybrid retrieval implementation because those systems may produce scores with different meanings.

The threshold is configurable so future evaluation can compare alternative settings using the same controlled evaluation dataset.

### Controlled Messages

`backend/core/messages.py`

Contains controlled user-facing messages used by the reference response workflow.

This keeps user-facing response content separate from evidence-assessment and orchestration logic.

### Response Orchestration

`backend/core/response.py`

Coordinates:

1. question validation;
2. evidence retrieval;
3. evidence-sufficiency assessment;
4. supported reference-response construction; or
5. controlled insufficient-evidence response.

The evidence-sufficiency decision occurs before supported-answer construction.

This ordering is deliberate.

```text
Retrieve candidate evidence
        ↓
Assess sufficiency
        ↓
   ┌────┴────┐
   │         │
No           Yes
│            │
Refuse       Construct supported response
```

The supported response is currently deterministic and intentionally does not use an LLM.

### Terminal Demonstration

`backend/demo.py`

Provides a locally runnable demonstration of the reference slice without requiring:

- a frontend;
- an API framework;
- a database;
- a vector store;
- an LLM;
- a hosted service.

Run from the repository root with:

```powershell
python -m backend.demo
```

## Current Safety Rule

The central reference rule is:

> No sufficient evidence means no supported policy answer.

If sufficient evidence is not established:

- no supported policy answer is produced;
- no policy source is presented as supporting evidence;
- a controlled insufficient-evidence response is returned.

This means the existence of a retrieval result is not enough by itself to permit an answer.

## Demonstrated Baseline Behaviour

### Direct Terminology Match

Question:

```text
When can a student request an assessment extension?
```

Observed result:

```text
SUPPORTED
```

Retrieved source:

```text
Example Assessment Extension Policy
```

Observed lexical retrieval score:

```text
0.75
```

This demonstrates the supported-response path when user terminology overlaps sufficiently with the representative policy.

### Natural-Language Terminology Mismatch

Question:

```text
When can I get an exam extension?
```

Expected policy intent:

```text
Example Assessment Extension Policy
```

Current lexical baseline result:

```text
INSUFFICIENT_EVIDENCE
```

This is considered a retrieval-quality failure rather than an evidence-gating failure.

The lexical retriever does not understand that terms such as:

```text
exam
```

and:

```text
assessment
```

may refer to related concepts in this context.

The evidence gate nevertheless behaved as designed because it received weak retrieval evidence and refused to produce a supported policy answer.

This failure is intentionally retained as a baseline evaluation case rather than corrected using a growing manually maintained synonym or alias dictionary.

The same question should be included in the future RAVIN answer-quality evaluation dataset.

This will allow later semantic or hybrid retrieval implementations to be tested against the same realistic user wording.

### Unrelated Question

Question:

```text
Which colour should I paint my car?
```

Expected and observed result:

```text
INSUFFICIENT_EVIDENCE
```

The controlled refusal path returns no supporting policy source.

This demonstrates that weak incidental lexical overlap does not automatically permit a policy answer.

## Retrieval Versus Evidence Sufficiency

COPF-218 deliberately treats retrieval and evidence sufficiency as separate responsibilities.

### Retrieval asks

```text
What policy evidence might be relevant?
```

### Evidence sufficiency asks

```text
Is the retrieved evidence strong enough that RAVIN is permitted to answer?
```

This distinction is important because a retrieval system may return weak or incidental matches even for unrelated questions.

For example:

```text
Question
   ↓
weak lexical match
   ↓
retrieval result exists
   ↓
evidence score below threshold
   ↓
INSUFFICIENT_EVIDENCE
```

Therefore:

```text
retrieval result exists
```

does not mean:

```text
evidence is sufficient
```

## Threshold Configuration and Future Accuracy Evaluation

The evidence threshold is configurable so different settings can later be evaluated systematically.

For example:

```text
threshold 0.40
      ↓
evaluation dataset

threshold 0.50
      ↓
evaluation dataset

threshold 0.60
      ↓
evaluation dataset
```

Threshold changes should be justified using measured results rather than subjective judgement.

Future evaluation should consider outcomes such as:

- correct supported responses;
- incorrect supported responses;
- correct refusals;
- incorrect refusals;
- retrieval accuracy;
- source/citation correctness;
- groundedness;
- answer completeness.

A higher threshold is not automatically better.

Increasing the threshold may reduce unsupported answers while also increasing incorrect refusals.

The target should therefore be an evidence-backed balance appropriate to RAVIN's policy-answering risk profile.

## Running the Demonstration

From the repository root:

```powershell
python -m backend.demo
```

### Supported Demonstration

Use:

```text
When can a student request an assessment extension?
```

Expected:

```text
Outcome: SUPPORTED
```

with the representative Assessment Extension Policy shown as supporting evidence.

### Controlled Refusal Demonstration

Use:

```text
Which colour should I paint my car?
```

Expected:

```text
Outcome: INSUFFICIENT_EVIDENCE
```

and:

```text
Supporting sources: None
```

### Known Baseline Retrieval Limitation

Use:

```text
When can I get an exam extension?
```

Current expected lexical-baseline behaviour:

```text
Outcome: INSUFFICIENT_EVIDENCE
```

This behaviour is retained as an evaluation case for later retrieval improvement.

## Running Tests

Install the development dependency:

```powershell
python -m pip install -r requirements-dev.txt
```

Run the COPF-218 test suite:

```powershell
python -m pytest tests/test_models.py tests/test_retrieval.py tests/test_evidence.py tests/test_response.py -v
```

The current automated suite verifies:

- policy source metadata preservation;
- immutable policy records;
- supported-response evidence retention;
- insufficient-evidence source exclusion;
- deterministic retrieval;
- retrieval ordering;
- weak-evidence behaviour;
- source provenance;
- evidence-sufficiency decisions;
- evidence threshold boundary behaviour;
- configurable evidence thresholds;
- invalid threshold rejection;
- supported response construction;
- controlled insufficient-evidence behaviour;
- invalid question rejection.

At the time of this reference-slice implementation, the complete suite contains:

```text
20 automated tests
```

## Current Dependencies

The framework-neutral runtime core currently uses Python standard-library functionality only.

Development testing currently requires:

```text
pytest==9.1.1
```

FastAPI and Flask are not dependencies of the COPF-218 core.

This separation is intentional because the final application/API framework remains subject to team architecture review.

## Known Limitations

The current reference slice does not implement:

- real La Trobe policy ingestion;
- policy scraping;
- document parsing;
- document chunking;
- embeddings;
- semantic retrieval;
- vector retrieval;
- hybrid retrieval;
- reranking;
- an LLM;
- generative answer synthesis;
- grounding validation of generated claims;
- production citation formatting;
- persistent QA logging;
- API integration;
- frontend integration;
- production deployment.

The deterministic lexical retriever also performs poorly when users use terminology that differs from the policy wording.

These limitations are deliberate and provide measurable baseline conditions for later increments.

## Replacement Strategy

The reference slice is structured so implementation technologies can change behind stable boundaries.

For example:

```text
Deterministic lexical retriever
        ↓
replace with
        ↓
Semantic / vector / hybrid retriever
```

while continuing to return retrieved evidence with source provenance.

Similarly:

```text
Deterministic reference response
        ↓
replace with
        ↓
Local LLM answer generation
```

while retaining the evidence-before-answer requirement.

The eventual application adapter could similarly be:

```text
FastAPI
   ↓
RAVIN core
```

or:

```text
Flask
   ↓
RAVIN core
```

without embedding web-framework dependencies into the response core.

## Next Evaluation Increment

The next recommended technical increment is an answer-quality evaluation harness.

The evaluation dataset should contain both canonical and natural user phrasings, including cases such as:

```text
When can a student request an assessment extension?
```

and:

```text
When can I get an exam extension?
```

along with unrelated questions that should correctly produce controlled refusals.

This will establish measurable baseline results before semantic retrieval is introduced.

Future retrieval improvements can then be evaluated against the same cases to demonstrate:

- genuine improvements;
- remaining failures; and
- regressions.

## Status

This is a technical reference implementation under COPF-218.

It does not constitute final approval of:

- the retrieval strategy;
- the evidence threshold;
- the LLM;
- the vector store;
- the application framework;
- the policy ingestion approach; or
- the production architecture.
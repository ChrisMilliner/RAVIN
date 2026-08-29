> **Implementation scope:** RAVIN backend workstream
>
> **Backend checkpoint:** `5c8e2ce`
>
> **Verification baseline:** 725 passing tests
>
> **Purpose:** Source material for later project-wide documentation

# RAVIN Backend Architecture

## 1. Purpose

This document describes the backend architecture implemented for RAVIN.

It deliberately excludes unimplemented or separately owned application components. In particular, the final FastAPI application layer and user interface are integration consumers of this backend rather than backend components documented here as completed work.

## 2. Backend Objective

RAVIN is designed as an evidence-first policy question-answering backend.

The implemented backend separates evidence discovery, evidence sufficiency, answer routing, language generation, and answer release.

The principal flow is:

```text
Current policy sources
-> acquisition
-> normalization and chunking
-> semantic index
-> hybrid retrieval
-> reranking
-> structural context assembly
-> question intent assessment
-> material proposition extraction
-> evidence coverage assessment
-> deterministic answer routing
-> grounded answer generation
-> citation validation
-> generated-claim grounding validation
-> application-facing answer result
```

The important architectural rule is that a generative language model does not decide question intent, evidence sufficiency, or answer behavior.

## 3. Main Backend Areas

### `backend/ingestion`

Responsible for acquiring policy content, validating that policy material is current, normalizing text, preserving heading structure, and producing traceable `PolicyChunk` objects.

Each chunk retains:

- policy identifier
- policy title
- source URL
- status
- effective and review dates
- chunk index
- text
- heading path

This provenance is preserved so retrieved evidence can later be traced back to the originating policy.

### `backend/retrieval`

Responsible for finding and organising candidate policy evidence.

Production retrieval performs:

```text
query
-> query embedding
-> semantic + lexical hybrid scoring
-> candidate selection
-> cross-encoder reranking
-> final Top-K evidence
-> bounded structural neighbour expansion
-> grounded evidence blocks
```

Production embeddings use policy body text.

The index also retains richer retrieval text containing policy title, heading path, and body text for lexical scoring and reranking.

Retrieval scores are ranking signals. They do not independently determine whether the evidence is sufficient to answer the question.

### `backend/routing`

Responsible for deterministic question interpretation, evidence sufficiency, and answer behavior.

Implemented answer behaviors are:

- `DIRECT_ANSWER`
- `GROUNDED_OVERVIEW`
- `CLARIFY`
- `NO_GROUNDED_ANSWER`

The routing rules are:

```text
ambiguous question
-> CLARIFY

clear question + insufficient/uncertain evidence
-> NO_GROUNDED_ANSWER

focused question + sufficient evidence
-> DIRECT_ANSWER

broad question + sufficient evidence
-> GROUNDED_OVERVIEW
```

Ambiguous questions do not proceed through evidence sufficiency assessment because the question must first be clarified.

### `backend/generation`

Responsible for grounded answer wording and post-generation validation.

Generation receives:

- the user question
- an already selected answer behavior
- approved evidence text

It does not select its own behavior.

Generated answers must pass:

```text
generation
-> evidence citation validation
-> claim-to-evidence grounding validation
-> release
```

Unsupported generated output is rejected rather than returned as a grounded answer.

### `backend/llm`

Provides the replaceable language-model provider boundary.

The current local adapter communicates with Ollama, while the remainder of the backend depends on the neutral `LanguageModelProvider` contract.

### `backend/evaluation`

Provides retrieval and routing evaluation, experiment comparison, quality gates, dataset loading, and reproducibility evidence.

The evaluation code explicitly distinguishes preliminary development datasets from human-validated datasets.

### `backend/service`

Provides the application-facing integration boundary.

`RavinAnswerService` coordinates retrieval, routing, generation, validation, and source mapping.

Application adapters should call:

```python
result = service.answer(question)
```

rather than reimplementing individual backend stages.

## 4. Shared Startup Architecture

The shared bootstrap is:

```text
backend/service/bootstrap.py
-> acquire current policies
-> process policy chunks
-> load runtime configuration
-> construct providers
-> build retrieval index
-> construct RavinAnswerService
```

The resulting service is intended to be constructed once during application startup and reused.

This avoids repeatedly downloading policies, loading model weights, rebuilding embeddings, rebuilding the retrieval index, and reconstructing backend services for each user request.

## 5. Replaceable Provider Architecture

Concrete models and libraries are kept behind provider contracts.

Implemented provider boundaries include:

- embedding provider
- reranker provider
- question parser provider
- answerability provider
- entailment provider
- language-model provider

Provider selection is resolved in the composition layer rather than inside business logic.

This allows model replacement without rewriting retrieval, routing, or service orchestration.

## 6. Parser Boundary

spaCy currently supplies linguistic analysis through an adapter.

spaCy concepts such as token part-of-speech and dependency labels are converted into RAVIN-neutral parse structures.

Higher-level concepts including material requirements, material propositions, RELATION, CONDITION, CONCEPT, REQUESTED_ATTRIBUTE, and RECOVERY are RAVIN concepts rather than spaCy concepts.

## 7. Plain-English Retrieval Concepts

### Embeddings

An embedding converts text into a numerical vector that represents aspects of its meaning.

RAVIN uses embeddings so a policy passage can still be considered relevant when it does not contain exactly the same words as the user's question.

For example, a question about a student "not progressing satisfactorily" may still be semantically close to policy text describing "unsatisfactory academic progress", even though the wording is not identical.

The production embedding vectors contain many dimensions. A simplified two-dimensional example is useful for explaining the comparison.

Assume:

```text
query vector   -> (1, 1)
policy chunk A -> (1, 1)
policy chunk B -> (1, 0)
```

### Cosine Similarity

Cosine similarity compares the direction of two vectors.

For the query and policy chunk A, cosine similarity is `1.0` because their directions are identical.

For the query and policy chunk B:

```text
cosine similarity
= (1 x 1 + 1 x 0)
  / (sqrt(2) x 1)
= approximately 0.707
```

Chunk A is therefore more semantically similar to the query in this simplified example.

RAVIN does not use cosine similarity as a final evidence-sufficiency decision. Production retrieval also applies lexical scoring and cross-encoder reranking, and later routing separately assesses whether the retrieved evidence is sufficient to answer the question.

## 8. Application Boundary

The backend currently exposes its completed behavior through `RavinAnswerService`.

The CLI is an implemented application adapter.

A FastAPI adapter can use the same service boundary without duplicating retrieval or RAG implementation. An example is provided in `backend-handover.md`.

## 9. Scope Boundary

This document does not claim completion of:

- the final FastAPI application
- the final frontend/user interface
- project-wide deployment architecture
- team-owned cybersecurity deliverables
- final human-validated accuracy evaluation

Those areas should be merged into the final project documentation from the corresponding verified team deliverables.

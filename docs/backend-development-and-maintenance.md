> **Implementation scope:** RAVIN backend workstream
>
> **Backend checkpoint:** `5c8e2ce`
>
> **Verification baseline:** 725 passing tests
>
> **Purpose:** Source material for later project-wide documentation

# RAVIN Backend Development and Maintenance

## 1. Purpose

This document provides maintenance information for the implemented RAVIN backend.

It is intended as technical source material for a later project-wide maintenance guide.

## 2. Python Environment

Development has been verified using the project virtual environment.

Use:

```powershell
.\.venv\Scripts\python.exe
```

rather than relying on a system Python installation.

## 3. Runtime Configuration

### Embeddings

```text
RAVIN_EMBEDDING_PROVIDER
RAVIN_EMBEDDING_MODEL
```

Default:

```text
sentence_transformer
sentence-transformers/all-MiniLM-L6-v2
```

### Reranking

```text
RAVIN_RERANKER_PROVIDER
RAVIN_RERANKER_MODEL
```

Default:

```text
cross_encoder
cross-encoder/ms-marco-MiniLM-L6-v2
```

### Question Parsing

```text
RAVIN_PRIMARY_PARSER_PROVIDER
RAVIN_PRIMARY_PARSER_MODEL
RAVIN_FALLBACK_PARSER_PROVIDER
RAVIN_FALLBACK_PARSER_MODEL
```

Current primary default:

```text
spacy
en_core_web_sm
```

Configured fallback:

```text
spacy
en_core_web_md
```

### Answerability

```text
RAVIN_ANSWERABILITY_PROVIDER
RAVIN_ANSWERABILITY_MODEL
```

Default:

```text
cross_encoder_answerability
cross-encoder/qnli-electra-base
```

### Entailment

```text
RAVIN_ENTAILMENT_PROVIDER
RAVIN_ENTAILMENT_MODEL
```

Default:

```text
cross_encoder_entailment
cross-encoder/nli-deberta-v3-base
```

### Generation

```text
RAVIN_GENERATION_PROVIDER
RAVIN_GENERATION_MODEL
```

Default:

```text
ollama
qwen3:4b-instruct
```

## 4. Provider Registry

Current provider identifiers include:

- `sentence_transformer`
- `cross_encoder`
- `spacy`
- `cross_encoder_answerability`
- `cross_encoder_entailment`
- `ollama`

Business components consume neutral provider contracts rather than constructing these implementations directly.

## 5. Runtime Prerequisites

The current development backend has several runtime dependencies in addition to the Python environment.

Live policy acquisition requires network access to the configured policy source.

Grounded answer generation requires the configured Ollama service to be available and the configured generation model to be accessible to Ollama.

Sentence Transformer and cross-encoder providers require their configured model resources to be available. Initial model use may involve downloading model files.

The configured spaCy parsing model must also be available in the Python environment.

A missing runtime dependency should be treated as a startup or provider configuration failure rather than silently bypassed.

## 6. Shared Backend Bootstrap

The service bootstrap:

```text
acquires policies
-> processes chunks
-> creates providers
-> builds the production index
-> constructs RavinAnswerService
```

A startup failure should be treated as a real backend failure rather than silently starting with missing evidence.

## 7. Running the CLI

```powershell
.\.venv\Scripts\python.exe -m scripts.run_ravin
```

The CLI is an adapter over the same shared `RavinAnswerService` intended for API integration.

## 8. Running Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Backend checkpoint `5c8e2ce` passed 725 tests.

## 9. Documentation Audit

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_backend_documentation
```

At the documented checkpoint:

```text
modules documented -> 82/82
public definitions documented -> 291/291
parameters typed -> 171/171
returns typed -> 171/171
formatting violations -> 0
```

## 10. Retrieval Evaluation

Development evaluation:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_retrieval_evaluation
```

Experiment runner:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_retrieval_experiment
```

These workflows may perform live policy acquisition and load model weights.

## 11. Replacing a Provider

A replacement provider should:

1. implement the corresponding neutral protocol
2. be registered in the provider factory registry
3. be selectable through runtime configuration
4. pass existing provider and integration tests
5. be evaluated against the relevant development and validated datasets

Do not put vendor-specific logic into routing or retrieval orchestration.

## 12. Updating Policy Sources

When extending the corpus, verify:

- source URL
- policy identifier
- current status
- successful extraction
- non-empty normalized content
- successful chunk production
- source metadata retained in resulting chunks

## 13. External Dependencies

The implemented backend uses external libraries/models including:

- Requests
- Beautiful Soup
- Sentence Transformers
- cross-encoder models
- spaCy
- Ollama

Exact dependency versions should ultimately be drawn from the project's dependency files when the final maintenance guide is assembled.

## 14. Hugging Face Authentication Warning

Model loading may display an unauthenticated Hugging Face warning.

This warning has not prevented verified development execution.

## 15. Startup Performance

Model loading, live policy acquisition, embedding, indexing, and local generation can be comparatively expensive.

For this reason the backend service should be constructed once and reused by application adapters.

## 16. Failure Philosophy

The backend deliberately surfaces several failures rather than silently continuing, including:

- policy acquisition failure
- empty policy content
- invalid provider configuration
- malformed model output
- missing evidence citations
- unsupported generated claims

This behavior supports fail-closed answer generation.

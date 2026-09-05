> **Implementation scope:** RAVIN backend workstream
>
> **Backend checkpoint:** `5c8e2ce`
>
> **Verification baseline:** 725 passing tests
>
> **Purpose:** Source material for later project-wide documentation

# RAVIN Backend Design Decisions

## 1. Purpose

This document records major technical decisions implemented during the RAVIN backend workstream and the reasoning behind them.

## 2. Evidence-First Control Architecture

Generative language models are not used to control:

- intent classification
- evidence sufficiency
- answer routing
- evidence release

These decisions directly control whether RAVIN is allowed to answer.

Keeping them explicit and testable provides a more auditable failure mode than asking a generative model to decide whether its own response should be trusted.

Generation is therefore restricted to wording an answer from previously approved evidence.

## 3. API Framework Comparison and FastAPI Selection

Small proof-of-concept implementations were built for both FastAPI and Flask during the backend framework comparison.

The comparison verified that the backend contract could be exposed through either framework before the project committed the application layer to one framework.

FastAPI was subsequently selected as the agreed API framework.

The backend remains framework-neutral through `RavinAnswerService`, so framework-specific request handling does not need to be embedded into retrieval, routing, or generation logic.

The intended integration remains:

```text
HTTP request
-> FastAPI validation
-> RavinAnswerService.answer()
-> JSON response
```

This separation also reduces the cost of replacing the web framework later because the implemented backend service does not depend on FastAPI.

The final FastAPI application layer is not claimed here as implemented by this workstream.

## 4. Replaceable Model Providers

Model and library implementations are isolated behind provider contracts.

The project should not require business-logic changes simply because an embedding model, reranker, parser, answerability model, entailment model, or generation provider changes.

Provider composition therefore occurs at startup.

## 5. Local Language Model

The implemented generation adapter uses a local Ollama service.

Current default generation configuration:

```text
provider -> ollama
model -> qwen3:4b-instruct
temperature -> 0
```

A local model was suitable for the project because it avoids requiring a paid hosted generation API and keeps the provider replaceable.

## 6. Hybrid Retrieval and Reranking

Production retrieval combines semantic and lexical evidence discovery.

Current production configuration:

```text
Top-K -> 5
rerank depth -> 11
semantic weight -> 0.85
lexical weight -> 0.15
```

Candidate retrieval is followed by cross-encoder reranking.

The retrieval layer deliberately separates candidate discovery from evidence sufficiency.

## 7. Production Body-Only Embeddings

Production semantic embeddings use policy body text rather than the complete title-heading-body retrieval text.

The richer retrieval representation remains available for lexical matching and reranking.

This configuration emerged from retrieval experimentation rather than being assumed initially.

## 8. Structural Context Expansion

Retrieval Top-K chunks are expanded with bounded neighbouring chunks from the same policy heading.

Default context configuration:

```text
neighbor window -> 1
maximum context chunks -> 15
```

This provides surrounding policy context while preserving an explicit upper bound.

## 9. Material Proposition Coverage

A clear question is decomposed into material propositions.

Evidence sufficiency then considers whether each material proposition is:

```text
COVERED
PARTIAL
UNCOVERED
```

The deterministic outcome is:

```text
any UNCOVERED
-> INSUFFICIENT

any PARTIAL
-> UNCERTAIN

all COVERED
-> SUFFICIENT
```

This prevents one strong evidence match from masking another part of a multi-part question that is unsupported.

## 10. Clarification Versus No Grounded Answer

These are intentionally different outcomes.

`CLARIFY` means the question itself is not sufficiently resolved to determine what evidence should answer it.

`NO_GROUNDED_ANSWER` means the question is clear, but available policy evidence has not been established as sufficient.

This distinction is preserved throughout routing and service results.

## 11. Generated Citation Requirement

Generated factual responses must cite supplied evidence using approved evidence identifiers.

A citation outside the supplied evidence range causes release rejection.

## 12. Generated Claim Validation

Citation presence alone is not treated as proof of factual support.

Generated claims are also assessed against their cited policy evidence through the entailment provider.

Unsupported claims fail closed.

## 13. Claim-Support Window Refinement

An earlier approach evaluated generated claims against large evidence blocks.

During live validation, a genuine policy-supported re-admission claim received an answerability-style score of approximately `0.461832` and was incorrectly rejected.

Investigation showed the claim was strongly supported when evaluated against the relevant policy paragraph.

The implementation was refined to construct deterministic consecutive evidence windows of one to three textual units and perform NLI entailment against those focused support windows.

This allowed the release gate to assess the relevant evidence instead of diluting the relationship inside a much larger block.

## 14. Fail-Closed Generation

When generated output fails required validation, RAVIN does not fall back to releasing the unverified text.

The backend prefers:

```text
reject unsupported generated output
```

over:

```text
return plausible but unverified output
```

## 15. Quality Threshold

The project uses a minimum validated Top-1 retrieval quality threshold of 95%.

The threshold is a release/selection requirement.

A development dataset passing 95% does not itself establish validated system accuracy.

Human-validated evaluation data is required before making a validated accuracy claim.

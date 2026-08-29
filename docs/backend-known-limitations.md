> **Implementation scope:** RAVIN backend workstream
>
> **Backend checkpoint:** `5c8e2ce`
>
> **Verification baseline:** 725 passing tests
>
> **Purpose:** Source material for later project-wide documentation

# RAVIN Backend Known Limitations

## 1. Purpose

This document records limitations observed or identified during the implemented backend workstream.

It is intentionally limited to known backend evidence rather than speculating about unfinished project components.

## 2. Accuracy Is Not Yet a Validated Final Claim

Development retrieval evaluation has produced strong preliminary results.

However, the current development dataset is explicitly preliminary.

Therefore the project cannot use those results alone to claim that RAVIN is 95% accurate.

A human-validated gold-standard evaluation set is still required for a validated quality claim.

## 3. Development Thresholds Require Calibration

Some answerability and generated-claim support thresholds remain development controls.

They should be calibrated against appropriately validated data before being treated as final production operating thresholds.

## 4. Live Runtime Latency

A live vertical-slice execution involving policy acquisition, multiple model providers, local generation, and claim validation took approximately 252 seconds in development.

This is sufficient to demonstrate integration but is not an acceptable final interactive latency target.

Performance optimisation remains necessary.

## 5. Policy Text Encoding

Live policy material has shown encoding/mojibake artifacts where source characters are decoded into incorrect character sequences.

The problem was identified but intentionally deferred while backend architecture and documentation were being checkpointed.

The acquisition and normalization path should be checked for encoding handling before final presentation-quality output.

## 6. Visible Evidence Markers

Grounded generation currently uses visible evidence markers such as `[E1]` and `[E2]`.

These markers provide useful traceability for validation.

The final UI may choose to transform them into more user-friendly source presentation while retaining backend evidence traceability.

## 7. Model Download Warning

Hugging Face model loading may emit an unauthenticated access warning.

This has not blocked development execution.

## 8. Parser Edge Cases

Question parsing includes fallback and deterministic recovery because dependency parsing can produce structurally suspicious analyses.

At least one development routing question exposed an additional parser hardening case.

This remains a known hardening item rather than a reason to reopen the completed routing architecture.

## 9. Current Policy Corpus

The shared bootstrap currently uses the policy set included in the implemented development vertical slice.

The final application may require a broader current-policy corpus and a maintained refresh process.

## 10. FastAPI Integration

The shared backend service is ready to be consumed by an API adapter.

The example FastAPI integration in the handover is an integration pattern, not evidence that the final project API layer has been implemented by this workstream.

## 11. User Interface

No final user-interface behavior is documented here as completed backend work.

User documentation should be produced from the final verified application interface.

## 12. Final Security and Deployment

Project-wide production security, deployment, monitoring, and operational procedures require evidence from the corresponding completed team deliverables.

They should not be inferred from the backend implementation alone.

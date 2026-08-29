> **Implementation scope:** RAVIN backend workstream
>
> **Backend checkpoint:** `5c8e2ce`
>
> **Verification baseline:** 725 passing tests
>
> **Purpose:** Source material for later project-wide documentation

# RAVIN Retrieval and Quality Evaluation

## 1. Purpose

This document records the evaluation framework implemented for the RAVIN backend.

It distinguishes measured development performance from validated quality claims.

## 2. Primary Retrieval Metrics

### Top-1 Accuracy

The proportion of evaluated Direct Answer questions for which expected evidence appears at rank 1.

### Hit@K

The proportion for which expected evidence appears anywhere within the configured Top-K.

### Mean Reciprocal Rank

MRR rewards systems that place the first relevant result closer to the top of the ranking.

## 3. Quality Gate

The configured minimum Top-1 threshold is 95%.

Passing this numerical threshold is necessary but not sufficient for a validated result.

## 4. Dataset Validation Status

Evaluation datasets explicitly record validation status.

```text
PRELIMINARY
-> suitable for development and optimisation
-> cannot support a validated accuracy claim

HUMAN_VALIDATED
-> eligible for validated quality-gate evidence
```

This prevents development results from being presented as final system accuracy.

## 5. Evaluation Population

The evaluation framework tracks dataset behavior composition.

Direct Answer questions are used for retrieval-ranking metrics.

Other behaviors are recorded separately:

- Grounded Overview
- Clarify
- No Grounded Answer

This avoids applying ranking metrics to questions for which retrieval ranking is not the relevant expected behavior.

## 6. Grounded Overview Evaluation

Grounded Overview questions may require multiple evidence concepts.

The implemented evaluation supports expected evidence groups, where each group may contain alternative acceptable evidence locations.

A Grounded Overview question passes only when every required group is covered.

The evaluation records:

- total questions
- passed questions
- question pass rate
- total evidence groups
- covered evidence groups
- evidence-group coverage

## 7. Routing Evaluation

Routing evaluation separately measures:

- question intent
- evidence sufficiency
- final answer behavior

Macro accuracy is used so performance across classes is visible rather than allowing a large class to dominate the result.

## 8. Retrieval Experiment Framework

The experiment framework compares a baseline and candidate using the same question population, Top-K, corpus, and evaluation configuration.

It records:

- Top-1 delta
- Hit@K delta
- MRR delta
- per-question rank changes
- relative direction
- quality-gate result
- dataset-validation result
- candidate selection decision

## 9. Reproducibility Evidence

Experiment records contain provenance information including:

- dataset SHA-256
- corpus SHA-256
- Git commit
- generation timestamp
- embedding provider and model
- retrieval strategy
- semantic and lexical weights
- reranker provider and model
- rerank depth

Experiment recording requires a clean Git working tree.

## 10. Development Retrieval Result

The selected production retrieval configuration reached the development quality gate on the preliminary retrieval dataset used during optimisation.

The recorded preliminary development result for the selected retrieval configuration was:

```text
Direct Answer Top-1 Accuracy -> 100%
Hit@5 -> 100%
MRR -> 1.0000
```

These values demonstrate performance on the preliminary development dataset only.

They must not be represented as overall RAVIN accuracy or as evidence that the final system has achieved validated 100% accuracy.

The development result shows that the selected retrieval configuration satisfied the project's numerical development gate on that evaluation population.

## 11. Human Validation Still Required

A formal claim that RAVIN meets the 95% requirement requires a human-validated gold-standard question set.

That work remains separate from the preliminary optimisation dataset.

## 12. Continuous Improvement Pattern

```text
candidate change
-> run evaluation
-> compare against baseline
-> inspect per-question changes
-> check quality threshold
-> check dataset validation status
-> retain evidence record
-> decide whether candidate is eligible
```

A candidate should not be adopted solely because it improves one metric relative to the baseline.

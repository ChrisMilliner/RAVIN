# RAVIN

RAVIN is a Policy Database Chatbot being developed as part of the CSECAP Capstone Project at La Trobe University.

The project aims to provide users with answers grounded in authoritative policy material while prioritising answer accuracy, source traceability, controlled refusal when sufficient evidence cannot be found, and repeatable quality evaluation.

## Project Status

RAVIN is currently in proof-of-concept development.

The current repository establishes the development baseline for progressively implementing and evaluating the system.

Technology selections made during proof-of-concept development are not necessarily final architecture decisions.

## Core Objectives

- Retrieve relevant evidence from current authoritative policy material.
- Generate responses only when sufficient supporting evidence is available.
- Provide traceable source information with supported answers.
- Return a controlled no-answer response when sufficient evidence cannot be established.
- Measure retrieval, grounding, citation and response quality using repeatable evaluation.
- Support iterative improvement without introducing regressions.

## Repository Structure

```text
backend/      Backend application and AI/retrieval services
frontend/     User interface
evaluation/   Evaluation datasets, metrics and evaluation tooling
tests/        Automated tests
docs/         Technical and project documentation
scripts/      Development and maintenance scripts
```

## Development Status

The repository is being established incrementally. Setup, run and test instructions will be expanded as executable components are introduced.

## Authors

Developed by the CSECAP capstone project team at La Trobe University.

Full author attribution and citation information will be added before public distribution.

## Licence

Licensing terms are currently being finalised.

Until a licence is added, no permission is granted to copy, modify, redistribute or commercially use this repository beyond rights otherwise provided by law.


### Important detail

That last section is intentional.

Because we haven't finalised the specialised non-commercial reciprocal software licence yet, we are **not pretending a licence exists**.

---

# Step 6 - Create `CONTRIBUTING.md`

Paste:

```markdown
# Contributing to RAVIN

## Development Principles

Contributions should support the following principles:

1. Evidence must be established before a policy answer is generated.
2. The system must fail safely when sufficient evidence cannot be established.
3. Source provenance must be preserved throughout the response pipeline.
4. Quality changes should be measurable using repeatable tests or evaluation.
5. Secrets, credentials and ordinary-user identity data must not be committed.
6. Proof-of-concept technology choices must not automatically be treated as final architecture decisions.

## Branching

Development work should normally occur on a dedicated branch rather than directly on `main`.

Suggested branch naming:

```text
feature/<short-description>
fix/<short-description>
test/<short-description>
docs/<short-description>
chore/<short-description>
```

Examples:

```text
feature/policy-retrieval
test/grounding-evaluation
docs/architecture-baseline
chore/repository-setup
```

## Commits

Commits should be small enough to describe one meaningful change.

Example:

```text
Add initial repository development baseline
```

## Pull Requests

Before merging a contribution:

- Review the changes.
- Run applicable tests.
- Confirm that no secrets or credentials are included.
- Confirm that relevant documentation has been updated.
- Link the work to the relevant Jira issue where applicable.

More detailed contribution and review rules will be added as the development workflow matures.
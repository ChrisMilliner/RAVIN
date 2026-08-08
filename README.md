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


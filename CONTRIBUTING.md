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
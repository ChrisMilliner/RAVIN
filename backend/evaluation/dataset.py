"""
Load and validate structured retrieval-evaluation question datasets.

This module converts evaluation data into typed RAVIN evaluation
questions and rejects malformed or incomplete records before an
evaluation run begins.

Dataset loading preserves expected policy and evidence references used
to determine whether retrieval results match the intended evidence.
"""

import json
from pathlib import Path
from backend.evaluation.models import (
    EvaluationBehavior,
    EvaluationQuestion,
    ExpectedEvidence,
    ExpectedEvidenceGroup,
)

def _parse_expected_evidence(
    raw_evidence: object,
) -> ExpectedEvidence:
    if not isinstance(raw_evidence, dict):
        raise ValueError(
            "Expected evidence must be an object."
        )

    policy_id = raw_evidence.get("policy_id")
    heading_path = raw_evidence.get("heading_path")
    allow_descendants = raw_evidence.get(
        "allow_descendants",
        False,
    )
    text_contains = raw_evidence.get(
        "text_contains"
    )

    if not isinstance(policy_id, str):
        raise ValueError(
            "Expected evidence policy ID must be a string."
        )

    if not isinstance(heading_path, list):
        raise ValueError(
            "Expected evidence heading path must be a list."
        )

    if not all(
        isinstance(part, str)
        for part in heading_path
    ):
        raise ValueError(
            "Expected evidence heading path must contain strings."
        )

    if not isinstance(allow_descendants, bool):
        raise ValueError(
            "Expected evidence allow_descendants must be a boolean."
        )

    if (
        text_contains is not None
        and not isinstance(text_contains, str)
    ):
        raise ValueError(
            "Expected evidence text_contains must be a string."
        )

    return ExpectedEvidence(
        policy_id=policy_id,
        heading_path=tuple(heading_path),
        allow_descendants=allow_descendants,
        text_contains=text_contains,
    )

def _parse_expected_evidence_group(
    raw_group: object,
) -> ExpectedEvidenceGroup:
    if not isinstance(
        raw_group,
        dict,
    ):
        raise ValueError(
            "Expected evidence group must be an object."
        )

    group_id = raw_group.get("group_id")
    description = raw_group.get("description")
    raw_alternatives = raw_group.get(
        "alternatives"
    )

    if not isinstance(
        group_id,
        str,
    ):
        raise ValueError(
            "Expected evidence group ID "
            "must be a string."
        )

    if not isinstance(
        description,
        str,
    ):
        raise ValueError(
            "Expected evidence group description "
            "must be a string."
        )

    if not isinstance(
        raw_alternatives,
        list,
    ):
        raise ValueError(
            "Expected evidence group alternatives "
            "must be a list."
        )

    alternatives = tuple(
        _parse_expected_evidence(item)
        for item in raw_alternatives
    )

    return ExpectedEvidenceGroup(
        group_id=group_id,
        description=description,
        alternatives=alternatives,
    )

def _parse_question(
    raw_question: object,
) -> EvaluationQuestion:
    if not isinstance(raw_question, dict):
        raise ValueError(
            "Evaluation question must be an object."
        )

    question_id = raw_question.get("question_id")
    question = raw_question.get("question")
    raw_expected = raw_question.get(
        "expected_evidence"
    )
    raw_expected_groups = raw_question.get(
        "expected_evidence_groups",
        [],
    )
    notes = raw_question.get("notes")

    raw_behavior = raw_question.get(
        "behavior",
        EvaluationBehavior.DIRECT_ANSWER.value,
    )

    if not isinstance(question_id, str):
        raise ValueError(
            "Evaluation question ID must be a string."
        )

    if not isinstance(question, str):
        raise ValueError(
            "Evaluation question text must be a string."
        )

    if not isinstance(raw_expected, list):
        raise ValueError(
            "Expected evidence must be a list."
        )

    if notes is not None and not isinstance(
        notes,
        str,
    ):
        raise ValueError(
            "Evaluation question notes must be a string."
        )

    if not isinstance(
        raw_behavior,
        str,
    ):
        raise ValueError(
            "Evaluation question behavior must be a string."
        )

    if not isinstance(
        raw_expected_groups,
        list,
    ):
        raise ValueError(
            "Evaluation question expected evidence "
            "groups must be a list."
        )

    try:
        behavior = EvaluationBehavior(
            raw_behavior
        )
    except ValueError as exc:
        raise ValueError(
            "Evaluation question behavior must be one of: "
            "direct_answer, grounded_overview, clarify, "
            "no_grounded_answer."
        ) from exc

    expected_evidence = tuple(
        _parse_expected_evidence(item)
        for item in raw_expected
    )

    expected_evidence_groups = tuple(
        _parse_expected_evidence_group(item)
        for item in raw_expected_groups
    )

    return EvaluationQuestion(
        question_id=question_id,
        question=question,
        expected_evidence=expected_evidence,
        notes=notes,
        behavior=behavior,
        expected_evidence_groups=(
            expected_evidence_groups
        ),
    )

def load_evaluation_questions(
    path: str | Path,
) -> tuple[EvaluationQuestion, ...]:
    """Load and validate retrieval evaluation questions from a JSON dataset.

    Malformed, missing, or empty question collections are rejected before
    evaluation begins.
    """
    dataset_path = Path(path)

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Evaluation dataset root must be an object."
        )

    raw_questions = data.get("questions")

    if not isinstance(raw_questions, list):
        raise ValueError(
            "Evaluation dataset questions must be a list."
        )

    if not raw_questions:
        raise ValueError(
            "Evaluation dataset cannot be empty."
        )

    return tuple(
        _parse_question(raw_question)
        for raw_question in raw_questions
    )
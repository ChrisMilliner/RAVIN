import json
from pathlib import Path
from backend.behavior import AnswerBehavior
from backend.evaluation.routing_models import (
    RoutingEvaluationQuestion,
)
from backend.routing.models import (
    EvidenceSufficiency,
    QuestionIntent,
)

def _parse_routing_question(
    raw_question: object,
) -> RoutingEvaluationQuestion:
    if not isinstance(raw_question, dict):
        raise ValueError(
            "Routing evaluation question must "
            "be an object."
        )

    question_id = raw_question.get(
        "question_id"
    )
    question = raw_question.get(
        "question"
    )
    raw_intent = raw_question.get(
        "expected_intent"
    )
    raw_sufficiency = raw_question.get(
        "expected_sufficiency"
    )
    raw_behavior = raw_question.get(
        "expected_behavior"
    )
    notes = raw_question.get(
        "notes"
    )

    if not isinstance(
        question_id,
        str,
    ):
        raise ValueError(
            "Routing evaluation question ID "
            "must be a string."
        )

    if not isinstance(
        question,
        str,
    ):
        raise ValueError(
            "Routing evaluation question text "
            "must be a string."
        )

    if not isinstance(
        raw_intent,
        str,
    ):
        raise ValueError(
            "Routing evaluation expected intent "
            "must be a string."
        )

    if (
        raw_sufficiency is not None
        and not isinstance(
            raw_sufficiency,
            str,
        )
    ):
        raise ValueError(
            "Routing evaluation expected "
            "sufficiency must be a string "
            "or null."
        )

    if not isinstance(
        raw_behavior,
        str,
    ):
        raise ValueError(
            "Routing evaluation expected "
            "behavior must be a string."
        )

    if (
        notes is not None
        and not isinstance(
            notes,
            str,
        )
    ):
        raise ValueError(
            "Routing evaluation notes must "
            "be a string or null."
        )

    try:
        expected_intent = QuestionIntent(
            raw_intent
        )
    except ValueError as error:
        raise ValueError(
            "Routing evaluation expected intent "
            "is unsupported."
        ) from error

    if raw_sufficiency is None:
        expected_sufficiency = None
    else:
        try:
            expected_sufficiency = (
                EvidenceSufficiency(
                    raw_sufficiency
                )
            )
        except ValueError as error:
            raise ValueError(
                "Routing evaluation expected "
                "sufficiency is unsupported."
            ) from error

    try:
        expected_behavior = AnswerBehavior(
            raw_behavior
        )
    except ValueError as error:
        raise ValueError(
            "Routing evaluation expected "
            "behavior is unsupported."
        ) from error

    return RoutingEvaluationQuestion(
        question_id=question_id,
        question=question,
        expected_intent=expected_intent,
        expected_sufficiency=(
            expected_sufficiency
        ),
        expected_behavior=(
            expected_behavior
        ),
        notes=notes,
    )

def load_routing_evaluation_questions(
    path: str | Path,
) -> tuple[
    RoutingEvaluationQuestion,
    ...
]:
    dataset_path = Path(path)

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_dataset = json.load(file)

    if not isinstance(
        raw_dataset,
        dict,
    ):
        raise ValueError(
            "Routing evaluation dataset must "
            "be an object."
        )

    raw_questions = raw_dataset.get(
        "questions"
    )

    if not isinstance(
        raw_questions,
        list,
    ):
        raise ValueError(
            "Routing evaluation dataset "
            "questions must be a list."
        )

    questions = tuple(
        _parse_routing_question(
            raw_question
        )
        for raw_question in raw_questions
    )

    if not questions:
        raise ValueError(
            "Routing evaluation dataset must "
            "contain at least one question."
        )

    question_ids = tuple(
        question.question_id
        for question in questions
    )

    if (
        len(set(question_ids))
        != len(question_ids)
    ):
        raise ValueError(
            "Routing evaluation question IDs "
            "must be unique."
        )

    return questions
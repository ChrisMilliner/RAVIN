import json
from pathlib import Path
import pytest
from backend.behavior import AnswerBehavior
from backend.evaluation.routing_dataset import (
    load_routing_evaluation_questions,
)
from backend.routing.models import (
    EvidenceSufficiency,
    QuestionIntent,
)

def write_dataset(
    path: Path,
    questions: list[object],
) -> None:
    path.write_text(
        json.dumps(
            {
                "name": (
                    "RAVIN Preliminary Routing "
                    "Development Baseline v1.0"
                ),
                "status": (
                    "preliminary-not-gold-standard"
                ),
                "questions": questions,
            }
        ),
        encoding="utf-8",
    )

def test_loads_valid_focused_question(
    tmp_path: Path,
):
    dataset_path = (
        tmp_path / "routing.json"
    )

    write_dataset(
        dataset_path,
        [
            {
                "question_id": "RQ001",
                "question": (
                    "Who approves changes to "
                    "academic dress?"
                ),
                "expected_intent": "focused",
                "expected_sufficiency": (
                    "sufficient"
                ),
                "expected_behavior": (
                    "direct_answer"
                ),
                "notes": "Known supported question.",
            }
        ],
    )

    questions = (
        load_routing_evaluation_questions(
            dataset_path
        )
    )

    assert len(questions) == 1

    question = questions[0]

    assert question.question_id == "RQ001"

    assert question.expected_intent is (
        QuestionIntent.FOCUSED
    )

    assert question.expected_sufficiency is (
        EvidenceSufficiency.SUFFICIENT
    )

    assert question.expected_behavior is (
        AnswerBehavior.DIRECT_ANSWER
    )

    assert question.notes == (
        "Known supported question."
    )

def test_loads_valid_ambiguous_question(
    tmp_path: Path,
):
    dataset_path = (
        tmp_path / "routing.json"
    )

    write_dataset(
        dataset_path,
        [
            {
                "question_id": "RQ002",
                "question": (
                    "What do I need to submit?"
                ),
                "expected_intent": "ambiguous",
                "expected_sufficiency": None,
                "expected_behavior": "clarify",
            }
        ],
    )

    questions = (
        load_routing_evaluation_questions(
            dataset_path
        )
    )

    question = questions[0]

    assert question.expected_intent is (
        QuestionIntent.AMBIGUOUS
    )

    assert (
        question.expected_sufficiency
        is None
    )

    assert question.expected_behavior is (
        AnswerBehavior.CLARIFY
    )

def test_loads_valid_no_grounded_answer_question(
    tmp_path: Path,
):
    dataset_path = (
        tmp_path / "routing.json"
    )

    write_dataset(
        dataset_path,
        [
            {
                "question_id": "RQ003",
                "question": (
                    "Clear unsupported question?"
                ),
                "expected_intent": "focused",
                "expected_sufficiency": (
                    "insufficient"
                ),
                "expected_behavior": (
                    "no_grounded_answer"
                ),
            }
        ],
    )

    questions = (
        load_routing_evaluation_questions(
            dataset_path
        )
    )

    question = questions[0]

    assert question.expected_sufficiency is (
        EvidenceSufficiency.INSUFFICIENT
    )

    assert question.expected_behavior is (
        AnswerBehavior.NO_GROUNDED_ANSWER
    )

def test_rejects_non_object_question(
    tmp_path: Path,
):
    dataset_path = (
        tmp_path / "routing.json"
    )

    write_dataset(
        dataset_path,
        [
            "not-an-object",
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation question must "
            "be an object."
        ),
    ):
        load_routing_evaluation_questions(
            dataset_path
        )

def test_rejects_unsupported_intent(
    tmp_path: Path,
):
    dataset_path = (
        tmp_path / "routing.json"
    )

    write_dataset(
        dataset_path,
        [
            {
                "question_id": "RQ004",
                "question": "Question?",
                "expected_intent": "unclear",
                "expected_sufficiency": (
                    "sufficient"
                ),
                "expected_behavior": (
                    "direct_answer"
                ),
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation expected intent "
            "is unsupported."
        ),
    ):
        load_routing_evaluation_questions(
            dataset_path
        )

def test_rejects_unsupported_sufficiency(
    tmp_path: Path,
):
    dataset_path = (
        tmp_path / "routing.json"
    )

    write_dataset(
        dataset_path,
        [
            {
                "question_id": "RQ005",
                "question": "Question?",
                "expected_intent": "focused",
                "expected_sufficiency": "maybe",
                "expected_behavior": (
                    "direct_answer"
                ),
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation expected "
            "sufficiency is unsupported."
        ),
    ):
        load_routing_evaluation_questions(
            dataset_path
        )

def test_rejects_unsupported_behavior(
    tmp_path: Path,
):
    dataset_path = (
        tmp_path / "routing.json"
    )

    write_dataset(
        dataset_path,
        [
            {
                "question_id": "RQ006",
                "question": "Question?",
                "expected_intent": "focused",
                "expected_sufficiency": (
                    "sufficient"
                ),
                "expected_behavior": "guess",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation expected "
            "behavior is unsupported."
        ),
    ):
        load_routing_evaluation_questions(
            dataset_path
        )

def test_rejects_missing_questions_list(
    tmp_path: Path,
):
    dataset_path = (
        tmp_path / "routing.json"
    )

    dataset_path.write_text(
        json.dumps(
            {
                "name": "Routing dataset",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation dataset "
            "questions must be a list."
        ),
    ):
        load_routing_evaluation_questions(
            dataset_path
        )

def test_rejects_empty_dataset(
    tmp_path: Path,
):
    dataset_path = (
        tmp_path / "routing.json"
    )

    write_dataset(
        dataset_path,
        [],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation dataset must "
            "contain at least one question."
        ),
    ):
        load_routing_evaluation_questions(
            dataset_path
        )

def test_rejects_duplicate_question_ids(
    tmp_path: Path,
):
    dataset_path = (
        tmp_path / "routing.json"
    )

    question = {
        "question_id": "RQ007",
        "question": "Question?",
        "expected_intent": "focused",
        "expected_sufficiency": "sufficient",
        "expected_behavior": "direct_answer",
    }

    write_dataset(
        dataset_path,
        [
            question,
            {
                **question,
                "question": (
                    "Different question?"
                ),
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation question IDs "
            "must be unique."
        ),
    ):
        load_routing_evaluation_questions(
            dataset_path
        )

def test_rejects_uncertain_as_gold_truth(
    tmp_path: Path,
):
    dataset_path = (
        tmp_path / "routing.json"
    )

    write_dataset(
        dataset_path,
        [
            {
                "question_id": "RQ008",
                "question": "Question?",
                "expected_intent": "focused",
                "expected_sufficiency": (
                    "uncertain"
                ),
                "expected_behavior": (
                    "no_grounded_answer"
                ),
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Uncertain is a runtime fallback "
            "and cannot be evaluation truth."
        ),
    ):
        load_routing_evaluation_questions(
            dataset_path
        )

def test_rejects_contradictory_route_labels(
    tmp_path: Path,
):
    dataset_path = (
        tmp_path / "routing.json"
    )

    write_dataset(
        dataset_path,
        [
            {
                "question_id": "RQ009",
                "question": "Question?",
                "expected_intent": "focused",
                "expected_sufficiency": (
                    "sufficient"
                ),
                "expected_behavior": (
                    "no_grounded_answer"
                ),
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Expected behavior does not match "
            "the intent and evidence labels."
        ),
    ):
        load_routing_evaluation_questions(
            dataset_path
        )
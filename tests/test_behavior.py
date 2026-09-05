import pytest
from backend.behavior import AnswerBehavior
from backend.evaluation.models import (
    EvaluationBehavior,
)

def test_answer_behavior_defines_expected_values():
    assert tuple(
        behavior.value
        for behavior in AnswerBehavior
    ) == (
        "direct_answer",
        "grounded_overview",
        "clarify",
        "no_grounded_answer",
    )

def test_evaluation_behavior_reuses_answer_behavior():
    assert EvaluationBehavior is AnswerBehavior

def test_answer_behavior_can_be_created_from_value():
    behavior = AnswerBehavior(
        "grounded_overview"
    )

    assert behavior is (
        AnswerBehavior.GROUNDED_OVERVIEW
    )

def test_answer_behavior_rejects_unknown_value():
    with pytest.raises(ValueError):
        AnswerBehavior(
            "unsupported_behavior"
        )
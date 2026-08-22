from enum import Enum
from typing import cast
import pytest
from backend.behavior import AnswerBehavior
from backend.evaluation.routing_metrics import (
    calculate_class_accuracy,
    calculate_classification_accuracy,
    calculate_macro_accuracy,
    meets_classification_quality_gate,
)
from backend.routing.models import (
    QuestionIntent,
)

def test_classification_accuracy_returns_fraction_correct():
    expected = (
        QuestionIntent.FOCUSED,
        QuestionIntent.FOCUSED,
        QuestionIntent.BROAD,
        QuestionIntent.AMBIGUOUS,
    )

    predicted = (
        QuestionIntent.FOCUSED,
        QuestionIntent.BROAD,
        QuestionIntent.BROAD,
        QuestionIntent.AMBIGUOUS,
    )

    assert (
        calculate_classification_accuracy(
            expected,
            predicted,
        )
        == 0.75
    )

def test_classification_accuracy_rejects_empty_set():
    with pytest.raises(
        ValueError,
        match=(
            "Cannot calculate classification "
            "metrics from an empty evaluation set."
        ),
    ):
        calculate_classification_accuracy(
            (),
            (),
        )

def test_classification_accuracy_rejects_mismatched_counts():
    with pytest.raises(
        ValueError,
        match=(
            "Expected and predicted label counts "
            "must match."
        ),
    ):
        calculate_classification_accuracy(
            (
                QuestionIntent.FOCUSED,
                QuestionIntent.BROAD,
            ),
            (
                QuestionIntent.FOCUSED,
            ),
        )

def test_classification_accuracy_rejects_non_enum_values():
    invalid_expected = cast(
        tuple[Enum, ...],
        ("focused",),
    )

    invalid_predicted = cast(
        tuple[Enum, ...],
        ("focused",),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Expected labels must be Enum values."
        ),
    ):
        calculate_classification_accuracy(
            invalid_expected,
            (
                QuestionIntent.FOCUSED,
            ),
        )

    with pytest.raises(
        ValueError,
        match=(
            "Predicted labels must be Enum values."
        ),
    ):
        calculate_classification_accuracy(
            (
                QuestionIntent.FOCUSED,
            ),
            invalid_predicted,
        )

def test_classification_accuracy_rejects_mixed_expected_types():
    expected = cast(
        tuple[Enum, ...],
        (
            QuestionIntent.FOCUSED,
            AnswerBehavior.DIRECT_ANSWER,
        ),
    )

    predicted = cast(
        tuple[Enum, ...],
        (
            QuestionIntent.FOCUSED,
            QuestionIntent.BROAD,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Expected labels must use one "
            "Enum type."
        ),
    ):
        calculate_classification_accuracy(
            expected,
            predicted,
        )

def test_classification_accuracy_rejects_different_prediction_type():
    expected = cast(
        tuple[Enum, ...],
        (
            QuestionIntent.FOCUSED,
        ),
    )

    predicted = cast(
        tuple[Enum, ...],
        (
            AnswerBehavior.DIRECT_ANSWER,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Expected and predicted labels must "
            "use the same Enum type."
        ),
    ):
        calculate_classification_accuracy(
            expected,
            predicted,
        )

def test_class_accuracy_returns_requested_class_accuracy():
    expected = (
        QuestionIntent.FOCUSED,
        QuestionIntent.FOCUSED,
        QuestionIntent.BROAD,
        QuestionIntent.AMBIGUOUS,
    )

    predicted = (
        QuestionIntent.FOCUSED,
        QuestionIntent.BROAD,
        QuestionIntent.BROAD,
        QuestionIntent.AMBIGUOUS,
    )

    assert (
        calculate_class_accuracy(
            expected,
            predicted,
            QuestionIntent.FOCUSED,
        )
        == 0.5
    )

def test_class_accuracy_ignores_other_expected_classes():
    expected = (
        QuestionIntent.FOCUSED,
        QuestionIntent.BROAD,
        QuestionIntent.AMBIGUOUS,
    )

    predicted = (
        QuestionIntent.FOCUSED,
        QuestionIntent.FOCUSED,
        QuestionIntent.FOCUSED,
    )

    assert (
        calculate_class_accuracy(
            expected,
            predicted,
            QuestionIntent.FOCUSED,
        )
        == 1.0
    )

def test_class_accuracy_rejects_absent_class():
    expected = (
        QuestionIntent.FOCUSED,
        QuestionIntent.FOCUSED,
    )

    predicted = (
        QuestionIntent.FOCUSED,
        QuestionIntent.FOCUSED,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Cannot calculate class accuracy "
            "for a class with no expected cases."
        ),
    ):
        calculate_class_accuracy(
            expected,
            predicted,
            QuestionIntent.BROAD,
        )

def test_class_accuracy_rejects_non_enum_label():
    invalid_label = cast(
        Enum,
        "focused",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Classification label must be "
            "an Enum value."
        ),
    ):
        calculate_class_accuracy(
            (
                QuestionIntent.FOCUSED,
            ),
            (
                QuestionIntent.FOCUSED,
            ),
            invalid_label,
        )

def test_macro_accuracy_averages_class_accuracies():
    expected = (
        QuestionIntent.FOCUSED,
        QuestionIntent.FOCUSED,
        QuestionIntent.BROAD,
        QuestionIntent.AMBIGUOUS,
    )

    predicted = (
        QuestionIntent.FOCUSED,
        QuestionIntent.BROAD,
        QuestionIntent.BROAD,
        QuestionIntent.AMBIGUOUS,
    )

    result = calculate_macro_accuracy(
        expected,
        predicted,
        (
            QuestionIntent.FOCUSED,
            QuestionIntent.BROAD,
            QuestionIntent.AMBIGUOUS,
        ),
    )

    assert result == pytest.approx(
        (
            0.5
            + 1.0
            + 1.0
        )
        / 3
    )

def test_macro_accuracy_rejects_empty_labels():
    with pytest.raises(
        ValueError,
        match=(
            "Macro accuracy requires at least "
            "one classification label."
        ),
    ):
        calculate_macro_accuracy(
            (
                QuestionIntent.FOCUSED,
            ),
            (
                QuestionIntent.FOCUSED,
            ),
            (),
        )

def test_macro_accuracy_rejects_duplicate_labels():
    with pytest.raises(
        ValueError,
        match=(
            "Macro accuracy labels must "
            "be unique."
        ),
    ):
        calculate_macro_accuracy(
            (
                QuestionIntent.FOCUSED,
            ),
            (
                QuestionIntent.FOCUSED,
            ),
            (
                QuestionIntent.FOCUSED,
                QuestionIntent.FOCUSED,
            ),
        )

def test_quality_gate_passes_at_threshold():
    assert meets_classification_quality_gate(
        accuracy=0.95,
        threshold=0.95,
    )

def test_quality_gate_fails_below_threshold():
    assert not meets_classification_quality_gate(
        accuracy=0.949,
        threshold=0.95,
    )

def test_quality_gate_rejects_out_of_range_values():
    with pytest.raises(
        ValueError,
        match=(
            "Classification accuracy must be "
            "between 0 and 1."
        ),
    ):
        meets_classification_quality_gate(
            accuracy=1.01,
            threshold=0.95,
        )

    with pytest.raises(
        ValueError,
        match=(
            "Classification quality threshold "
            "must be between 0 and 1."
        ),
    ):
        meets_classification_quality_gate(
            accuracy=0.95,
            threshold=-0.01,
        )
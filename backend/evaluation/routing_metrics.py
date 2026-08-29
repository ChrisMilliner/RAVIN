"""
Calculate classification metrics for RAVIN routing evaluation.

This module measures overall and per-class routing performance and
applies configured classification quality gates. Per-class reporting
helps expose failures that could otherwise be hidden by aggregate
accuracy.

Threshold checks describe evaluation performance only for the dataset
being measured.
"""

from enum import Enum

def _validate_label_sequences(
    expected_labels: tuple[Enum, ...],
    predicted_labels: tuple[Enum, ...],
) -> None:
    if not expected_labels:
        raise ValueError(
            "Cannot calculate classification "
            "metrics from an empty evaluation set."
        )

    if len(expected_labels) != len(
        predicted_labels
    ):
        raise ValueError(
            "Expected and predicted label counts "
            "must match."
        )

    if any(
        not isinstance(label, Enum)
        for label in expected_labels
    ):
        raise ValueError(
            "Expected labels must be Enum values."
        )

    if any(
        not isinstance(label, Enum)
        for label in predicted_labels
    ):
        raise ValueError(
            "Predicted labels must be Enum values."
        )

    expected_type = type(
        expected_labels[0]
    )

    if any(
        type(label) is not expected_type
        for label in expected_labels
    ):
        raise ValueError(
            "Expected labels must use one "
            "Enum type."
        )

    if any(
        type(label) is not expected_type
        for label in predicted_labels
    ):
        raise ValueError(
            "Expected and predicted labels must "
            "use the same Enum type."
        )

def calculate_classification_accuracy(
    expected_labels: tuple[Enum, ...],
    predicted_labels: tuple[Enum, ...],
) -> float:
    _validate_label_sequences(
        expected_labels,
        predicted_labels,
    )

    correct = sum(
        1
        for expected, predicted in zip(
            expected_labels,
            predicted_labels,
        )
        if expected == predicted
    )

    return correct / len(expected_labels)

def calculate_class_accuracy(
    expected_labels: tuple[Enum, ...],
    predicted_labels: tuple[Enum, ...],
    label: Enum,
) -> float:
    _validate_label_sequences(
        expected_labels,
        predicted_labels,
    )

    if not isinstance(label, Enum):
        raise ValueError(
            "Classification label must be "
            "an Enum value."
        )

    matching_positions = tuple(
        index
        for index, expected in enumerate(
            expected_labels
        )
        if expected == label
    )

    if not matching_positions:
        raise ValueError(
            "Cannot calculate class accuracy "
            "for a class with no expected cases."
        )

    correct = sum(
        1
        for index in matching_positions
        if predicted_labels[index] == label
    )

    return correct / len(matching_positions)

def calculate_macro_accuracy(
    expected_labels: tuple[Enum, ...],
    predicted_labels: tuple[Enum, ...],
    labels: tuple[Enum, ...],
) -> float:
    _validate_label_sequences(
        expected_labels,
        predicted_labels,
    )

    if not labels:
        raise ValueError(
            "Macro accuracy requires at least "
            "one classification label."
        )

    if len(set(labels)) != len(labels):
        raise ValueError(
            "Macro accuracy labels must "
            "be unique."
        )

    class_accuracies = tuple(
        calculate_class_accuracy(
            expected_labels,
            predicted_labels,
            label,
        )
        for label in labels
    )

    return sum(class_accuracies) / len(
        class_accuracies
    )

def meets_classification_quality_gate(
    accuracy: float,
    threshold: float,
) -> bool:
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(
            "Classification accuracy must be "
            "between 0 and 1."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Classification quality threshold "
            "must be between 0 and 1."
        )

    return accuracy >= threshold
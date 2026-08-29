import pytest
from backend.generation.evidence_windows import (
    EvidenceSupportWindow,
    EvidenceSupportWindowBuilder,
    split_evidence_support_units,
)

def test_split_support_units_normalizes_whitespace():
    units = split_evidence_support_units(
        "First sentence.   Second sentence.\n"
        "Third sentence?"
    )

    assert units == (
        "First sentence.",
        "Second sentence.",
        "Third sentence?",
    )

def test_split_support_units_preserves_single_unit():
    units = split_evidence_support_units(
        "One policy statement without another sentence"
    )

    assert units == (
        "One policy statement without another sentence",
    )

def test_split_support_units_rejects_empty_evidence():
    with pytest.raises(
        ValueError,
        match="Evidence text cannot be empty",
    ):
        split_evidence_support_units(
            "   "
        )

def test_builder_creates_one_to_three_unit_windows():
    builder = EvidenceSupportWindowBuilder(
        max_units=3
    )

    windows = builder.build(
        "One. Two. Three. Four."
    )

    assert tuple(
        window.text
        for window in windows
    ) == (
        "One.",
        "One. Two.",
        "One. Two. Three.",
        "Two.",
        "Two. Three.",
        "Two. Three. Four.",
        "Three.",
        "Three. Four.",
        "Four.",
    )

def test_builder_records_window_unit_indexes():
    builder = EvidenceSupportWindowBuilder(
        max_units=3
    )

    windows = builder.build(
        "One. Two. Three."
    )

    assert windows[2] == (
        EvidenceSupportWindow(
            start_unit_index=0,
            end_unit_index=2,
            text="One. Two. Three.",
        )
    )

    assert windows[4] == (
        EvidenceSupportWindow(
            start_unit_index=1,
            end_unit_index=2,
            text="Two. Three.",
        )
    )

def test_builder_respects_smaller_maximum():
    builder = EvidenceSupportWindowBuilder(
        max_units=2
    )

    windows = builder.build(
        "One. Two. Three."
    )

    assert tuple(
        window.text
        for window in windows
    ) == (
        "One.",
        "One. Two.",
        "Two.",
        "Two. Three.",
        "Three.",
    )

def test_builder_rejects_invalid_maximum():
    with pytest.raises(
        ValueError,
        match="must be at least 1",
    ):
        EvidenceSupportWindowBuilder(
            max_units=0
        )

def test_support_window_rejects_negative_start():
    with pytest.raises(
        ValueError,
        match="start index cannot be negative",
    ):
        EvidenceSupportWindow(
            start_unit_index=-1,
            end_unit_index=0,
            text="Evidence.",
        )

def test_support_window_rejects_reversed_indexes():
    with pytest.raises(
        ValueError,
        match="cannot precede",
    ):
        EvidenceSupportWindow(
            start_unit_index=2,
            end_unit_index=1,
            text="Evidence.",
        )

def test_support_window_rejects_empty_text():
    with pytest.raises(
        ValueError,
        match="text cannot be empty",
    ):
        EvidenceSupportWindow(
            start_unit_index=0,
            end_unit_index=0,
            text="   ",
        )
from typing import cast
import pytest
from backend.routing.material_requirement_extractor import (
    extract_material_requirements,
)
from backend.routing.material_requirements import (
    MaterialQuestionRequirements,
    MaterialRequirement,
    MaterialRequirementExtractionResult,
    MaterialRequirementKind,
    MaterialRequirementResolution,
)

class FakeExtractor:
    def __init__(
        self,
        result: MaterialRequirementExtractionResult,
    ) -> None:
        self.result = result
        self.received_question: str | None = None

    def extract(
        self,
        question: str,
    ) -> MaterialRequirementExtractionResult:
        self.received_question = question

        return self.result

def _make_requirements(
) -> MaterialQuestionRequirements:
    return MaterialQuestionRequirements(
        requirements=(
            MaterialRequirement(
                kind=(
                    MaterialRequirementKind.CONCEPT
                ),
                text="Professional Equivalence",
            ),
            MaterialRequirement(
                kind=(
                    MaterialRequirementKind.TEMPORAL
                ),
                text="within 14 days",
            ),
        )
    )

def _make_result(
) -> MaterialRequirementExtractionResult:
    return MaterialRequirementExtractionResult(
        primary=_make_requirements()
    )

def test_extracts_material_requirement_result():
    expected = _make_result()

    extractor = FakeExtractor(
        expected
    )

    result = extract_material_requirements(
        "Question?",
        extractor,
    )

    assert result is expected

def test_strips_question_before_extraction():
    extractor = FakeExtractor(
        _make_result()
    )

    extract_material_requirements(
        "  What is required?  ",
        extractor,
    )

    assert extractor.received_question == (
        "What is required?"
    )

def test_preserves_dynamic_requirement_text():
    extractor = FakeExtractor(
        _make_result()
    )

    result = extract_material_requirements(
        "How long is allowed?",
        extractor,
    )

    active = result.active

    assert active is not None

    assert active.requirements[1].text == (
        "within 14 days"
    )

def test_preserves_unresolved_result():
    primary = _make_requirements()

    result = (
        MaterialRequirementExtractionResult(
            primary=primary,
            resolution=(
                MaterialRequirementResolution.UNRESOLVED
            ),
            selection=None,
        )
    )

    extracted = extract_material_requirements(
        "Question?",
        FakeExtractor(result),
    )

    assert extracted is result
    assert extracted.active is None

def test_rejects_empty_question():
    extractor = FakeExtractor(
        _make_result()
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        extract_material_requirements(
            "   ",
            extractor,
        )

def test_rejects_invalid_extractor_result():
    class InvalidExtractor:
        def extract(
            self,
            question: str,
        ) -> MaterialRequirementExtractionResult:
            return cast(
                MaterialRequirementExtractionResult,
                "invalid",
            )

    with pytest.raises(
        ValueError,
        match=(
            "Material requirement extractor "
            "must return "
            "MaterialRequirementExtractionResult."
        ),
    ):
        extract_material_requirements(
            "Question?",
            InvalidExtractor(),
        )
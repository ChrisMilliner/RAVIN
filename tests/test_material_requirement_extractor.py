from typing import cast
import pytest
from backend.routing.material_requirement_extractor import (
    extract_material_requirements,
)
from backend.routing.material_requirements import (
    MaterialQuestionRequirements,
    MaterialRequirement,
    MaterialRequirementKind,
)

class FakeExtractor:
    def __init__(
        self,
        result: MaterialQuestionRequirements,
    ) -> None:
        self.result = result
        self.received_question: str | None = None

    def extract(
        self,
        question: str,
    ) -> MaterialQuestionRequirements:
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

def test_extracts_material_requirements():
    expected = _make_requirements()

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
        _make_requirements()
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
        _make_requirements()
    )

    result = extract_material_requirements(
        "How long is allowed?",
        extractor,
    )

    assert result.requirements[1].text == (
        "within 14 days"
    )

def test_rejects_empty_question():
    extractor = FakeExtractor(
        _make_requirements()
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
        ) -> MaterialQuestionRequirements:
            return cast(
                MaterialQuestionRequirements,
                "invalid",
            )

    with pytest.raises(
        ValueError,
        match=(
            "Material requirement extractor "
            "must return "
            "MaterialQuestionRequirements."
        ),
    ):
        extract_material_requirements(
            "Question?",
            InvalidExtractor(),
        )
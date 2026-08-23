from typing import cast
import pytest
from backend.routing.material_requirements import (
    MaterialQuestionRequirements,
    MaterialRequirement,
    MaterialRequirementKind,
)

def test_creates_material_requirement():
    requirement = MaterialRequirement(
        kind=MaterialRequirementKind.CONCEPT,
        text="Professional Equivalence",
    )

    assert requirement.kind is (
        MaterialRequirementKind.CONCEPT
    )

    assert requirement.text == (
        "Professional Equivalence"
    )

def test_preserves_multiple_requirement_types():
    requirements = (
        MaterialQuestionRequirements(
            requirements=(
                MaterialRequirement(
                    kind=(
                        MaterialRequirementKind.CONCEPT
                    ),
                    text="Professional Equivalence",
                ),
                MaterialRequirement(
                    kind=(
                        MaterialRequirementKind.RELATION
                    ),
                    text="replace",
                ),
                MaterialRequirement(
                    kind=(
                        MaterialRequirementKind.TEMPORAL
                    ),
                    text="permanently",
                ),
            )
        )
    )

    assert tuple(
        requirement.kind
        for requirement
        in requirements.requirements
    ) == (
        MaterialRequirementKind.CONCEPT,
        MaterialRequirementKind.RELATION,
        MaterialRequirementKind.TEMPORAL,
    )

def test_accepts_dynamic_requirement_text():
    requirement = MaterialRequirement(
        kind=MaterialRequirementKind.TEMPORAL,
        text="within 14 days",
    )

    assert requirement.text == (
        "within 14 days"
    )

def test_rejects_invalid_requirement_kind():
    invalid_kind = cast(
        MaterialRequirementKind,
        "temporal",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Material requirement kind must "
            "be a MaterialRequirementKind."
        ),
    ):
        MaterialRequirement(
            kind=invalid_kind,
            text="within 14 days",
        )

def test_rejects_empty_requirement_text():
    with pytest.raises(
        ValueError,
        match=(
            "Material requirement text "
            "cannot be empty."
        ),
    ):
        MaterialRequirement(
            kind=MaterialRequirementKind.CONCEPT,
            text="   ",
        )

def test_rejects_empty_question_requirements():
    with pytest.raises(
        ValueError,
        match=(
            "Material question requirements "
            "cannot be empty."
        ),
    ):
        MaterialQuestionRequirements(
            requirements=(),
        )

def test_rejects_invalid_requirement_value():
    invalid_requirement = cast(
        MaterialRequirement,
        "invalid",
    )

    with pytest.raises(
        ValueError,
        match=(
            "All question requirements must "
            "be MaterialRequirement values."
        ),
    ):
        MaterialQuestionRequirements(
            requirements=(
                invalid_requirement,
            )
        )
from typing import cast
import pytest
from backend.routing.material_requirements import (
    MaterialQuestionRequirements,
    MaterialRequirement,
    MaterialRequirementKind,
    MaterialRequirementExtractionResult,
    MaterialRequirementResolution,
    MaterialRequirementSelection,
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

def _sample_question_requirements(
) -> MaterialQuestionRequirements:
    return MaterialQuestionRequirements(
        requirements=(
            MaterialRequirement(
                kind=(
                    MaterialRequirementKind.CONCEPT
                ),
                text="sample concept",
            ),
        )
    )

def test_resolved_primary_extraction_selects_primary():
    primary = _sample_question_requirements()

    result = MaterialRequirementExtractionResult(
        primary=primary,
    )

    assert (
        result.resolution
        == MaterialRequirementResolution.RESOLVED
    )

    assert (
        result.selection
        == MaterialRequirementSelection.PRIMARY
    )

    assert result.active is primary

def test_resolved_fallback_extraction_selects_fallback():
    primary = _sample_question_requirements()

    fallback = MaterialQuestionRequirements(
        requirements=(
            MaterialRequirement(
                kind=(
                    MaterialRequirementKind.RELATION
                ),
                text="apply",
            ),
        )
    )

    result = MaterialRequirementExtractionResult(
        primary=primary,
        fallback=fallback,
        resolution=(
            MaterialRequirementResolution.RESOLVED
        ),
        selection=(
            MaterialRequirementSelection.FALLBACK
        ),
    )

    assert result.active is fallback

def test_unresolved_extraction_has_no_active_interpretation():
    primary = _sample_question_requirements()

    fallback = MaterialQuestionRequirements(
        requirements=(
            MaterialRequirement(
                kind=(
                    MaterialRequirementKind.CONCEPT
                ),
                text="alternative concept",
            ),
        )
    )

    result = MaterialRequirementExtractionResult(
        primary=primary,
        fallback=fallback,
        resolution=(
            MaterialRequirementResolution.UNRESOLVED
        ),
        selection=None,
    )

    assert result.active is None

def test_unresolved_extraction_cannot_select_interpretation():
    with pytest.raises(
        ValueError,
        match=(
            "Unresolved material requirements "
            "cannot select an interpretation."
        ),
    ):
        MaterialRequirementExtractionResult(
            primary=(
                _sample_question_requirements()
            ),
            resolution=(
                MaterialRequirementResolution.UNRESOLVED
            ),
            selection=(
                MaterialRequirementSelection.PRIMARY
            ),
        )

def test_fallback_selection_requires_fallback_requirements():
    with pytest.raises(
        ValueError,
        match=(
            "Fallback material requirements "
            "must exist when fallback is selected."
        ),
    ):
        MaterialRequirementExtractionResult(
            primary=(
                _sample_question_requirements()
            ),
            selection=(
                MaterialRequirementSelection.FALLBACK
            ),
        )

def test_recovery_selection_returns_recovered_requirements():
    primary = MaterialQuestionRequirements(
        requirements=(
            MaterialRequirement(
                kind=MaterialRequirementKind.CONCEPT,
                text="incorrect primary",
            ),
        )
    )

    recovery = MaterialQuestionRequirements(
        requirements=(
            MaterialRequirement(
                kind=MaterialRequirementKind.RELATION,
                text="apply",
            ),
        )
    )

    result = MaterialRequirementExtractionResult(
        primary=primary,
        recovery=recovery,
        resolution=(
            MaterialRequirementResolution.RESOLVED
        ),
        selection=(
            MaterialRequirementSelection.RECOVERY
        ),
    )

    assert result.active is recovery


def test_recovery_selection_requires_recovered_requirements():
    primary = MaterialQuestionRequirements(
        requirements=(
            MaterialRequirement(
                kind=MaterialRequirementKind.CONCEPT,
                text="primary",
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Recovered material requirements "
            "must exist when recovery is selected."
        ),
    ):
        MaterialRequirementExtractionResult(
            primary=primary,
            resolution=(
                MaterialRequirementResolution.RESOLVED
            ),
            selection=(
                MaterialRequirementSelection.RECOVERY
            ),
        )


def test_rejects_invalid_recovered_requirements():
    primary = MaterialQuestionRequirements(
        requirements=(
            MaterialRequirement(
                kind=MaterialRequirementKind.CONCEPT,
                text="primary",
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Recovered material requirements "
            "must be MaterialQuestionRequirements."
        ),
    ):
        MaterialRequirementExtractionResult(
            primary=primary,
            recovery=cast(
                MaterialQuestionRequirements,
                "invalid",
            ),
        )

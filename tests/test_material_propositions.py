from typing import cast
import pytest
from backend.routing.material_propositions import (
    MaterialProposition,
    MaterialPropositionKind,
    MaterialQuestionPropositions,
)
from backend.routing.material_requirements import (
    MaterialRequirement,
    MaterialRequirementKind,
)

def _requirement(
    kind: MaterialRequirementKind,
    text: str,
) -> MaterialRequirement:
    return MaterialRequirement(
        kind=kind,
        text=text,
    )

def test_relational_proposition_supports_full_structure():
    proposition = MaterialProposition(
        kind=(
            MaterialPropositionKind.RELATIONAL
        ),
        subjects=(
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "Professional Equivalence",
            ),
        ),
        relations=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "replace",
            ),
        ),
        objects=(
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "academic qualification",
            ),
        ),
        qualifiers=(
            _requirement(
                MaterialRequirementKind.QUALIFIER,
                "permanently",
            ),
        ),
    )

    assert (
        proposition.relations[0].text
        == "replace"
    )

def test_relational_proposition_supports_predicate_chain():
    proposition = MaterialProposition(
        kind=(
            MaterialPropositionKind.RELATIONAL
        ),
        subjects=(
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "student",
            ),
        ),
        relations=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "have",
            ),
            _requirement(
                MaterialRequirementKind.RELATION,
                "submit",
            ),
        ),
        objects=(
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "show cause response",
            ),
        ),
    )

    assert tuple(
        relation.text
        for relation in proposition.relations
    ) == (
        "have",
        "submit",
    )

def test_information_request_can_be_relationless():
    proposition = MaterialProposition(
        kind=(
            MaterialPropositionKind.INFORMATION_REQUEST
        ),
        subjects=(
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "university applicants",
            ),
        ),
        objects=(
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "admission requirements",
            ),
        ),
    )

    assert proposition.relations == ()

def test_multiple_propositions_are_preserved():
    first = MaterialProposition(
        kind=(
            MaterialPropositionKind.RELATIONAL
        ),
        relations=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "apply",
            ),
        ),
    )

    second = MaterialProposition(
        kind=(
            MaterialPropositionKind.RELATIONAL
        ),
        relations=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "extend",
            ),
        ),
    )

    result = MaterialQuestionPropositions(
        propositions=(
            first,
            second,
        )
    )

    assert len(result.propositions) == 2

def test_proposition_supports_requested_attribute():
    proposition = MaterialProposition(
        kind=(
            MaterialPropositionKind.RELATIONAL
        ),
        relations=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "provide",
            ),
        ),
        requested_attributes=(
            _requirement(
                MaterialRequirementKind.REQUESTED_ATTRIBUTE,
                "accommodation",
            ),
        ),
    )

    assert (
        proposition.requested_attributes[0].text
        == "accommodation"
    )

def test_proposition_supports_condition_modality_and_negation():
    proposition = MaterialProposition(
        kind=(
            MaterialPropositionKind.RELATIONAL
        ),
        relations=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "make",
            ),
        ),
        conditions=(
            _requirement(
                MaterialRequirementKind.CONDITION,
                "when progress is unsatisfactory",
            ),
        ),
        modalities=(
            _requirement(
                MaterialRequirementKind.MODALITY,
                "must",
            ),
        ),
        negations=(
            _requirement(
                MaterialRequirementKind.NEGATION,
                "not",
            ),
        ),
    )

    assert proposition.conditions
    assert proposition.modalities
    assert proposition.negations

def test_relational_proposition_requires_relation():
    with pytest.raises(
        ValueError,
        match=(
            "Relational propositions must "
            "contain at least one relation."
        ),
    ):
        MaterialProposition(
            kind=(
                MaterialPropositionKind.RELATIONAL
            ),
            subjects=(
                _requirement(
                    MaterialRequirementKind.CONCEPT,
                    "student",
                ),
            ),
        )

def test_information_request_rejects_relation():
    with pytest.raises(
        ValueError,
        match=(
            "Information request propositions "
            "cannot contain relations."
        ),
    ):
        MaterialProposition(
            kind=(
                MaterialPropositionKind.INFORMATION_REQUEST
            ),
            relations=(
                _requirement(
                    MaterialRequirementKind.RELATION,
                    "provide",
                ),
            ),
        )

def test_relation_field_rejects_wrong_requirement_kind():
    with pytest.raises(
        ValueError,
        match=(
            "Material proposition relations "
            "must contain relation requirements."
        ),
    ):
        MaterialProposition(
            kind=(
                MaterialPropositionKind.RELATIONAL
            ),
            relations=(
                _requirement(
                    MaterialRequirementKind.CONCEPT,
                    "wrong",
                ),
            ),
        )

def test_rejects_invalid_proposition_value():
    valid = MaterialProposition(
        kind=(
            MaterialPropositionKind.RELATIONAL
        ),
        relations=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "apply",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "All question propositions must "
            "be MaterialProposition values."
        ),
    ):
        MaterialQuestionPropositions(
            propositions=(
                valid,
                cast(
                    MaterialProposition,
                    "invalid",
                ),
            )
        )

def test_question_propositions_cannot_be_empty():
    with pytest.raises(
        ValueError,
        match=(
            "Material question propositions "
            "cannot be empty."
        ),
    ):
        MaterialQuestionPropositions(
            propositions=()
        )
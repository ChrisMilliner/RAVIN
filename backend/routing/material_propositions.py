from dataclasses import dataclass
from enum import Enum
from backend.routing.material_requirements import (
    MaterialRequirement,
    MaterialRequirementKind,
)

class MaterialPropositionKind(
    str,
    Enum,
):
    RELATIONAL = "relational"
    INFORMATION_REQUEST = (
        "information_request"
    )

@dataclass(frozen=True)
class MaterialProposition:
    kind: MaterialPropositionKind

    subjects: tuple[
        MaterialRequirement,
        ...
    ] = ()

    relations: tuple[
        MaterialRequirement,
        ...
    ] = ()

    objects: tuple[
        MaterialRequirement,
        ...
    ] = ()

    qualifiers: tuple[
        MaterialRequirement,
        ...
    ] = ()

    conditions: tuple[
        MaterialRequirement,
        ...
    ] = ()

    modalities: tuple[
        MaterialRequirement,
        ...
    ] = ()

    negations: tuple[
        MaterialRequirement,
        ...
    ] = ()

    requested_attributes: tuple[
        MaterialRequirement,
        ...
    ] = ()

    def __post_init__(self) -> None:
        if not isinstance(
            self.kind,
            MaterialPropositionKind,
        ):
            raise ValueError(
                "Material proposition kind must "
                "be a MaterialPropositionKind."
            )

        fields = (
            self.subjects,
            self.relations,
            self.objects,
            self.qualifiers,
            self.conditions,
            self.modalities,
            self.negations,
            self.requested_attributes,
        )

        if not all(
            isinstance(field, tuple)
            for field in fields
        ):
            raise ValueError(
                "Material proposition fields "
                "must be tuples."
            )

        if not all(
            isinstance(
                requirement,
                MaterialRequirement,
            )
            for field in fields
            for requirement in field
        ):
            raise ValueError(
                "Material proposition values "
                "must be MaterialRequirement "
                "instances."
            )

        self._validate_requirement_kinds(
            self.relations,
            MaterialRequirementKind.RELATION,
            "relations",
        )

        self._validate_requirement_kinds(
            self.qualifiers,
            MaterialRequirementKind.QUALIFIER,
            "qualifiers",
        )

        self._validate_requirement_kinds(
            self.conditions,
            MaterialRequirementKind.CONDITION,
            "conditions",
        )

        self._validate_requirement_kinds(
            self.modalities,
            MaterialRequirementKind.MODALITY,
            "modalities",
        )

        self._validate_requirement_kinds(
            self.negations,
            MaterialRequirementKind.NEGATION,
            "negations",
        )

        self._validate_requirement_kinds(
            self.requested_attributes,
            MaterialRequirementKind.REQUESTED_ATTRIBUTE,
            "requested attributes",
        )

        if (
            self.kind
            == MaterialPropositionKind.RELATIONAL
            and not self.relations
        ):
            raise ValueError(
                "Relational propositions must "
                "contain at least one relation."
            )

        if (
            self.kind
            == MaterialPropositionKind.INFORMATION_REQUEST
            and self.relations
        ):
            raise ValueError(
                "Information request propositions "
                "cannot contain relations."
            )

        if not any(fields):
            raise ValueError(
                "Material proposition cannot "
                "be empty."
            )

    def _validate_requirement_kinds(
        self,
        requirements: tuple[
            MaterialRequirement,
            ...
        ],
        expected_kind: MaterialRequirementKind,
        field_name: str,
    ) -> None:
        if not all(
            requirement.kind == expected_kind
            for requirement in requirements
        ):
            raise ValueError(
                f"Material proposition "
                f"{field_name} must contain "
                f"{expected_kind.value} "
                f"requirements."
            )

@dataclass(frozen=True)
class MaterialQuestionPropositions:
    propositions: tuple[
        MaterialProposition,
        ...
    ]

    def __post_init__(self) -> None:
        if not self.propositions:
            raise ValueError(
                "Material question propositions "
                "cannot be empty."
            )

        if not all(
            isinstance(
                proposition,
                MaterialProposition,
            )
            for proposition
            in self.propositions
        ):
            raise ValueError(
                "All question propositions must "
                "be MaterialProposition values."
            )
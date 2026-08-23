from dataclasses import dataclass
from enum import Enum

class MaterialRequirementKind(
    str,
    Enum,
):
    CONCEPT = "concept"
    RELATION = "relation"
    CONDITION = "condition"
    TEMPORAL = "temporal"
    QUANTITY = "quantity"
    MODALITY = "modality"
    NEGATION = "negation"
    SCOPE = "scope"
    REQUESTED_ATTRIBUTE = (
        "requested_attribute"
    )

@dataclass(frozen=True)
class MaterialRequirement:
    kind: MaterialRequirementKind
    text: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.kind,
            MaterialRequirementKind,
        ):
            raise ValueError(
                "Material requirement kind must "
                "be a MaterialRequirementKind."
            )

        if not self.text.strip():
            raise ValueError(
                "Material requirement text "
                "cannot be empty."
            )

@dataclass(frozen=True)
class MaterialQuestionRequirements:
    requirements: tuple[
        MaterialRequirement,
        ...
    ]

    def __post_init__(self) -> None:
        if not self.requirements:
            raise ValueError(
                "Material question requirements "
                "cannot be empty."
            )

        if not all(
            isinstance(
                requirement,
                MaterialRequirement,
            )
            for requirement
            in self.requirements
        ):
            raise ValueError(
                "All question requirements must "
                "be MaterialRequirement values."
            )
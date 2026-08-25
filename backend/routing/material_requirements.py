from dataclasses import dataclass
from enum import Enum

class MaterialRequirementKind(
    str,
    Enum,
):
    CONCEPT = "concept"
    RELATION = "relation"
    QUALIFIER = "qualifier"
    CONDITION = "condition"
    TEMPORAL = "temporal"
    QUANTITY = "quantity"
    MODALITY = "modality"
    NEGATION = "negation"
    SCOPE = "scope"
    REQUESTED_ATTRIBUTE = (
        "requested_attribute"
    )

class MaterialRequirementResolution(
    str,
    Enum,
):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"

class MaterialRequirementSelection(
    str,
    Enum,
):
    PRIMARY = "primary"
    FALLBACK = "fallback"

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

@dataclass(frozen=True)
class MaterialRequirementExtractionResult:
    primary: MaterialQuestionRequirements
    fallback: (
        MaterialQuestionRequirements | None
    ) = None
    resolution: (
        MaterialRequirementResolution
    ) = MaterialRequirementResolution.RESOLVED
    selection: (
        MaterialRequirementSelection | None
    ) = MaterialRequirementSelection.PRIMARY

    def __post_init__(self) -> None:
        if not isinstance(
            self.primary,
            MaterialQuestionRequirements,
        ):
            raise ValueError(
                "Primary material requirements "
                "must be MaterialQuestionRequirements."
            )

        if (
            self.fallback is not None
            and not isinstance(
                self.fallback,
                MaterialQuestionRequirements,
            )
        ):
            raise ValueError(
                "Fallback material requirements "
                "must be MaterialQuestionRequirements."
            )

        if not isinstance(
            self.resolution,
            MaterialRequirementResolution,
        ):
            raise ValueError(
                "Material requirement resolution "
                "must be a "
                "MaterialRequirementResolution."
            )

        if (
            self.selection is not None
            and not isinstance(
                self.selection,
                MaterialRequirementSelection,
            )
        ):
            raise ValueError(
                "Material requirement selection "
                "must be a "
                "MaterialRequirementSelection."
            )

        if (
            self.resolution
            == MaterialRequirementResolution.UNRESOLVED
            and self.selection is not None
        ):
            raise ValueError(
                "Unresolved material requirements "
                "cannot select an interpretation."
            )

        if (
            self.resolution
            == MaterialRequirementResolution.RESOLVED
            and self.selection is None
        ):
            raise ValueError(
                "Resolved material requirements "
                "must select an interpretation."
            )

        if (
            self.selection
            == MaterialRequirementSelection.FALLBACK
            and self.fallback is None
        ):
            raise ValueError(
                "Fallback material requirements "
                "must exist when fallback is selected."
            )

    @property
    def active(
        self,
    ) -> MaterialQuestionRequirements | None:
        if (
            self.resolution
            == MaterialRequirementResolution.UNRESOLVED
        ):
            return None

        if (
            self.selection
            == MaterialRequirementSelection.FALLBACK
        ):
            return self.fallback

        return self.primary
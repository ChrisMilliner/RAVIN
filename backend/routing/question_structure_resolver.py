"""
Resolve the question structure used by downstream routing analysis.

The resolver coordinates primary parsing, configured fallback parsing,
reliability checks, and optional deterministic recovery. It exposes one
shared resolved structure so material-requirement and proposition
extraction operate from the same interpretation of the question.

Unresolved structure remains explicit rather than being silently
replaced with a generated interpretation.
"""

from dataclasses import dataclass
from enum import Enum
from backend.routing.question_parser import (
    QuestionParse,
    QuestionParseResult,
    QuestionParserService,
)
from backend.routing.question_structure_recovery import (
    QuestionStructureRecoveryProvider,
    recover_question_structure,
)

class QuestionStructureSelection(
    str,
    Enum,
):
    PRIMARY = "primary"
    FALLBACK = "fallback"
    RECOVERY = "recovery"

@dataclass(frozen=True)
class QuestionStructureResolutionResult:
    primary: QuestionParse
    fallback: QuestionParse | None = None
    recovery: QuestionParse | None = None
    selection: QuestionStructureSelection | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.primary,
            QuestionParse,
        ):
            raise ValueError(
                "Primary structure must be a QuestionParse."
            )

        if (
            self.fallback is not None
            and not isinstance(
                self.fallback,
                QuestionParse,
            )
        ):
            raise ValueError(
                "Fallback structure must be "
                "a QuestionParse or None."
            )

        if (
            self.recovery is not None
            and not isinstance(
                self.recovery,
                QuestionParse,
            )
        ):
            raise ValueError(
                "Recovery structure must be "
                "a QuestionParse or None."
            )

        if (
            self.selection
            is not None
            and not isinstance(
                self.selection,
                QuestionStructureSelection,
            )
        ):
            raise ValueError(
                "Selection must be a "
                "QuestionStructureSelection or None."
            )

        if (
            self.selection
            == QuestionStructureSelection.FALLBACK
            and self.fallback is None
        ):
            raise ValueError(
                "Fallback selection requires "
                "a fallback structure."
            )

        if (
            self.selection
            == QuestionStructureSelection.RECOVERY
            and self.recovery is None
        ):
            raise ValueError(
                "Recovery selection requires "
                "a recovery structure."
            )

    @property
    def active(
        self,
    ) -> QuestionParse | None:
        if (
            self.selection
            == QuestionStructureSelection.PRIMARY
        ):
            return self.primary

        if (
            self.selection
            == QuestionStructureSelection.FALLBACK
        ):
            return self.fallback

        if (
            self.selection
            == QuestionStructureSelection.RECOVERY
        ):
            return self.recovery

        return None

    @property
    def resolved(
        self,
    ) -> bool:
        return self.active is not None

class QuestionStructureResolver:
    def __init__(
        self,
        parser: QuestionParserService,
        recovery_provider: (
            QuestionStructureRecoveryProvider
            | None
        ) = None,
    ) -> None:
        self._parser = parser
        self._recovery_provider = (
            recovery_provider
        )

    def resolve(
        self,
        question: str,
    ) -> QuestionStructureResolutionResult:
        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        parse_result = self._parser.parse(
            question
        )

        return self._resolve_parse_result(
            question,
            parse_result,
        )

    def _resolve_parse_result(
        self,
        question: str,
        parse_result: QuestionParseResult,
    ) -> QuestionStructureResolutionResult:
        if not parse_result.primary_suspicious:
            return QuestionStructureResolutionResult(
                primary=parse_result.primary,
                fallback=parse_result.fallback,
                selection=(
                    QuestionStructureSelection.PRIMARY
                ),
            )

        if (
            parse_result.fallback is not None
            and not parse_result.fallback_suspicious
        ):
            return QuestionStructureResolutionResult(
                primary=parse_result.primary,
                fallback=parse_result.fallback,
                selection=(
                    QuestionStructureSelection.FALLBACK
                ),
            )

        recovery = None

        if self._recovery_provider is not None:
            recovery = recover_question_structure(
                question,
                parse_result,
                self._recovery_provider,
            )

        if recovery is not None:
            return QuestionStructureResolutionResult(
                primary=parse_result.primary,
                fallback=parse_result.fallback,
                recovery=recovery,
                selection=(
                    QuestionStructureSelection.RECOVERY
                ),
            )

        return QuestionStructureResolutionResult(
            primary=parse_result.primary,
            fallback=parse_result.fallback,
            recovery=recovery,
            selection=None,
        )
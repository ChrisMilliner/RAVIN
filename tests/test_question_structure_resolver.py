import pytest
from backend.routing.question_parser import (
    ParsedToken,
    QuestionParse,
    QuestionParseResult,
)
from backend.routing.question_structure_resolver import (
    QuestionStructureResolutionResult,
    QuestionStructureResolver,
    QuestionStructureSelection,
)

def _reliable_parse(
    text: str = "apply",
) -> QuestionParse:
    return QuestionParse(
        tokens=(
            ParsedToken(
                index=0,
                text=text,
                lemma=text,
                pos="VERB",
                tag="VBP",
                dependency="ROOT",
                head_index=0,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
        ),
        noun_phrases=(),
    )

def _suspicious_parse(
) -> QuestionParse:
    return QuestionParse(
        tokens=(
            ParsedToken(
                index=0,
                text="student",
                lemma="student",
                pos="NOUN",
                tag="NN",
                dependency="ROOT",
                head_index=0,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
        ),
        noun_phrases=(),
    )

class FakeParser:
    def __init__(
        self,
        result: QuestionParseResult,
    ) -> None:
        self._result = result
        self.received_question: str | None = None

    def parse(
        self,
        question: str,
    ) -> QuestionParseResult:
        self.received_question = question
        return self._result

class FakeRecoveryProvider:
    def __init__(
        self,
        result: QuestionParse | None,
    ) -> None:
        self._result = result
        self.call_count = 0

    def recover(
        self,
        question: str,
        parse_result: QuestionParseResult,
    ) -> QuestionParse | None:
        self.call_count += 1
        return self._result

def test_resolver_selects_reliable_primary():
    primary = _reliable_parse()

    parser = FakeParser(
        QuestionParseResult(
            primary=primary,
        )
    )

    recovery = FakeRecoveryProvider(
        _reliable_parse(
            "recover"
        )
    )

    resolver = QuestionStructureResolver(
        parser,
        recovery_provider=recovery,
    )

    result = resolver.resolve(
        "Can a student apply?"
    )

    assert (
        result.selection
        == QuestionStructureSelection.PRIMARY
    )

    assert result.active is primary
    assert result.resolved
    assert recovery.call_count == 0

def test_resolver_selects_reliable_fallback():
    primary = _suspicious_parse()
    fallback = _reliable_parse()

    resolver = QuestionStructureResolver(
        FakeParser(
            QuestionParseResult(
                primary=primary,
                fallback=fallback,
            )
        )
    )

    result = resolver.resolve(
        "Can a student apply?"
    )

    assert (
        result.selection
        == QuestionStructureSelection.FALLBACK
    )

    assert result.active is fallback
    assert result.resolved

def test_resolver_selects_recovery_when_normal_parses_fail():
    primary = _suspicious_parse()

    fallback = _suspicious_parse()

    recovery_parse = _reliable_parse(
        "enrol"
    )

    recovery = FakeRecoveryProvider(
        recovery_parse
    )

    resolver = QuestionStructureResolver(
        FakeParser(
            QuestionParseResult(
                primary=primary,
                fallback=fallback,
            )
        ),
        recovery_provider=recovery,
    )

    result = resolver.resolve(
        "Can a student enrol?"
    )

    assert (
        result.selection
        == QuestionStructureSelection.RECOVERY
    )

    assert result.active is recovery_parse
    assert result.recovery is recovery_parse
    assert result.resolved
    assert recovery.call_count == 1

def test_resolver_returns_unresolved_when_recovery_fails():
    primary = _suspicious_parse()
    fallback = _suspicious_parse()

    resolver = QuestionStructureResolver(
        FakeParser(
            QuestionParseResult(
                primary=primary,
                fallback=fallback,
            )
        ),
        recovery_provider=(
            FakeRecoveryProvider(
                None
            )
        ),
    )

    result = resolver.resolve(
        "Question?"
    )

    assert result.selection is None
    assert result.active is None
    assert not result.resolved

def test_resolver_can_be_unresolved_without_recovery_provider():
    primary = _suspicious_parse()

    resolver = QuestionStructureResolver(
        FakeParser(
            QuestionParseResult(
                primary=primary,
            )
        )
    )

    result = resolver.resolve(
        "Question?"
    )

    assert result.selection is None
    assert result.active is None
    assert not result.resolved

def test_resolver_strips_question():
    parser = FakeParser(
        QuestionParseResult(
            primary=_reliable_parse(),
        )
    )

    resolver = QuestionStructureResolver(
        parser
    )

    resolver.resolve(
        "  Can a student apply?  "
    )

    assert (
        parser.received_question
        == "Can a student apply?"
    )

def test_resolver_rejects_empty_question():
    resolver = QuestionStructureResolver(
        FakeParser(
            QuestionParseResult(
                primary=_reliable_parse(),
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        resolver.resolve("   ")

def test_result_fallback_selection_requires_fallback():
    with pytest.raises(
        ValueError,
        match=(
            "Fallback selection requires "
            "a fallback structure."
        ),
    ):
        QuestionStructureResolutionResult(
            primary=_reliable_parse(),
            selection=(
                QuestionStructureSelection.FALLBACK
            ),
        )

def test_result_recovery_selection_requires_recovery():
    with pytest.raises(
        ValueError,
        match=(
            "Recovery selection requires "
            "a recovery structure."
        ),
    ):
        QuestionStructureResolutionResult(
            primary=_reliable_parse(),
            selection=(
                QuestionStructureSelection.RECOVERY
            ),
        )

def test_result_primary_selection_exposes_primary():
    primary = _reliable_parse()

    result = QuestionStructureResolutionResult(
        primary=primary,
        selection=(
            QuestionStructureSelection.PRIMARY
        ),
    )

    assert result.active is primary

def test_result_unselected_has_no_active_structure():
    result = QuestionStructureResolutionResult(
        primary=_suspicious_parse(),
    )

    assert result.active is None
    assert not result.resolved
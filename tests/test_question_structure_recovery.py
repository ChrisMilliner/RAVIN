from typing import cast
import pytest
from backend.routing.question_parser import (
    ParsedToken,
    QuestionParse,
    QuestionParseResult,
)
from backend.routing.question_structure_recovery import (
    recover_question_structure,
)

class FakeRecoveryProvider:
    def __init__(
        self,
        result: QuestionParse | None,
    ) -> None:
        self.result = result
        self.received_question: str | None = None
        self.received_parse_result: (
            QuestionParseResult | None
        ) = None

    def recover(
        self,
        question: str,
        parse_result: QuestionParseResult,
    ) -> QuestionParse | None:
        self.received_question = question
        self.received_parse_result = (
            parse_result
        )

        return self.result

def _parse(
) -> QuestionParse:
    return QuestionParse(
        tokens=(
            ParsedToken(
                index=0,
                text="apply",
                lemma="apply",
                pos="VERB",
                tag="VB",
                dependency="ROOT",
                head_index=0,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
        ),
        noun_phrases=(),
    )

def _parse_result(
) -> QuestionParseResult:
    return QuestionParseResult(
        primary=_parse()
    )

def test_returns_recovered_question_parse():
    expected = _parse()

    provider = FakeRecoveryProvider(
        expected
    )

    result = recover_question_structure(
        "Can members apply?",
        _parse_result(),
        provider,
    )

    assert result is expected

def test_strips_question_before_recovery():
    provider = FakeRecoveryProvider(
        _parse()
    )

    recover_question_structure(
        "  Can members apply?  ",
        _parse_result(),
        provider,
    )

    assert provider.received_question == (
        "Can members apply?"
    )

def test_passes_parse_result_to_provider():
    parse_result = _parse_result()

    provider = FakeRecoveryProvider(
        _parse()
    )

    recover_question_structure(
        "Question?",
        parse_result,
        provider,
    )

    assert (
        provider.received_parse_result
        is parse_result
    )

def test_none_means_structure_was_not_recovered():
    result = recover_question_structure(
        "Question?",
        _parse_result(),
        FakeRecoveryProvider(None),
    )

    assert result is None

def test_rejects_empty_question():
    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        recover_question_structure(
            "   ",
            _parse_result(),
            FakeRecoveryProvider(None),
        )

def test_rejects_invalid_parse_result():
    with pytest.raises(
        ValueError,
        match=(
            "Parse result must be a "
            "QuestionParseResult."
        ),
    ):
        recover_question_structure(
            "Question?",
            cast(
                QuestionParseResult,
                "invalid",
            ),
            FakeRecoveryProvider(None),
        )

def test_rejects_invalid_provider_result():
    class InvalidProvider:
        def recover(
            self,
            question: str,
            parse_result: QuestionParseResult,
        ) -> QuestionParse | None:
            return cast(
                QuestionParse,
                "invalid",
            )

    with pytest.raises(
        ValueError,
        match=(
            "Question structure recovery "
            "provider must return a "
            "QuestionParse or None."
        ),
    ):
        recover_question_structure(
            "Question?",
            _parse_result(),
            InvalidProvider(),
        )
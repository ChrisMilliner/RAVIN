from typing import cast
import pytest
from backend.routing.question_parser import (
    ParsedToken,
    QuestionParse,
    QuestionParser,
    is_question_parse_suspicious,
)

class FakeParseProvider:
    def __init__(
        self,
        result: QuestionParse,
    ) -> None:
        self.result = result
        self.received_questions: list[str] = []

    def parse(
        self,
        question: str,
    ) -> QuestionParse:
        self.received_questions.append(
            question
        )

        return self.result

def _token(
    *,
    index: int,
    text: str,
    pos: str,
    dependency: str,
    head_index: int,
    tag: str = "",
) -> ParsedToken:
    return ParsedToken(
        index=index,
        text=text,
        lemma=text.lower(),
        pos=pos,
        tag=tag,
        dependency=dependency,
        head_index=head_index,
        is_stop=False,
        is_punct=False,
        is_alpha=True,
    )

def _usable_parse(
) -> QuestionParse:
    return QuestionParse(
        tokens=(
            _token(
                index=0,
                text="students",
                pos="NOUN",
                dependency="nsubj",
                head_index=1,
            ),
            _token(
                index=1,
                text="apply",
                pos="VERB",
                dependency="ROOT",
                head_index=1,
            ),
        ),
        noun_phrases=(),
    )

def _nominal_root_parse(
) -> QuestionParse:
    return QuestionParse(
        tokens=(
            _token(
                index=0,
                text="support",
                pos="NOUN",
                dependency="ROOT",
                head_index=0,
            ),
        ),
        noun_phrases=(),
    )

def _modal_without_verb_parse(
) -> QuestionParse:
    return QuestionParse(
        tokens=(
            _token(
                index=0,
                text="Can",
                pos="AUX",
                tag="MD",
                dependency="ROOT",
                head_index=0,
            ),
            _token(
                index=1,
                text="student",
                pos="NOUN",
                dependency="nsubj",
                head_index=0,
            ),
        ),
        noun_phrases=(),
    )

def test_usable_parse_is_not_suspicious():
    assert not is_question_parse_suspicious(
        _usable_parse()
    )

def test_nominal_root_is_suspicious():
    assert is_question_parse_suspicious(
        _nominal_root_parse()
    )

def test_modal_root_without_verb_is_suspicious():
    assert is_question_parse_suspicious(
        _modal_without_verb_parse()
    )

def test_multiple_roots_are_suspicious():
    parse = QuestionParse(
        tokens=(
            _token(
                index=0,
                text="apply",
                pos="VERB",
                dependency="ROOT",
                head_index=0,
            ),
            _token(
                index=1,
                text="extend",
                pos="VERB",
                dependency="ROOT",
                head_index=1,
            ),
        ),
        noun_phrases=(),
    )

    assert is_question_parse_suspicious(
        parse
    )

def test_parser_strips_question_before_primary_provider():
    provider = FakeParseProvider(
        _usable_parse()
    )

    parser = QuestionParser(
        primary_provider=provider,
    )

    parser.parse(
        "  Can students apply?  "
    )

    assert provider.received_questions == [
        "Can students apply?"
    ]

def test_usable_primary_parse_does_not_create_fallback():
    primary = FakeParseProvider(
        _usable_parse()
    )

    fallback_factory_calls = 0

    def fallback_factory(
    ) -> FakeParseProvider:
        nonlocal fallback_factory_calls

        fallback_factory_calls += 1

        return FakeParseProvider(
            _usable_parse()
        )

    parser = QuestionParser(
        primary_provider=primary,
        fallback_provider_factory=(
            fallback_factory
        ),
    )

    result = parser.parse(
        "Can students apply?"
    )

    assert result.primary is primary.result
    assert result.fallback is None
    assert not result.used_fallback
    assert fallback_factory_calls == 0

def test_suspicious_primary_parse_uses_fallback():
    primary = FakeParseProvider(
        _nominal_root_parse()
    )

    fallback = FakeParseProvider(
        _usable_parse()
    )

    parser = QuestionParser(
        primary_provider=primary,
        fallback_provider_factory=(
            lambda: fallback
        ),
    )

    result = parser.parse(
        "How does support work?"
    )

    assert result.primary is primary.result
    assert result.fallback is fallback.result
    assert result.used_fallback

def test_fallback_provider_is_loaded_lazily_once():
    primary = FakeParseProvider(
        _nominal_root_parse()
    )

    fallback = FakeParseProvider(
        _usable_parse()
    )

    fallback_factory_calls = 0

    def fallback_factory(
    ) -> FakeParseProvider:
        nonlocal fallback_factory_calls

        fallback_factory_calls += 1

        return fallback

    parser = QuestionParser(
        primary_provider=primary,
        fallback_provider_factory=(
            fallback_factory
        ),
    )

    parser.parse(
        "First question?"
    )

    parser.parse(
        "Second question?"
    )

    assert fallback_factory_calls == 1

    assert fallback.received_questions == [
        "First question?",
        "Second question?",
    ]

def test_suspicious_parse_without_fallback_returns_primary():
    primary = FakeParseProvider(
        _nominal_root_parse()
    )

    parser = QuestionParser(
        primary_provider=primary,
    )

    result = parser.parse(
        "Question?"
    )

    assert result.primary is primary.result
    assert result.fallback is None
    assert not result.used_fallback

def test_rejects_empty_question():
    parser = QuestionParser(
        primary_provider=FakeParseProvider(
            _usable_parse()
        ),
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        parser.parse(
            "   "
        )

def test_rejects_invalid_primary_provider_result():
    class InvalidProvider:
        def parse(
            self,
            question: str,
        ) -> QuestionParse:
            return cast(
                QuestionParse,
                "invalid",
            )

    parser = QuestionParser(
        primary_provider=InvalidProvider(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Question parse provider must "
            "return a QuestionParse."
        ),
    ):
        parser.parse(
            "Question?"
        )

def test_rejects_invalid_fallback_provider_result():
    primary = FakeParseProvider(
        _nominal_root_parse()
    )

    class InvalidProvider:
        def parse(
            self,
            question: str,
        ) -> QuestionParse:
            return cast(
                QuestionParse,
                "invalid",
            )

    parser = QuestionParser(
        primary_provider=primary,
        fallback_provider_factory=(
            InvalidProvider
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Question parse provider must "
            "return a QuestionParse."
        ),
    ):
        parser.parse(
            "Question?"
        )
from dataclasses import dataclass
from typing import Callable, Protocol

@dataclass(frozen=True)
class ParsedToken:
    index: int
    text: str
    lemma: str
    pos: str
    tag: str
    dependency: str
    head_index: int
    is_stop: bool
    is_punct: bool
    is_alpha: bool

@dataclass(frozen=True)
class ParsedSpan:
    text: str
    start_index: int
    end_index: int
    root_index: int

@dataclass(frozen=True)
class QuestionParse:
    tokens: tuple[
        ParsedToken,
        ...
    ]
    noun_phrases: tuple[
        ParsedSpan,
        ...
    ]

    @property
    def roots(
        self,
    ) -> tuple[
        ParsedToken,
        ...
    ]:
        return tuple(
            token
            for token in self.tokens
            if token.dependency == "ROOT"
        )

@dataclass(frozen=True)
class QuestionParseResult:
    primary: QuestionParse
    fallback: QuestionParse | None = None

    @property
    def used_fallback(
        self,
    ) -> bool:
        return self.fallback is not None

class QuestionParserService(
    Protocol
):
    def parse(
        self,
        question: str,
    ) -> QuestionParseResult:
        ...

class QuestionParseProvider(
    Protocol
):
    def parse(
        self,
        question: str,
    ) -> QuestionParse:
        ...

QuestionParseProviderFactory = Callable[
    [],
    QuestionParseProvider,
]

def is_question_parse_suspicious(
    parse: QuestionParse,
) -> bool:
    roots = parse.roots

    if len(roots) != 1:
        return True

    root = roots[0]

    verbs = tuple(
        token
        for token in parse.tokens
        if token.pos == "VERB"
    )

    if root.pos in (
        "NOUN",
        "PROPN",
    ):
        return True

    if (
        root.pos == "AUX"
        and root.tag == "MD"
        and not verbs
    ):
        return True

    return False

class QuestionParser:
    def __init__(
        self,
        primary_provider: QuestionParseProvider,
        fallback_provider_factory: (
            QuestionParseProviderFactory | None
        ) = None,
    ) -> None:
        self._primary_provider = (
            primary_provider
        )

        self._fallback_provider_factory = (
            fallback_provider_factory
        )

        self._fallback_provider: (
            QuestionParseProvider | None
        ) = None

    def parse(
        self,
        question: str,
    ) -> QuestionParseResult:
        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        primary = (
            self._primary_provider.parse(
                question
            )
        )

        if not isinstance(
            primary,
            QuestionParse,
        ):
            raise ValueError(
                "Question parse provider must "
                "return a QuestionParse."
            )

        if (
            not is_question_parse_suspicious(
                primary
            )
            or self._fallback_provider_factory
            is None
        ):
            return QuestionParseResult(
                primary=primary,
            )

        fallback_provider = (
            self._get_fallback_provider()
        )

        fallback = fallback_provider.parse(
            question
        )

        if not isinstance(
            fallback,
            QuestionParse,
        ):
            raise ValueError(
                "Question parse provider must "
                "return a QuestionParse."
            )

        return QuestionParseResult(
            primary=primary,
            fallback=fallback,
        )

    def _get_fallback_provider(
        self,
    ) -> QuestionParseProvider:
        if self._fallback_provider is None:
            if (
                self._fallback_provider_factory
                is None
            ):
                raise RuntimeError(
                    "Fallback provider factory "
                    "is not configured."
                )

            self._fallback_provider = (
                self._fallback_provider_factory()
            )

        return self._fallback_provider
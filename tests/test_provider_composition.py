import pytest
from backend.core.provider_composition import (
    ProviderFactories,
    compose_answerability_provider,
    compose_embedding_provider,
    compose_question_parser,
    compose_reranker_provider,
    compose_runtime_providers,
    compose_language_model_provider,
)
from backend.core.runtime_config import (
    ProviderModelConfig,
    QuestionParserProviderConfig,
    RetrievalProviderConfig,
    RuntimeProviderConfig,
)
from backend.routing.question_parser import (
    ParsedToken,
    QuestionParse,
)
from backend.routing.answerability import (
    AnswerabilityResult,
)

class FakeEmbeddingProvider:
    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[
        tuple[float, ...],
        ...
    ]:
        return tuple(
            (1.0,)
            for _ in texts
        )

    def embed_query(
        self,
        text: str,
    ) -> tuple[float, ...]:
        return (1.0,)

class FakeRerankerProvider:
    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name

    def score(
        self,
        query: str,
        documents: tuple[str, ...],
    ) -> tuple[float, ...]:
        return tuple(
            1.0
            for _ in documents
        )

class FakeAnswerabilityProvider:
    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name

    def score(
        self,
        question: str,
        evidence_texts: tuple[
            str,
            ...
        ],
    ) -> AnswerabilityResult:
        return AnswerabilityResult(
            scores=tuple(
                0.5
                for _ in evidence_texts
            )
        )

class FakeLanguageModelProvider:
    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return "Generated answer."

class FakeQuestionParseProvider:
    def __init__(
        self,
        model_name: str,
        parse: QuestionParse,
    ) -> None:
        self.model_name = model_name
        self.result = parse
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
                text="apply",
                pos="VERB",
                dependency="ROOT",
                head_index=0,
            ),
        ),
        noun_phrases=(),
    )

def _suspicious_parse(
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

def _runtime_config(
    *,
    embedding_provider: str = "embedding-a",
    embedding_model: str = "embedding-model-a",
    reranker_provider: str = "reranker-a",
    reranker_model: str = "reranker-model-a",
    parser_provider: str = "parser-a",
    parser_model: str = "parser-model-a",
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
    answerability_provider: str = (
        "answerability-a"
    ),
    answerability_model: str = (
        "answerability-model-a"
    ),
    generation_provider: str = (
        "language-model-a"
    ),
    generation_model: str = (
        "generation-model-a"
    ),
) -> RuntimeProviderConfig:
    fallback = None

    if (
        fallback_provider is not None
        and fallback_model is not None
    ):
        fallback = ProviderModelConfig(
            provider=fallback_provider,
            model=fallback_model,
        )

    return RuntimeProviderConfig(
        retrieval=RetrievalProviderConfig(
            embedding=ProviderModelConfig(
                provider=embedding_provider,
                model=embedding_model,
            ),
            reranker=ProviderModelConfig(
                provider=reranker_provider,
                model=reranker_model,
            ),
        ),
        question_parser=(
            QuestionParserProviderConfig(
                primary=ProviderModelConfig(
                    provider=parser_provider,
                    model=parser_model,
                ),
                fallback=fallback,
            )
        ),
        answerability=ProviderModelConfig(
            provider=answerability_provider,
            model=answerability_model,
        ),
        generation=ProviderModelConfig(
            provider=generation_provider,
            model=generation_model,
        ),
    )

def test_composition_passes_configured_models_to_factories():
    created: list[
        tuple[str, str]
    ] = []

    def create_embedding(
        model: str,
    ) -> FakeEmbeddingProvider:
        created.append(
            (
                "embedding",
                model,
            )
        )

        return FakeEmbeddingProvider(
            model
        )

    def create_reranker(
        model: str,
    ) -> FakeRerankerProvider:
        created.append(
            (
                "reranker",
                model,
            )
        )

        return FakeRerankerProvider(
            model
        )

    def create_parser(
        model: str,
    ) -> FakeQuestionParseProvider:
        created.append(
            (
                "parser",
                model,
            )
        )

        return FakeQuestionParseProvider(
            model,
            _usable_parse(),
        )

    def create_answerability(
        model: str,
    ) -> FakeAnswerabilityProvider:
        created.append(
            (
                "answerability",
                model,
            )
        )

        return FakeAnswerabilityProvider(
            model
        )

    compose_runtime_providers(
        _runtime_config(),
        ProviderFactories(
            embedding={
                "embedding-a": (
                    create_embedding
                ),
            },
            reranker={
                "reranker-a": (
                    create_reranker
                ),
            },
            question_parser={
                "parser-a": (
                    create_parser
                ),
            },
            answerability={
                "answerability-a": (
                    create_answerability
                ),
            },
        ),
    )

    assert created == [
        (
            "embedding",
            "embedding-model-a",
        ),
        (
            "reranker",
            "reranker-model-a",
        ),
        (
            "parser",
            "parser-model-a",
        ),
        (
            "answerability",
            "answerability-model-a",
        ),
    ]

def test_configuration_can_select_different_provider_factories():
    selected: list[
        tuple[str, str]
    ] = []

    def create_embedding_a(
        model: str,
    ) -> FakeEmbeddingProvider:
        selected.append(
            (
                "embedding-a",
                model,
            )
        )

        return FakeEmbeddingProvider(
            model
        )

    def create_embedding_b(
        model: str,
    ) -> FakeEmbeddingProvider:
        selected.append(
            (
                "embedding-b",
                model,
            )
        )

        return FakeEmbeddingProvider(
            model
        )

    factories = ProviderFactories(
        embedding={
            "embedding-a": create_embedding_a,
            "embedding-b": create_embedding_b,
        },
        reranker={
            "reranker-a": (
                FakeRerankerProvider
            ),
        },
        question_parser={
            "parser-a": lambda model: (
                FakeQuestionParseProvider(
                    model,
                    _usable_parse(),
                )
            ),
        },
        answerability={
            "answerability-a": (
                FakeAnswerabilityProvider
            ),
        },
    )

    compose_runtime_providers(
        _runtime_config(
            embedding_provider=(
                "embedding-b"
            ),
            embedding_model=(
                "replacement-model"
            ),
        ),
        factories,
    )

    assert selected == [
        (
            "embedding-b",
            "replacement-model",
        )
    ]

def test_fallback_parser_is_not_created_during_composition():
    fallback_factory_calls = 0

    def create_fallback(
        model: str,
    ) -> FakeQuestionParseProvider:
        nonlocal fallback_factory_calls

        fallback_factory_calls += 1

        return FakeQuestionParseProvider(
            model,
            _usable_parse(),
        )

    providers = compose_runtime_providers(
        _runtime_config(
            fallback_provider="parser-b",
            fallback_model="fallback-model",
        ),
        ProviderFactories(
            embedding={
                "embedding-a": (
                    FakeEmbeddingProvider
                ),
            },
            reranker={
                "reranker-a": (
                    FakeRerankerProvider
                ),
            },
            question_parser={
                "parser-a": lambda model: (
                    FakeQuestionParseProvider(
                        model,
                        _usable_parse(),
                    )
                ),
                "parser-b": create_fallback,
            },
            answerability={
                "answerability-a": (
                    FakeAnswerabilityProvider
                ),
            },
        ),
    )

    assert fallback_factory_calls == 0

    result = (
        providers.question_parser.parse(
            "Can students apply?"
        )
    )

    assert not result.used_fallback
    assert fallback_factory_calls == 0

def test_fallback_parser_is_created_only_when_required():
    fallback_factory_calls = 0
    fallback_models: list[str] = []

    def create_fallback(
        model: str,
    ) -> FakeQuestionParseProvider:
        nonlocal fallback_factory_calls

        fallback_factory_calls += 1
        fallback_models.append(
            model
        )

        return FakeQuestionParseProvider(
            model,
            _usable_parse(),
        )

    providers = compose_runtime_providers(
        _runtime_config(
            fallback_provider="parser-b",
            fallback_model="fallback-model",
        ),
        ProviderFactories(
            embedding={
                "embedding-a": (
                    FakeEmbeddingProvider
                ),
            },
            reranker={
                "reranker-a": (
                    FakeRerankerProvider
                ),
            },
            question_parser={
                "parser-a": lambda model: (
                    FakeQuestionParseProvider(
                        model,
                        _suspicious_parse(),
                    )
                ),
                "parser-b": create_fallback,
            },
            answerability={
                "answerability-a": (
                    FakeAnswerabilityProvider
                ),
            },
        ),
    )

    assert fallback_factory_calls == 0

    first = providers.question_parser.parse(
        "First question?"
    )

    second = providers.question_parser.parse(
        "Second question?"
    )

    assert first.used_fallback
    assert second.used_fallback

    assert fallback_factory_calls == 1

    assert fallback_models == [
        "fallback-model"
    ]

def test_unknown_embedding_provider_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Unknown embedding provider: "
            "missing."
        ),
    ):
        compose_runtime_providers(
            _runtime_config(
                embedding_provider="missing",
            ),
            ProviderFactories(
                embedding={},
                reranker={
                    "reranker-a": (
                        FakeRerankerProvider
                    ),
                },
                question_parser={
                    "parser-a": lambda model: (
                        FakeQuestionParseProvider(
                            model,
                            _usable_parse(),
                        )
                    ),
                },
                answerability={
                    "answerability-a": (
                        FakeAnswerabilityProvider
                    ),
                },
            ),
        )

def test_unknown_reranker_provider_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Unknown reranker provider: "
            "missing."
        ),
    ):
        compose_runtime_providers(
            _runtime_config(
                reranker_provider="missing",
            ),
            ProviderFactories(
                embedding={
                    "embedding-a": (
                        FakeEmbeddingProvider
                    ),
                },
                reranker={},
                question_parser={
                    "parser-a": lambda model: (
                        FakeQuestionParseProvider(
                            model,
                            _usable_parse(),
                        )
                    ),
                },
                answerability={
                    "answerability-a": (
                        FakeAnswerabilityProvider
                    ),
                },
            ),
        )

def test_unknown_primary_parser_provider_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Unknown question parser "
            "provider: missing."
        ),
    ):
        compose_runtime_providers(
            _runtime_config(
                parser_provider="missing",
            ),
            ProviderFactories(
                embedding={
                    "embedding-a": (
                        FakeEmbeddingProvider
                    ),
                },
                reranker={
                    "reranker-a": (
                        FakeRerankerProvider
                    ),
                },
                question_parser={},
                answerability={
                    "answerability-a": (
                        FakeAnswerabilityProvider
                    ),
                },
            ),
        )

def test_unknown_fallback_provider_is_lazy_failure():
    providers = compose_runtime_providers(
        _runtime_config(
            fallback_provider="missing",
            fallback_model="fallback-model",
        ),
        ProviderFactories(
            embedding={
                "embedding-a": (
                    FakeEmbeddingProvider
                ),
            },
            reranker={
                "reranker-a": (
                    FakeRerankerProvider
                ),
            },
            question_parser={
                "parser-a": lambda model: (
                    FakeQuestionParseProvider(
                        model,
                        _suspicious_parse(),
                    )
                ),
            },
            answerability={
                "answerability-a": (
                    FakeAnswerabilityProvider
                ),
            },
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unknown question parser "
            "provider: missing."
        ),
    ):
        providers.question_parser.parse(
            "Question?"
        )

def test_compose_embedding_provider_constructs_only_embedding():
    created: list[
        tuple[str, str]
    ] = []

    def create_embedding(
        model: str,
    ) -> FakeEmbeddingProvider:
        created.append(
            (
                "embedding",
                model,
            )
        )

        return FakeEmbeddingProvider(
            model
        )

    provider = compose_embedding_provider(
        ProviderModelConfig(
            provider="embedding-a",
            model="embedding-model",
        ),
        ProviderFactories(
            embedding={
                "embedding-a": create_embedding,
            },
            reranker={},
            question_parser={},
            answerability={
                "answerability-a": (
                    FakeAnswerabilityProvider
                ),
            },
        ),
    )

    assert created == [
        (
            "embedding",
            "embedding-model",
        )
    ]

    assert isinstance(
        provider,
        FakeEmbeddingProvider,
    )

def test_compose_reranker_provider_constructs_only_reranker():
    created: list[
        tuple[str, str]
    ] = []

    def create_reranker(
        model: str,
    ) -> FakeRerankerProvider:
        created.append(
            (
                "reranker",
                model,
            )
        )

        return FakeRerankerProvider(
            model
        )

    provider = compose_reranker_provider(
        ProviderModelConfig(
            provider="reranker-a",
            model="reranker-model",
        ),
        ProviderFactories(
            embedding={},
            reranker={
                "reranker-a": create_reranker,
            },
            question_parser={},
            answerability={
                "answerability-a": (
                    FakeAnswerabilityProvider
                ),
            },
        ),
    )

    assert created == [
        (
            "reranker",
            "reranker-model",
        )
    ]

    assert isinstance(
        provider,
        FakeRerankerProvider,
    )

def test_compose_question_parser_keeps_fallback_lazy():
    fallback_calls = 0

    def create_fallback(
        model: str,
    ) -> FakeQuestionParseProvider:
        nonlocal fallback_calls

        fallback_calls += 1

        return FakeQuestionParseProvider(
            model,
            _usable_parse(),
        )

    parser = compose_question_parser(
        QuestionParserProviderConfig(
            primary=ProviderModelConfig(
                provider="parser-a",
                model="primary-model",
            ),
            fallback=ProviderModelConfig(
                provider="parser-b",
                model="fallback-model",
            ),
        ),
        ProviderFactories(
            embedding={},
            reranker={},
            question_parser={
                "parser-a": lambda model: (
                    FakeQuestionParseProvider(
                        model,
                        _usable_parse(),
                    )
                ),
                "parser-b": create_fallback,
            },
            answerability={
                "answerability-a": (
                    FakeAnswerabilityProvider
                ),
            },
        ),
    )

    assert fallback_calls == 0

    result = parser.parse(
        "Can students apply?"
    )

    assert not result.used_fallback
    assert fallback_calls == 0

def test_compose_answerability_provider_constructs_only_answerability():
    created: list[
        tuple[str, str]
    ] = []

    def create_answerability(
        model: str,
    ) -> FakeAnswerabilityProvider:
        created.append(
            (
                "answerability",
                model,
            )
        )

        return FakeAnswerabilityProvider(
            model
        )

    provider = compose_answerability_provider(
        ProviderModelConfig(
            provider="answerability-a",
            model="answerability-model",
        ),
        ProviderFactories(
            embedding={},
            reranker={},
            question_parser={},
            answerability={
                "answerability-a": (
                    create_answerability
                ),
            },
        ),
    )

    assert created == [
        (
            "answerability",
            "answerability-model",
        )
    ]

    assert isinstance(
        provider,
        FakeAnswerabilityProvider,
    )

def test_unknown_answerability_provider_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Unknown answerability provider: "
            "missing."
        ),
    ):
        compose_answerability_provider(
            ProviderModelConfig(
                provider="missing",
                model="answerability-model",
            ),
            ProviderFactories(
                embedding={},
                reranker={},
                question_parser={},
                answerability={},
            ),
        )

def test_unknown_runtime_answerability_provider_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Unknown answerability provider: "
            "missing."
        ),
    ):
        compose_runtime_providers(
            _runtime_config(
                answerability_provider="missing",
            ),
            ProviderFactories(
                embedding={
                    "embedding-a": (
                        FakeEmbeddingProvider
                    ),
                },
                reranker={
                    "reranker-a": (
                        FakeRerankerProvider
                    ),
                },
                question_parser={
                    "parser-a": lambda model: (
                        FakeQuestionParseProvider(
                            model,
                            _usable_parse(),
                        )
                    ),
                },
                answerability={},
            ),
        )

def test_compose_language_model_provider_constructs_only_language_model():
    created: list[
        tuple[str, str]
    ] = []

    def create_language_model(
        model: str,
    ) -> FakeLanguageModelProvider:
        created.append(
            (
                "language-model",
                model,
            )
        )

        return FakeLanguageModelProvider(
            model
        )

    provider = compose_language_model_provider(
        ProviderModelConfig(
            provider="language-model-a",
            model="generation-model",
        ),
        ProviderFactories(
            embedding={},
            reranker={},
            question_parser={},
            answerability={},
            language_model={
                "language-model-a": (
                    create_language_model
                ),
            },
        ),
    )

    assert created == [
        (
            "language-model",
            "generation-model",
        )
    ]

    assert isinstance(
        provider,
        FakeLanguageModelProvider,
    )

def test_unknown_language_model_provider_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Unknown language model provider: "
            "missing."
        ),
    ):
        compose_language_model_provider(
            ProviderModelConfig(
                provider="missing",
                model="generation-model",
            ),
            ProviderFactories(
                embedding={},
                reranker={},
                question_parser={},
                answerability={},
                language_model={},
            ),
        )

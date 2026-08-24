from backend.core.provider_registry import (
    CROSS_ENCODER_PROVIDER,
    SENTENCE_TRANSFORMER_PROVIDER,
    SPACY_PROVIDER,
    create_provider_factories,
)
from backend.retrieval.embeddings import (
    EmbeddingProvider,
)
from backend.retrieval.reranking import (
    RerankerProvider,
)
from backend.routing.question_parser import (
    QuestionParse,
    QuestionParseProvider,
)

class FakeEmbeddingProvider:
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
    def score(
        self,
        query: str,
        documents: tuple[str, ...],
    ) -> tuple[float, ...]:
        return tuple(
            1.0
            for _ in documents
        )

class FakeQuestionParseProvider:
    def parse(
        self,
        question: str,
    ) -> QuestionParse:
        return QuestionParse(
            tokens=(),
            noun_phrases=(),
        )

def test_registry_exposes_expected_provider_ids():
    factories = create_provider_factories()

    assert tuple(
        factories.embedding
    ) == (
        SENTENCE_TRANSFORMER_PROVIDER,
    )

    assert tuple(
        factories.reranker
    ) == (
        CROSS_ENCODER_PROVIDER,
    )

    assert tuple(
        factories.question_parser
    ) == (
        SPACY_PROVIDER,
    )

    assert tuple(
        factories.answerability
    ) == ()

def test_embedding_factory_passes_model_name(
    monkeypatch,
):
    received_models: list[str] = []

    fake_provider = (
        FakeEmbeddingProvider()
    )

    def fake_create_embedding(
        model_name: str,
    ) -> EmbeddingProvider:
        received_models.append(
            model_name
        )

        return fake_provider

    monkeypatch.setattr(
        (
            "backend.core.provider_registry."
            "_create_sentence_transformer_provider"
        ),
        fake_create_embedding,
    )

    factories = create_provider_factories()

    provider = (
        factories.embedding[
            SENTENCE_TRANSFORMER_PROVIDER
        ](
            "replacement-embedding-model"
        )
    )

    assert received_models == [
        "replacement-embedding-model"
    ]

    assert provider is fake_provider

def test_reranker_factory_passes_model_name(
    monkeypatch,
):
    received_models: list[str] = []

    fake_provider = (
        FakeRerankerProvider()
    )

    def fake_create_reranker(
        model_name: str,
    ) -> RerankerProvider:
        received_models.append(
            model_name
        )

        return fake_provider

    monkeypatch.setattr(
        (
            "backend.core.provider_registry."
            "_create_cross_encoder_provider"
        ),
        fake_create_reranker,
    )

    factories = create_provider_factories()

    provider = (
        factories.reranker[
            CROSS_ENCODER_PROVIDER
        ](
            "replacement-reranker-model"
        )
    )

    assert received_models == [
        "replacement-reranker-model"
    ]

    assert provider is fake_provider

def test_parser_factory_passes_model_name(
    monkeypatch,
):
    received_models: list[str] = []

    fake_provider = (
        FakeQuestionParseProvider()
    )

    def fake_create_parser(
        model_name: str,
    ) -> QuestionParseProvider:
        received_models.append(
            model_name
        )

        return fake_provider

    monkeypatch.setattr(
        (
            "backend.core.provider_registry."
            "_create_spacy_provider"
        ),
        fake_create_parser,
    )

    factories = create_provider_factories()

    provider = (
        factories.question_parser[
            SPACY_PROVIDER
        ](
            "replacement-parser-model"
        )
    )

    assert received_models == [
        "replacement-parser-model"
    ]

    assert provider is fake_provider

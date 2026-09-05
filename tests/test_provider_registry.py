from backend.core.provider_registry import (
    CROSS_ENCODER_PROVIDER,
    SENTENCE_TRANSFORMER_PROVIDER,
    SPACY_PROVIDER,
    create_provider_factories,
    CROSS_ENCODER_ANSWERABILITY_PROVIDER,
    OLLAMA_PROVIDER,
    CROSS_ENCODER_ENTAILMENT_PROVIDER,
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
from backend.routing.answerability import (
    AnswerabilityResult,
    AnswerabilityProvider,
)
from backend.llm.provider import (
    LanguageModelProvider,
)
from backend.generation.entailment import (
    EntailmentPair,
    EntailmentProvider,
)

class FakeLanguageModelProvider:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return "Generated answer."

class FakeAnswerabilityProvider:
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

class FakeEntailmentProvider:
    def score_entailment(
        self,
        pairs: tuple[
            EntailmentPair,
            ...
        ],
    ) -> tuple[float, ...]:
        return tuple(
            0.5
            for _ in pairs
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
    ) == (
        CROSS_ENCODER_ANSWERABILITY_PROVIDER,
    )

    assert tuple(
        factories.entailment
    ) == (
        CROSS_ENCODER_ENTAILMENT_PROVIDER,
    )

    assert tuple(
        factories.language_model
    ) == (
        OLLAMA_PROVIDER,
    )

def test_answerability_factory_passes_model_name(
    monkeypatch,
):
    received_models: list[str] = []

    fake_provider = (
        FakeAnswerabilityProvider()
    )

    def fake_create_answerability(
        model_name: str,
    ) -> AnswerabilityProvider:
        received_models.append(
            model_name
        )

        return fake_provider

    monkeypatch.setattr(
        (
            "backend.core.provider_registry."
            "_create_cross_encoder_answerability_provider"
        ),
        fake_create_answerability,
    )

    factories = create_provider_factories()

    provider = (
        factories.answerability[
            CROSS_ENCODER_ANSWERABILITY_PROVIDER
        ](
            "replacement-answerability-model"
        )
    )

    assert received_models == [
        "replacement-answerability-model"
    ]

    assert provider is fake_provider

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

def test_ollama_factory_passes_model_name(
    monkeypatch,
):
    received_models: list[str] = []

    fake_provider = (
        FakeLanguageModelProvider()
    )

    def fake_create_ollama(
        model_name: str,
    ) -> LanguageModelProvider:
        received_models.append(
            model_name
        )

        return fake_provider

    monkeypatch.setattr(
        (
            "backend.core.provider_registry."
            "_create_ollama_provider"
        ),
        fake_create_ollama,
    )

    factories = create_provider_factories()

    provider = (
        factories.language_model[
            OLLAMA_PROVIDER
        ](
            "replacement-generation-model"
        )
    )

    assert received_models == [
        "replacement-generation-model"
    ]

    assert provider is fake_provider

def test_entailment_factory_passes_model_name(
    monkeypatch,
):
    received_models: list[str] = []

    fake_provider = (
        FakeEntailmentProvider()
    )

    def fake_create_entailment(
        model_name: str,
    ) -> EntailmentProvider:
        received_models.append(
            model_name
        )

        return fake_provider

    monkeypatch.setattr(
        (
            "backend.core.provider_registry."
            "_create_cross_encoder_entailment_provider"
        ),
        fake_create_entailment,
    )

    factories = create_provider_factories()

    provider = (
        factories.entailment[
            CROSS_ENCODER_ENTAILMENT_PROVIDER
        ](
            "replacement-entailment-model"
        )
    )

    assert received_models == [
        "replacement-entailment-model"
    ]

    assert provider is fake_provider

from backend.core.provider_composition import (
    ProviderFactories,
)
from backend.retrieval.embeddings import (
    EmbeddingProvider,
)
from backend.retrieval.reranking import (
    RerankerProvider,
)
from backend.routing.question_parser import (
    QuestionParseProvider,
)
from backend.routing.answerability import (
    AnswerabilityProvider,
)
from backend.llm.provider import (
    LanguageModelProvider,
)

SENTENCE_TRANSFORMER_PROVIDER = (
    "sentence_transformer"
)
CROSS_ENCODER_PROVIDER = (
    "cross_encoder"
)
SPACY_PROVIDER = "spacy"
CROSS_ENCODER_ANSWERABILITY_PROVIDER = (
    "cross_encoder_answerability"
)
OLLAMA_PROVIDER = "ollama"

def _create_cross_encoder_answerability_provider(
    model_name: str,
) -> AnswerabilityProvider:
    from backend.routing.cross_encoder_answerability_provider import (
        CrossEncoderAnswerabilityProvider,
    )

    return CrossEncoderAnswerabilityProvider(
        model_name=model_name,
    )

def _create_sentence_transformer_provider(
    model_name: str,
) -> EmbeddingProvider:
    from backend.retrieval.sentence_transformer_provider import (
        SentenceTransformerEmbeddingProvider,
    )

    return SentenceTransformerEmbeddingProvider(
        model_name=model_name,
    )

def _create_cross_encoder_provider(
    model_name: str,
) -> RerankerProvider:
    from backend.retrieval.cross_encoder_provider import (
        CrossEncoderRerankerProvider,
    )

    return CrossEncoderRerankerProvider(
        model_name=model_name,
    )

def _create_spacy_provider(
    model_name: str,
) -> QuestionParseProvider:
    from backend.routing.spacy_question_parser import (
        load_spacy_question_parse_provider,
    )

    return load_spacy_question_parse_provider(
        model_name
    )

def _create_ollama_provider(
    model_name: str,
) -> LanguageModelProvider:
    from backend.llm.ollama_provider import (
        OllamaLanguageModelProvider,
    )

    return OllamaLanguageModelProvider(
        model_name=model_name,
    )

def create_provider_factories(
) -> ProviderFactories:
    return ProviderFactories(
        embedding={
            SENTENCE_TRANSFORMER_PROVIDER: (
                _create_sentence_transformer_provider
            ),
        },
        reranker={
            CROSS_ENCODER_PROVIDER: (
                _create_cross_encoder_provider
            ),
        },
        question_parser={
            SPACY_PROVIDER: (
                _create_spacy_provider
            ),
        },
        answerability={
            CROSS_ENCODER_ANSWERABILITY_PROVIDER: (
                _create_cross_encoder_answerability_provider
            ),
        },
        language_model={
            OLLAMA_PROVIDER: (
                _create_ollama_provider
            ),
        },
)

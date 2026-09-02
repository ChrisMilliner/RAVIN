"""
Define framework-neutral runtime provider configuration for RAVIN.

The configuration models in this module identify provider and model
choices without importing the concrete implementation libraries. They
describe the runtime dependencies required for retrieval, question
analysis, evidence assessment, generated-claim validation, and grounded
answer wording.

Keeping configuration neutral allows individual providers and models to
be replaced without changing the business pipeline.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class ProviderModelConfig:
    """Identify one configured provider and model pair.

    The model keeps provider selection serializable and independent of the
    concrete implementation class that will later be constructed.
    """

    provider: str
    model: str

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError(
                "Provider cannot be empty."
            )

        if not self.model.strip():
            raise ValueError(
                "Model cannot be empty."
            )

@dataclass(frozen=True)
class RetrievalProviderConfig:
    """Group the embedding and reranker configuration used by retrieval.

    The configuration describes provider choices only and contains no
    retrieval ranking behaviour itself.
    """

    embedding: ProviderModelConfig
    reranker: ProviderModelConfig

@dataclass(frozen=True)
class QuestionParserProviderConfig:
    """Describe the primary and optional fallback question-parser providers.

    Parser configuration remains independent of the routing logic that
    consumes the resulting parsed question structure.
    """

    primary: ProviderModelConfig
    fallback: ProviderModelConfig | None = None

@dataclass(frozen=True)
class RuntimeProviderConfig:
    """Aggregate all provider and model selections required by RAVIN.

    The configuration forms the provider-neutral runtime description used
    during application startup to compose retrieval, parsing,
    answerability, entailment, and generation dependencies.
    """

    retrieval: RetrievalProviderConfig
    question_parser: QuestionParserProviderConfig
    answerability: ProviderModelConfig
    entailment: ProviderModelConfig
    generation: ProviderModelConfig

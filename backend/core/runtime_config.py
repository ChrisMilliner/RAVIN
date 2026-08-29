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
    embedding: ProviderModelConfig
    reranker: ProviderModelConfig

@dataclass(frozen=True)
class QuestionParserProviderConfig:
    primary: ProviderModelConfig
    fallback: ProviderModelConfig | None = None

@dataclass(frozen=True)
class RuntimeProviderConfig:
    retrieval: RetrievalProviderConfig
    question_parser: QuestionParserProviderConfig
    answerability: ProviderModelConfig
    entailment: ProviderModelConfig
    generation: ProviderModelConfig

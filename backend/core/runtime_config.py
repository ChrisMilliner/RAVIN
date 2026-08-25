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

from types import SimpleNamespace
from typing import cast
import pytest
import backend.service.composition as composition_module
from backend.core.answer_quality_config import (
    AnswerQualityConfig,
    DEVELOPMENT_NOT_VALIDATED_STATUS,
)
from backend.core.provider_composition import (
    ProviderFactories,
)
from backend.core.runtime_config import (
    RuntimeProviderConfig,
)
from backend.ingestion.models import (
    PolicyChunk,
)
from backend.service.answer_service import (
    RavinAnswerService,
)
from backend.service.composition import (
    create_ravin_answer_service,
)

def _quality_config(
) -> AnswerQualityConfig:
    return AnswerQualityConfig(
        schema_version=1,
        status=(
            DEVELOPMENT_NOT_VALIDATED_STATUS
        ),
        proposition_covered_threshold=0.8,
        proposition_partial_threshold=0.4,
        claim_support_threshold=0.8,
    )

def _runtime_config(
) -> RuntimeProviderConfig:
    return cast(
        RuntimeProviderConfig,
        object(),
    )

def _provider_factories(
) -> ProviderFactories:
    return cast(
        ProviderFactories,
        object(),
    )

def _policy_chunks(
) -> tuple[
    PolicyChunk,
    ...
]:
    return (
        cast(
            PolicyChunk,
            object(),
        ),
    )

def _fake_composed_providers():
    return SimpleNamespace(
        embedding=object(),
        reranker=object(),
        question_parser=object(),
        answerability=object(),
        language_model=object(),
    )

def test_empty_policy_corpus_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "At least one policy chunk "
            "is required"
        ),
    ):
        create_ravin_answer_service(
            ()
        )

def test_service_composition_builds_index_once(
    monkeypatch,
):
    build_calls = []

    providers = (
        _fake_composed_providers()
    )

    monkeypatch.setattr(
        composition_module,
        "compose_runtime_providers",
        lambda config, factories: (
            providers
        ),
    )

    def fake_build_index(
        chunks,
        embedding_provider,
    ):
        build_calls.append(
            (
                chunks,
                embedding_provider,
            )
        )

        return (
            object(),
        )

    monkeypatch.setattr(
        composition_module,
        "build_production_retrieval_index",
        fake_build_index,
    )

    chunks = _policy_chunks()

    service = create_ravin_answer_service(
        chunks,
        runtime_config=_runtime_config(),
        provider_factories=(
            _provider_factories()
        ),
        answer_quality_config=(
            _quality_config()
        ),
    )

    assert isinstance(
        service,
        RavinAnswerService,
    )

    assert len(
        build_calls
    ) == 1

    assert (
        build_calls[0][0]
        is chunks
    )

    assert (
        build_calls[0][1]
        is providers.embedding
    )

def test_quality_thresholds_are_used_in_composition(
    monkeypatch,
):
    captured = {}

    providers = (
        _fake_composed_providers()
    )

    monkeypatch.setattr(
        composition_module,
        "compose_runtime_providers",
        lambda config, factories: (
            providers
        ),
    )

    monkeypatch.setattr(
        composition_module,
        "build_production_retrieval_index",
        lambda chunks, embedding_provider: (
            object(),
        ),
    )

    def fake_coverage_assessor(
        answerability_provider,
        covered_threshold,
        partial_threshold,
    ):
        captured[
            "covered_threshold"
        ] = covered_threshold

        captured[
            "partial_threshold"
        ] = partial_threshold

        return object()

    def fake_claim_validator(
        answerability_provider,
        support_threshold,
    ):
        captured[
            "claim_support_threshold"
        ] = support_threshold

        return object()

    monkeypatch.setattr(
        composition_module,
        "PropositionCoverageAssessor",
        fake_coverage_assessor,
    )

    monkeypatch.setattr(
        composition_module,
        "GeneratedClaimGroundingValidator",
        fake_claim_validator,
    )

    quality_config = (
        AnswerQualityConfig(
            schema_version=1,
            status=(
                DEVELOPMENT_NOT_VALIDATED_STATUS
            ),
            proposition_covered_threshold=0.86,
            proposition_partial_threshold=0.47,
            claim_support_threshold=0.84,
        )
    )

    service = create_ravin_answer_service(
        _policy_chunks(),
        runtime_config=_runtime_config(),
        provider_factories=(
            _provider_factories()
        ),
        answer_quality_config=(
            quality_config
        ),
    )

    assert isinstance(
        service,
        RavinAnswerService,
    )

    assert (
        captured["covered_threshold"]
        == 0.86
    )

    assert (
        captured["partial_threshold"]
        == 0.47
    )

    assert (
        captured[
            "claim_support_threshold"
        ]
        == 0.84
    )
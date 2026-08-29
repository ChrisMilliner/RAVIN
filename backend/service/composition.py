from backend.core.answer_quality_config import (
    AnswerQualityConfig,
    load_answer_quality_config,
)
from backend.core.provider_composition import (
    ProviderFactories,
    compose_runtime_providers,
)
from backend.core.provider_registry import (
    create_provider_factories,
)
from backend.core.runtime_config import (
    RuntimeProviderConfig,
)
from backend.core.runtime_config_loader import (
    load_runtime_provider_config,
)
from backend.generation.claim_grounding_validator import (
    GeneratedClaimGroundingValidator,
)
from backend.generation.llm_grounded_generator import (
    LlmGroundedAnswerGenerator,
)
from backend.ingestion.models import (
    PolicyChunk,
)
from backend.retrieval.context import (
    ContextAssemblyConfig,
)
from backend.retrieval.production import (
    GroundedRetrievalResult,
    ProductionRetrievalConfig,
    build_production_retrieval_index,
    retrieve_grounded_context,
)
from backend.routing.dependency_material_proposition_extractor import (
    DependencyMaterialPropositionExtractor,
)
from backend.routing.dependency_material_requirement_extractor import (
    DependencyMaterialRequirementExtractor,
)
from backend.routing.deterministic_evidence_assessor import (
    DeterministicEvidenceSufficiencyAssessor,
)
from backend.routing.deterministic_question_structure_recovery import (
    DeterministicQuestionStructureRecoveryProvider,
)
from backend.routing.proposition_coverage_assessor import (
    PropositionCoverageAssessor,
)
from backend.routing.question_structure_resolver import (
    QuestionStructureResolver,
)
from backend.routing.rule_intent_classifier import (
    RuleBasedQuestionIntentClassifier,
)
from backend.service.answer_service import (
    RavinAnswerService,
)

def create_ravin_answer_service(
    chunks: tuple[
        PolicyChunk,
        ...
    ],
    *,
    runtime_config: (
        RuntimeProviderConfig | None
    ) = None,
    provider_factories: (
        ProviderFactories | None
    ) = None,
    answer_quality_config: (
        AnswerQualityConfig | None
    ) = None,
    retrieval_config: (
        ProductionRetrievalConfig | None
    ) = None,
    context_config: (
        ContextAssemblyConfig | None
    ) = None,
) -> RavinAnswerService:
    if not chunks:
        raise ValueError(
            "At least one policy chunk is "
            "required."
        )

    resolved_runtime_config = (
        load_runtime_provider_config()
        if runtime_config is None
        else runtime_config
    )

    resolved_provider_factories = (
        create_provider_factories()
        if provider_factories is None
        else provider_factories
    )

    resolved_answer_quality_config = (
        load_answer_quality_config()
        if answer_quality_config is None
        else answer_quality_config
    )

    resolved_retrieval_config = (
        ProductionRetrievalConfig()
        if retrieval_config is None
        else retrieval_config
    )

    resolved_context_config = (
        ContextAssemblyConfig()
        if context_config is None
        else context_config
    )

    providers = compose_runtime_providers(
        resolved_runtime_config,
        resolved_provider_factories,
    )

    indexed_chunks = (
        build_production_retrieval_index(
            chunks,
            providers.embedding,
        )
    )

    recovery_provider = (
        DeterministicQuestionStructureRecoveryProvider()
    )

    structure_resolver = (
        QuestionStructureResolver(
            parser=providers.question_parser,
            recovery_provider=recovery_provider,
        )
    )

    requirement_extractor = (
        DependencyMaterialRequirementExtractor(
            structure_resolver=(
                structure_resolver
            ),
        )
    )

    proposition_extractor = (
        DependencyMaterialPropositionExtractor()
    )

    coverage_assessor = (
        PropositionCoverageAssessor(
            answerability_provider=(
                providers.answerability
            ),
            covered_threshold=(
                resolved_answer_quality_config
                .proposition_covered_threshold
            ),
            partial_threshold=(
                resolved_answer_quality_config
                .proposition_partial_threshold
            ),
        )
    )

    evidence_assessor = (
        DeterministicEvidenceSufficiencyAssessor(
            structure_resolver=(
                structure_resolver
            ),
            requirement_extractor=(
                requirement_extractor
            ),
            proposition_extractor=(
                proposition_extractor
            ),
            coverage_assessor=(
                coverage_assessor
            ),
        )
    )

    intent_classifier = (
        RuleBasedQuestionIntentClassifier()
    )

    grounded_generator = (
        LlmGroundedAnswerGenerator(
            provider=providers.language_model,
        )
    )

    claim_grounding_validator = (
        GeneratedClaimGroundingValidator(
            answerability_provider=(
                providers.answerability
            ),
            support_threshold=(
                resolved_answer_quality_config
                .claim_support_threshold
            ),
        )
    )

    def retrieve(
        question: str,
    ) -> GroundedRetrievalResult:
        return retrieve_grounded_context(
            indexed_chunks=indexed_chunks,
            query=question,
            embedding_provider=(
                providers.embedding
            ),
            reranker_provider=(
                providers.reranker
            ),
            retrieval_config=(
                resolved_retrieval_config
            ),
            context_config=(
                resolved_context_config
            ),
        )

    return RavinAnswerService(
        retriever=retrieve,
        intent_classifier=intent_classifier,
        evidence_assessor=evidence_assessor,
        grounded_generator=grounded_generator,
        claim_grounding_validator=(
            claim_grounding_validator
        ),
    )
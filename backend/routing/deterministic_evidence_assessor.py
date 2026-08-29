"""
Convert proposition coverage into deterministic evidence sufficiency.

The assessor maps proposition-level evidence coverage to RAVIN's
SUFFICIENT, INSUFFICIENT, or UNCERTAIN states. Missing or uncovered
material propositions prevent a question from being treated as fully
supported.

This control decision is deterministic. A language model is not
permitted to override insufficient or uncertain evidence.
"""

from backend.retrieval.production import (
    GroundedRetrievalResult,
)
from backend.routing.dependency_material_requirement_extractor import (
    DependencyMaterialRequirementExtractor,
)
from backend.routing.material_proposition_extractor import (
    MaterialPropositionExtractor,
)
from backend.routing.models import (
    EvidenceAssessment,
    EvidenceSufficiency,
    QuestionIntent,
)
from backend.routing.proposition_coverage import (
    PropositionCoverageStatus,
)
from backend.routing.proposition_coverage_assessor import (
    PropositionCoverageAssessor,
)
from backend.routing.question_structure_resolver import (
    QuestionStructureResolver,
)
from backend.routing.signals import (
    extract_evidence_signals,
)

class DeterministicEvidenceSufficiencyAssessor:
    def __init__(
        self,
        structure_resolver: QuestionStructureResolver,
        requirement_extractor: (
            DependencyMaterialRequirementExtractor
        ),
        proposition_extractor: MaterialPropositionExtractor,
        coverage_assessor: PropositionCoverageAssessor,
    ) -> None:
        self._structure_resolver = (
            structure_resolver
        )

        self._requirement_extractor = (
            requirement_extractor
        )

        self._proposition_extractor = (
            proposition_extractor
        )

        self._coverage_assessor = (
            coverage_assessor
        )

    def assess(
        self,
        question: str,
        intent: QuestionIntent,
        retrieval_result: GroundedRetrievalResult,
    ) -> EvidenceAssessment:
        signals = extract_evidence_signals(
            retrieval_result
        )

        if (
            signals.retrieved_count == 0
            or signals.context_block_count == 0
        ):
            return EvidenceAssessment(
                sufficiency=(
                    EvidenceSufficiency.INSUFFICIENT
                ),
                signals=signals,
                reason=(
                    "No retrieved evidence was "
                    "available for assessment."
                ),
            )

        structure = (
            self._structure_resolver.resolve(
                question
            )
        )

        active_parse = structure.active

        if active_parse is None:
            return EvidenceAssessment(
                sufficiency=(
                    EvidenceSufficiency.UNCERTAIN
                ),
                signals=signals,
                reason=(
                    "Question structure could not "
                    "be resolved reliably."
                ),
            )

        requirement_result = (
            self._requirement_extractor.extract(
                question
            )
        )

        active_requirements = (
            requirement_result.active
        )

        if active_requirements is None:
            return EvidenceAssessment(
                sufficiency=(
                    EvidenceSufficiency.UNCERTAIN
                ),
                signals=signals,
                reason=(
                    "Material question requirements "
                    "could not be resolved reliably."
                ),
            )

        propositions = (
            self._proposition_extractor.extract(
                question,
                active_requirements,
                active_parse,
            )
        )

        evidence_texts = tuple(
            block.text
            for block
            in retrieval_result.context.blocks
        )

        coverage = (
            self._coverage_assessor.assess(
                question,
                propositions,
                evidence_texts,
            )
        )

        statuses = tuple(
            item.status
            for item
            in coverage.propositions
        )

        if (
            PropositionCoverageStatus.UNCOVERED
            in statuses
        ):
            return EvidenceAssessment(
                sufficiency=(
                    EvidenceSufficiency.INSUFFICIENT
                ),
                signals=signals,
                reason=(
                    "At least one material "
                    "proposition is not covered by "
                    "the retrieved evidence."
                ),
            )

        if (
            PropositionCoverageStatus.PARTIAL
            in statuses
        ):
            return EvidenceAssessment(
                sufficiency=(
                    EvidenceSufficiency.UNCERTAIN
                ),
                signals=signals,
                reason=(
                    "At least one material "
                    "proposition is only partially "
                    "covered by the retrieved evidence."
                ),
            )

        return EvidenceAssessment(
            sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            signals=signals,
            reason=(
                "All material propositions are "
                "covered by the retrieved evidence."
            ),
        )
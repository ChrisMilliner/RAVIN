"""
Provide the application-facing orchestration service for RAVIN answers.

RavinAnswerService coordinates the completed evidence-first answer
pipeline for one user question. It combines intent classification,
grounded retrieval, evidence sufficiency, deterministic behaviour
routing, grounded generation, release validation, and source mapping.

Application adapters such as the CLI and FastAPI should call this
service rather than implementing retrieval or control logic themselves.
Generated output fails closed when required grounding validation does
not pass.
"""

from dataclasses import dataclass
from typing import Callable
from backend.behavior import AnswerBehavior
from backend.generation.claim_grounding_validator import (
    GeneratedClaimGroundingValidator,
)
from backend.generation.grounded_generator import (
    GroundedAnswerGenerator,
    GroundedGenerationRequest,
)
from backend.generation.release_gate import (
    GroundedGenerationRejectedError,
    generate_validated_grounded_answer,
)
from backend.retrieval.production import (
    GroundedRetrievalResult,
)
from backend.routing.evidence_assessor import (
    EvidenceSufficiencyAssessor,
)
from backend.routing.intent_classifier import (
    QuestionIntentClassifier,
    classify_question_intent,
)
from backend.routing.models import (
    QuestionIntent,
)
from backend.routing.routing_orchestrator import (
    orchestrate_answer_routing,
)

GroundedRetriever = Callable[
    [str],
    GroundedRetrievalResult,
]

_CLARIFICATION_RESPONSE = (
    "Please clarify what policy topic or "
    "outcome you want information about."
)
_NO_GROUNDED_ANSWER_RESPONSE = (
    "I could not find sufficient evidence "
    "in the available policy sources."
)

@dataclass(frozen=True)
class AnswerSource:
    policy_id: str
    title: str
    heading: str
    url: str

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError(
                "Answer source policy ID "
                "cannot be empty."
            )

        if not self.title.strip():
            raise ValueError(
                "Answer source title "
                "cannot be empty."
            )

        if not self.url.strip():
            raise ValueError(
                "Answer source URL "
                "cannot be empty."
            )

@dataclass(frozen=True)
class IntegratedAnswerResult:
    behavior: AnswerBehavior
    answer: str
    grounded: bool
    sources: tuple[AnswerSource, ...]

    def __post_init__(self) -> None:
        if not self.answer.strip():
            raise ValueError(
                "Integrated answer cannot "
                "be empty."
            )

        answer_behavior = self.behavior in {
            AnswerBehavior.DIRECT_ANSWER,
            AnswerBehavior.GROUNDED_OVERVIEW,
        }

        if answer_behavior and not self.grounded:
            raise ValueError(
                "Answer-producing behavior must "
                "be grounded."
            )

        if answer_behavior and not self.sources:
            raise ValueError(
                "Grounded answer must contain "
                "at least one source."
            )

        if (
            not answer_behavior
            and self.grounded
        ):
            raise ValueError(
                "Non-answer behavior cannot be "
                "marked as grounded."
            )

        if (
            not answer_behavior
            and self.sources
        ):
            raise ValueError(
                "Non-answer behavior cannot "
                "contain sources."
            )

@dataclass(frozen=True)
class _ResolvedIntentClassifier:
    intent: QuestionIntent

    def classify(
        self,
        question: str,
    ) -> QuestionIntent:
        return self.intent

class RavinAnswerService:
    def __init__(
        self,
        retriever: GroundedRetriever,
        intent_classifier: QuestionIntentClassifier,
        evidence_assessor: EvidenceSufficiencyAssessor,
        grounded_generator: GroundedAnswerGenerator,
        claim_grounding_validator: (
            GeneratedClaimGroundingValidator
        ),
    ) -> None:
        self._retriever = retriever
        self._intent_classifier = (
            intent_classifier
        )
        self._evidence_assessor = (
            evidence_assessor
        )
        self._grounded_generator = (
            grounded_generator
        )
        self._claim_grounding_validator = (
            claim_grounding_validator
        )

    def answer(
        self,
        question: str,
    ) -> IntegratedAnswerResult:
        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        intent = classify_question_intent(
            question,
            self._intent_classifier,
        )

        retrieval_result = None

        if intent != QuestionIntent.AMBIGUOUS:
            retrieval_result = self._retriever(
                question
            )

            if not isinstance(
                retrieval_result,
                GroundedRetrievalResult,
            ):
                raise ValueError(
                    "Retriever must return a "
                    "GroundedRetrievalResult."
                )

        routing_result = (
            orchestrate_answer_routing(
                question=question,
                retrieval_result=retrieval_result,
                intent_classifier=(
                    _ResolvedIntentClassifier(
                        intent
                    )
                ),
                evidence_assessor=(
                    self._evidence_assessor
                ),
            )
        )

        if (
            routing_result.behavior
            == AnswerBehavior.CLARIFY
        ):
            return IntegratedAnswerResult(
                behavior=AnswerBehavior.CLARIFY,
                answer=_CLARIFICATION_RESPONSE,
                grounded=False,
                sources=(),
            )

        if (
            routing_result.behavior
            == AnswerBehavior.NO_GROUNDED_ANSWER
        ):
            return _no_grounded_answer_result()

        if retrieval_result is None:
            raise RuntimeError(
                "Grounded answer routing requires "
                "a retrieval result."
            )

        evidence_texts = tuple(
            block.text
            for block
            in retrieval_result.context.blocks
        )

        generation_request = (
            GroundedGenerationRequest(
                question=question,
                behavior=routing_result.behavior,
                evidence_texts=evidence_texts,
            )
        )

        try:
            released_answer = (
                generate_validated_grounded_answer(
                    request=generation_request,
                    generator=(
                        self._grounded_generator
                    ),
                    claim_grounding_validator=(
                        self._claim_grounding_validator
                    ),
                )
            )
        except GroundedGenerationRejectedError:
            return _no_grounded_answer_result()

        sources = _build_answer_sources(
            retrieval_result,
            released_answer.cited_evidence_indexes,
        )

        return IntegratedAnswerResult(
            behavior=routing_result.behavior,
            answer=released_answer.text,
            grounded=True,
            sources=sources,
        )

def _no_grounded_answer_result(
) -> IntegratedAnswerResult:
    return IntegratedAnswerResult(
        behavior=(
            AnswerBehavior.NO_GROUNDED_ANSWER
        ),
        answer=_NO_GROUNDED_ANSWER_RESPONSE,
        grounded=False,
        sources=(),
    )

def _build_answer_sources(
    retrieval_result: GroundedRetrievalResult,
    cited_evidence_indexes: tuple[int, ...],
) -> tuple[AnswerSource, ...]:
    sources = []

    for evidence_index in (
        cited_evidence_indexes
    ):
        block = (
            retrieval_result
            .context
            .blocks[evidence_index - 1]
        )

        heading = " > ".join(
            block.heading_path
        )

        source = AnswerSource(
            policy_id=block.policy_id,
            title=block.policy_title,
            heading=heading,
            url=block.source_url,
        )

        if source not in sources:
            sources.append(
                source
            )

    return tuple(
        sources
    )
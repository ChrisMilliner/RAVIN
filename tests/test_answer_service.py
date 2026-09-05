import pytest
from backend.behavior import (
    AnswerBehavior,
)
from backend.generation.claim_grounding_validator import (
    GeneratedClaimGroundingValidator,
)
from backend.generation.grounded_generator import (
    GroundedGenerationRequest,
    GroundedGenerationResult,
)
from backend.retrieval.context import (
    GroundedContext,
    GroundedContextBlock,
)
from backend.retrieval.production import (
    GroundedRetrievalResult,
)
from backend.routing.models import (
    EvidenceAssessment,
    EvidenceSignals,
    EvidenceSufficiency,
    QuestionIntent,
)
from backend.service.answer_service import (
    RavinAnswerService,
)
from backend.generation.entailment import (
    EntailmentPair,
)

class FakeIntentClassifier:
    def __init__(
        self,
        intent: QuestionIntent,
    ) -> None:
        self._intent = intent
        self.call_count = 0

    def classify(
        self,
        question: str,
    ) -> QuestionIntent:
        self.call_count += 1
        return self._intent

class RecordingRetriever:
    def __init__(
        self,
        result: GroundedRetrievalResult,
    ) -> None:
        self._result = result
        self.call_count = 0

    def __call__(
        self,
        question: str,
    ) -> GroundedRetrievalResult:
        self.call_count += 1
        return self._result

class FakeEvidenceAssessor:
    def __init__(
        self,
        sufficiency: EvidenceSufficiency,
    ) -> None:
        self._sufficiency = sufficiency
        self.call_count = 0

    def assess(
        self,
        question,
        intent,
        retrieval_result,
    ) -> EvidenceAssessment:
        self.call_count += 1

        return EvidenceAssessment(
            sufficiency=self._sufficiency,
            signals=EvidenceSignals(
                retrieved_count=2,
                context_block_count=2,
                distinct_policy_count=2,
                top_score=1.0,
                second_score=0.9,
                score_margin=0.1,
            ),
            reason="Test evidence assessment.",
        )

class FakeGroundedGenerator:
    def __init__(
        self,
        text: str,
    ) -> None:
        self._text = text
        self.call_count = 0

    def generate(
        self,
        request: GroundedGenerationRequest,
    ) -> GroundedGenerationResult:
        self.call_count += 1

        return GroundedGenerationResult(
            text=self._text
        )

class FakeEntailmentProvider:
    def __init__(
        self,
        score: float,
    ) -> None:
        self._score = score

        self.received_pairs: list[
            EntailmentPair
        ] = []

    def score_entailment(
        self,
        pairs: tuple[
            EntailmentPair,
            ...
        ],
    ) -> tuple[float, ...]:
        self.received_pairs.extend(
            pairs
        )

        return tuple(
            self._score
            for _ in pairs
        )

def _retrieval_result(
) -> GroundedRetrievalResult:
    return GroundedRetrievalResult(
        retrieval_results=(),
        context_chunks=(),
        context=GroundedContext(
            blocks=(
                GroundedContextBlock(
                    policy_id="208",
                    policy_title=(
                        "Academic Dress Policy"
                    ),
                    source_url=(
                        "https://example.test/"
                        "policy/208"
                    ),
                    heading_path=(
                        "Key Decisions",
                    ),
                    start_chunk_index=0,
                    end_chunk_index=0,
                    text=(
                        "Changes to academic dress "
                        "require approval."
                    ),
                ),
                GroundedContextBlock(
                    policy_id="220",
                    policy_title=(
                        "Academic Progression "
                        "Review Policy"
                    ),
                    source_url=(
                        "https://example.test/"
                        "policy/220"
                    ),
                    heading_path=(
                        "Academic Progression",
                        "Stage Two",
                    ),
                    start_chunk_index=1,
                    end_chunk_index=1,
                    text=(
                        "Stage Two requirements "
                        "apply."
                    ),
                ),
            )
        ),
        rendered_context="Test context.",
    )

def _claim_validator(
    score: float = 0.95,
) -> GeneratedClaimGroundingValidator:
    return GeneratedClaimGroundingValidator(
        entailment_provider=(
            FakeEntailmentProvider(
                score
            )
        ),
        support_threshold=0.80,
    )

def test_ambiguous_question_skips_retrieval_and_generation():
    retrieval_result = _retrieval_result()

    retriever = RecordingRetriever(
        retrieval_result
    )

    classifier = FakeIntentClassifier(
        QuestionIntent.AMBIGUOUS
    )

    assessor = FakeEvidenceAssessor(
        EvidenceSufficiency.SUFFICIENT
    )

    generator = FakeGroundedGenerator(
        "Unused [E1]."
    )

    service = RavinAnswerService(
        retriever=retriever,
        intent_classifier=classifier,
        evidence_assessor=assessor,
        grounded_generator=generator,
        claim_grounding_validator=(
            _claim_validator()
        ),
    )

    result = service.answer(
        "What about this?"
    )

    assert (
        result.behavior
        == AnswerBehavior.CLARIFY
    )

    assert result.grounded is False
    assert result.sources == ()
    assert retriever.call_count == 0
    assert assessor.call_count == 0
    assert generator.call_count == 0
    assert classifier.call_count == 1

def test_insufficient_evidence_returns_no_grounded_answer():
    retriever = RecordingRetriever(
        _retrieval_result()
    )

    generator = FakeGroundedGenerator(
        "Unused [E1]."
    )

    service = RavinAnswerService(
        retriever=retriever,
        intent_classifier=(
            FakeIntentClassifier(
                QuestionIntent.FOCUSED
            )
        ),
        evidence_assessor=(
            FakeEvidenceAssessor(
                EvidenceSufficiency.INSUFFICIENT
            )
        ),
        grounded_generator=generator,
        claim_grounding_validator=(
            _claim_validator()
        ),
    )

    result = service.answer(
        "Who approves the change?"
    )

    assert (
        result.behavior
        == AnswerBehavior.NO_GROUNDED_ANSWER
    )

    assert result.grounded is False
    assert result.sources == ()
    assert generator.call_count == 0

def test_focused_grounded_answer_returns_real_source_metadata():
    service = RavinAnswerService(
        retriever=RecordingRetriever(
            _retrieval_result()
        ),
        intent_classifier=(
            FakeIntentClassifier(
                QuestionIntent.FOCUSED
            )
        ),
        evidence_assessor=(
            FakeEvidenceAssessor(
                EvidenceSufficiency.SUFFICIENT
            )
        ),
        grounded_generator=(
            FakeGroundedGenerator(
                "Changes require approval [E1]."
            )
        ),
        claim_grounding_validator=(
            _claim_validator()
        ),
    )

    result = service.answer(
        "Who approves changes?"
    )

    assert (
        result.behavior
        == AnswerBehavior.DIRECT_ANSWER
    )

    assert result.grounded is True
    assert len(result.sources) == 1

    source = result.sources[0]

    assert source.policy_id == "208"

    assert (
        source.title
        == "Academic Dress Policy"
    )

    assert source.heading == "Key Decisions"

    assert (
        source.url
        == "https://example.test/policy/208"
    )

def test_second_evidence_maps_to_second_policy_source():
    service = RavinAnswerService(
        retriever=RecordingRetriever(
            _retrieval_result()
        ),
        intent_classifier=(
            FakeIntentClassifier(
                QuestionIntent.FOCUSED
            )
        ),
        evidence_assessor=(
            FakeEvidenceAssessor(
                EvidenceSufficiency.SUFFICIENT
            )
        ),
        grounded_generator=(
            FakeGroundedGenerator(
                "Stage Two requirements apply [E2]."
            )
        ),
        claim_grounding_validator=(
            _claim_validator()
        ),
    )

    result = service.answer(
        "What applies at Stage Two?"
    )

    source = result.sources[0]

    assert source.policy_id == "220"

    assert source.heading == (
        "Academic Progression > Stage Two"
    )

def test_broad_question_returns_grounded_overview():
    service = RavinAnswerService(
        retriever=RecordingRetriever(
            _retrieval_result()
        ),
        intent_classifier=(
            FakeIntentClassifier(
                QuestionIntent.BROAD
            )
        ),
        evidence_assessor=(
            FakeEvidenceAssessor(
                EvidenceSufficiency.SUFFICIENT
            )
        ),
        grounded_generator=(
            FakeGroundedGenerator(
                "The policies cover approval "
                "requirements [E1]."
            )
        ),
        claim_grounding_validator=(
            _claim_validator()
        ),
    )

    result = service.answer(
        "Summarise the approval requirements."
    )

    assert (
        result.behavior
        == AnswerBehavior.GROUNDED_OVERVIEW
    )

    assert result.grounded is True

def test_rejected_generation_fails_closed():
    service = RavinAnswerService(
        retriever=RecordingRetriever(
            _retrieval_result()
        ),
        intent_classifier=(
            FakeIntentClassifier(
                QuestionIntent.FOCUSED
            )
        ),
        evidence_assessor=(
            FakeEvidenceAssessor(
                EvidenceSufficiency.SUFFICIENT
            )
        ),
        grounded_generator=(
            FakeGroundedGenerator(
                "Approval must occur within "
                "14 days [E1]."
            )
        ),
        claim_grounding_validator=(
            _claim_validator(
                score=0.20
            )
        ),
    )

    result = service.answer(
        "When must approval occur?"
    )

    assert (
        result.behavior
        == AnswerBehavior.NO_GROUNDED_ANSWER
    )

    assert result.grounded is False
    assert result.sources == ()

    assert (
        "14 days"
        not in result.answer
    )

def test_empty_question_is_rejected():
    service = RavinAnswerService(
        retriever=RecordingRetriever(
            _retrieval_result()
        ),
        intent_classifier=(
            FakeIntentClassifier(
                QuestionIntent.FOCUSED
            )
        ),
        evidence_assessor=(
            FakeEvidenceAssessor(
                EvidenceSufficiency.SUFFICIENT
            )
        ),
        grounded_generator=(
            FakeGroundedGenerator(
                "Answer [E1]."
            )
        ),
        claim_grounding_validator=(
            _claim_validator()
        ),
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        service.answer(
            " "
        )
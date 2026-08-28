from types import SimpleNamespace
import pytest
import backend.routing.deterministic_evidence_assessor as assessor_module
from backend.routing.deterministic_evidence_assessor import (
    DeterministicEvidenceSufficiencyAssessor,
)
from backend.routing.material_propositions import (
    MaterialProposition,
    MaterialPropositionKind,
    MaterialQuestionPropositions,
)
from backend.routing.material_requirements import (
    MaterialRequirement,
    MaterialRequirementKind,
)
from backend.routing.models import (
    EvidenceSignals,
    EvidenceSufficiency,
    QuestionIntent,
)
from backend.routing.proposition_coverage import (
    PropositionCoverageStatus,
    PropositionEvidenceCoverage,
    QuestionEvidenceCoverage,
)

class FakeStructureResolver:
    def __init__(
        self,
        active=object(),
    ):
        self._active = active

    def resolve(
        self,
        question,
    ):
        return SimpleNamespace(
            active=self._active
        )

class FakeRequirementExtractor:
    def __init__(
        self,
        active=object(),
    ):
        self._active = active

    def extract(
        self,
        question,
    ):
        return SimpleNamespace(
            active=self._active
        )

class FakePropositionExtractor:
    def __init__(
        self,
        propositions,
    ):
        self._propositions = propositions

    def extract(
        self,
        question,
        requirements,
        parse,
    ):
        return self._propositions

class FakeCoverageAssessor:
    def __init__(
        self,
        coverage,
    ):
        self._coverage = coverage

    def assess(
        self,
        question,
        propositions,
        evidence_texts,
    ):
        return self._coverage

def _proposition():
    return MaterialProposition(
        kind=(
            MaterialPropositionKind.RELATIONAL
        ),
        relations=(
            MaterialRequirement(
                kind=(
                    MaterialRequirementKind.RELATION
                ),
                text="provide",
            ),
        ),
    )

def _propositions():
    return MaterialQuestionPropositions(
        propositions=(
            _proposition(),
        )
    )

def _coverage(
    status: PropositionCoverageStatus,
):
    if (
        status
        == PropositionCoverageStatus.COVERED
    ):
        score = 0.95
        evidence_indexes = (
            0,
        )

    elif (
        status
        == PropositionCoverageStatus.PARTIAL
    ):
        score = 0.55
        evidence_indexes = (
            0,
        )

    else:
        score = 0.10
        evidence_indexes = ()

    return QuestionEvidenceCoverage(
        propositions=(
            PropositionEvidenceCoverage(
                proposition=_proposition(),
                status=status,
                score=score,
                evidence_indexes=(
                    evidence_indexes
                ),
            ),
        )
    )

def _signals_with_evidence():
    return EvidenceSignals(
        retrieved_count=1,
        context_block_count=1,
        distinct_policy_count=1,
        top_score=1.0,
        second_score=None,
        score_margin=None,
    )

def _signals_without_evidence():
    return EvidenceSignals(
        retrieved_count=0,
        context_block_count=0,
        distinct_policy_count=0,
        top_score=None,
        second_score=None,
        score_margin=None,
    )

def _retrieval_result():
    return SimpleNamespace(
        context=SimpleNamespace(
            blocks=(
                SimpleNamespace(
                    text="Evidence text."
                ),
            )
        )
    )

def _assessor(
    status: PropositionCoverageStatus,
):
    propositions = _propositions()

    return (
        DeterministicEvidenceSufficiencyAssessor(
            structure_resolver=(
                FakeStructureResolver()
            ),
            requirement_extractor=(
                FakeRequirementExtractor()
            ),
            proposition_extractor=(
                FakePropositionExtractor(
                    propositions
                )
            ),
            coverage_assessor=(
                FakeCoverageAssessor(
                    _coverage(
                        status
                    )
                )
            ),
        )
    )

def test_all_covered_is_sufficient(
    monkeypatch,
):
    monkeypatch.setattr(
        assessor_module,
        "extract_evidence_signals",
        lambda retrieval_result: (
            _signals_with_evidence()
        ),
    )

    result = _assessor(
        PropositionCoverageStatus.COVERED
    ).assess(
        "Question?",
        QuestionIntent.FOCUSED,
        _retrieval_result(),
    )

    assert (
        result.sufficiency
        == EvidenceSufficiency.SUFFICIENT
    )

def test_partial_coverage_is_uncertain(
    monkeypatch,
):
    monkeypatch.setattr(
        assessor_module,
        "extract_evidence_signals",
        lambda retrieval_result: (
            _signals_with_evidence()
        ),
    )

    result = _assessor(
        PropositionCoverageStatus.PARTIAL
    ).assess(
        "Question?",
        QuestionIntent.FOCUSED,
        _retrieval_result(),
    )

    assert (
        result.sufficiency
        == EvidenceSufficiency.UNCERTAIN
    )

def test_uncovered_proposition_is_insufficient(
    monkeypatch,
):
    monkeypatch.setattr(
        assessor_module,
        "extract_evidence_signals",
        lambda retrieval_result: (
            _signals_with_evidence()
        ),
    )

    result = _assessor(
        PropositionCoverageStatus.UNCOVERED
    ).assess(
        "Question?",
        QuestionIntent.FOCUSED,
        _retrieval_result(),
    )

    assert (
        result.sufficiency
        == EvidenceSufficiency.INSUFFICIENT
    )

def test_no_evidence_is_insufficient(
    monkeypatch,
):
    monkeypatch.setattr(
        assessor_module,
        "extract_evidence_signals",
        lambda retrieval_result: (
            _signals_without_evidence()
        ),
    )

    result = _assessor(
        PropositionCoverageStatus.COVERED
    ).assess(
        "Question?",
        QuestionIntent.FOCUSED,
        _retrieval_result(),
    )

    assert (
        result.sufficiency
        == EvidenceSufficiency.INSUFFICIENT
    )

def test_unresolved_structure_is_uncertain(
    monkeypatch,
):
    monkeypatch.setattr(
        assessor_module,
        "extract_evidence_signals",
        lambda retrieval_result: (
            _signals_with_evidence()
        ),
    )

    assessor = (
        DeterministicEvidenceSufficiencyAssessor(
            structure_resolver=(
                FakeStructureResolver(
                    active=None
                )
            ),
            requirement_extractor=(
                FakeRequirementExtractor()
            ),
            proposition_extractor=(
                FakePropositionExtractor(
                    _propositions()
                )
            ),
            coverage_assessor=(
                FakeCoverageAssessor(
                    _coverage(
                        PropositionCoverageStatus.COVERED
                    )
                )
            ),
        )
    )

    result = assessor.assess(
        "Question?",
        QuestionIntent.FOCUSED,
        _retrieval_result(),
    )

    assert (
        result.sufficiency
        == EvidenceSufficiency.UNCERTAIN
    )
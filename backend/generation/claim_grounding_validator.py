"""
Validate generated factual claims against their cited policy evidence.

This module checks whether generated claims are supported by the
evidence blocks they cite. It evaluates deterministic support windows
with the configured entailment provider and retains the strongest
valid entailment score for each claim.

Claim grounding is separate from question-level evidence sufficiency.
A generated answer that fails this validation is not released.
"""

import re
from dataclasses import dataclass
from backend.generation.grounded_generator import (
    GroundedGenerationRequest,
    GroundedGenerationResult,
)
from backend.generation.entailment import (
    EntailmentPair,
    EntailmentProvider,
    score_entailment,
)
from backend.generation.evidence_windows import (
    EvidenceSupportWindowBuilder,
)

_EVIDENCE_MARKER_PATTERN = re.compile(
    r"\[E(\d+)\]"
)
_SENTENCE_PATTERN = re.compile(
    r"[^.!?\n]+(?:[.!?]+|$)"
)

@dataclass(frozen=True)
class ClaimGroundingResult:
    """Record support evidence and grounding score for one generated factual claim.
    """

    claim: str
    cited_evidence_indexes: tuple[int, ...]
    score: float
    supported: bool

    def __post_init__(self) -> None:
        if not self.claim.strip():
            raise ValueError(
                "Grounding claim cannot be empty."
            )

        if any(
            index < 1
            for index in self.cited_evidence_indexes
        ):
            raise ValueError(
                "Grounding evidence indexes must "
                "be positive."
            )

        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                "Grounding score must be between "
                "0 and 1."
            )

@dataclass(frozen=True)
class ClaimGroundingValidationResult:
    """Aggregate claim-level grounding results for one generated answer.
    """

    valid: bool
    claims: tuple[ClaimGroundingResult, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.claims:
            raise ValueError(
                "Claim grounding validation must "
                "contain at least one claim."
            )

        if not self.reason.strip():
            raise ValueError(
                "Claim grounding validation reason "
                "cannot be empty."
            )

class GeneratedClaimGroundingValidator:
    """Validate generated factual claims against the evidence they cite.

    Claim support is checked independently of earlier question-answerability
    assessment so generated wording must itself remain grounded.
    """

    def __init__(
        self,
        entailment_provider: EntailmentProvider,
        support_threshold: float,
        window_builder: (
            EvidenceSupportWindowBuilder
            | None
        ) = None,
    ) -> None:
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError(
                "Claim grounding threshold must "
                "be between 0 and 1."
            )

        self._entailment_provider = (
            entailment_provider
        )

        self._support_threshold = (
            support_threshold
        )

        self._window_builder = (
            EvidenceSupportWindowBuilder(
                max_units=3
            )
            if window_builder is None
            else window_builder
        )
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError(
                "Claim grounding threshold must "
                "be between 0 and 1."
            )

        self._support_threshold = (
            support_threshold
        )

    def validate(
        self,
        request: GroundedGenerationRequest,
        result: GroundedGenerationResult,
    ) -> ClaimGroundingValidationResult:
        """Validate every generated claim and fail when citations or support are missing.
        """
        if not isinstance(
            request,
            GroundedGenerationRequest,
        ):
            raise ValueError(
                "Grounding request must be a "
                "GroundedGenerationRequest."
            )

        if not isinstance(
            result,
            GroundedGenerationResult,
        ):
            raise ValueError(
                "Grounding result must be a "
                "GroundedGenerationResult."
            )

        claim_texts = _extract_claim_texts(
            result.text
        )

        claims = tuple(
            self._assess_claim(
                claim_text,
                request.evidence_texts,
            )
            for claim_text in claim_texts
        )

        if any(
            not claim.cited_evidence_indexes
            for claim in claims
        ):
            return ClaimGroundingValidationResult(
                valid=False,
                claims=claims,
                reason=(
                    "At least one generated claim "
                    "does not cite approved evidence."
                ),
            )

        if any(
            not claim.supported
            for claim in claims
        ):
            return ClaimGroundingValidationResult(
                valid=False,
                claims=claims,
                reason=(
                    "At least one generated claim "
                    "is not sufficiently supported "
                    "by its cited evidence."
                ),
            )

        return ClaimGroundingValidationResult(
            valid=True,
            claims=claims,
            reason=(
                "All generated claims are "
                "supported by their cited evidence."
            ),
        )

    def _assess_claim(
        self,
        claim_text: str,
        evidence_texts: tuple[str, ...],
    ) -> ClaimGroundingResult:
        cited_indexes = (
            _extract_cited_indexes(
                claim_text
            )
        )

        clean_claim = (
            _remove_evidence_markers(
                claim_text
            )
        )

        if not cited_indexes:
            return ClaimGroundingResult(
                claim=clean_claim,
                cited_evidence_indexes=(),
                score=0.0,
                supported=False,
            )

        maximum_index = len(
            evidence_texts
        )

        if any(
            index > maximum_index
            for index in cited_indexes
        ):
            return ClaimGroundingResult(
                claim=clean_claim,
                cited_evidence_indexes=(
                    cited_indexes
                ),
                score=0.0,
                supported=False,
            )

        cited_evidence = tuple(
            evidence_texts[index - 1]
            for index in cited_indexes
        )

        support_windows = tuple(
            window
            for evidence_text in cited_evidence
            for window in self._window_builder.build(
                evidence_text
            )
        )

        entailment_pairs = tuple(
            EntailmentPair(
                premise=window.text,
                hypothesis=clean_claim,
            )
            for window in support_windows
        )

        entailment_scores = score_entailment(
            self._entailment_provider,
            entailment_pairs,
        )

        strongest_score = max(
            entailment_scores
        )

        return ClaimGroundingResult(
            claim=clean_claim,
            cited_evidence_indexes=(
                cited_indexes
            ),
            score=strongest_score,
            supported=(
                strongest_score
                >= self._support_threshold
            ),
        )

def _extract_claim_texts(
    text: str,
) -> tuple[str, ...]:
    claims = tuple(
        match.group(0).strip()
        for match in _SENTENCE_PATTERN.finditer(
            text
        )
        if match.group(0).strip()
    )

    if not claims:
        raise ValueError(
            "Generated answer contains no "
            "claims to validate."
        )

    return claims

def _extract_cited_indexes(
    text: str,
) -> tuple[int, ...]:
    indexes = tuple(
        int(match)
        for match in (
            _EVIDENCE_MARKER_PATTERN.findall(
                text
            )
        )
    )

    return tuple(
        dict.fromkeys(
            indexes
        )
    )

def _remove_evidence_markers(
    text: str,
) -> str:
    clean_text = (
        _EVIDENCE_MARKER_PATTERN.sub(
            "",
            text,
        )
        .strip()
    )

    clean_text = re.sub(
        r"\s+([.!?,;:])",
        r"\1",
        clean_text,
    )

    if not clean_text:
        raise ValueError(
            "Generated claim cannot contain "
            "only evidence markers."
        )

    return clean_text
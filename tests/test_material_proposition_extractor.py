from typing import cast
import pytest
from backend.routing.material_proposition_extractor import (
    extract_material_propositions,
)
from backend.routing.material_propositions import (
    MaterialProposition,
    MaterialPropositionKind,
    MaterialQuestionPropositions,
)
from backend.routing.material_requirements import (
    MaterialQuestionRequirements,
    MaterialRequirement,
    MaterialRequirementKind,
)
from backend.routing.question_parser import (
    ParsedToken,
    QuestionParse,
)

class FakeExtractor:
    def __init__(
        self,
        result: MaterialQuestionPropositions,
    ) -> None:
        self._result = result

        self.received_question: (
            str | None
        ) = None

        self.received_requirements: (
            MaterialQuestionRequirements
            | None
        ) = None

        self.received_parse: (
            QuestionParse | None
        ) = None

    def extract(
        self,
        question: str,
        requirements: MaterialQuestionRequirements,
        parse: QuestionParse,
    ) -> MaterialQuestionPropositions:
        self.received_question = question
        self.received_requirements = (
            requirements
        )
        self.received_parse = parse

        return self._result

def _requirements(
) -> MaterialQuestionRequirements:
    return MaterialQuestionRequirements(
        requirements=(
            MaterialRequirement(
                kind=(
                    MaterialRequirementKind.RELATION
                ),
                text="apply",
            ),
            MaterialRequirement(
                kind=(
                    MaterialRequirementKind.CONCEPT
                ),
                text="student",
            ),
        )
    )

def _parse(
) -> QuestionParse:
    return QuestionParse(
        tokens=(
            ParsedToken(
                index=0,
                text="apply",
                lemma="apply",
                pos="VERB",
                tag="VBP",
                dependency="ROOT",
                head_index=0,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
        ),
        noun_phrases=(),
    )

def _propositions(
) -> MaterialQuestionPropositions:
    return MaterialQuestionPropositions(
        propositions=(
            MaterialProposition(
                kind=(
                    MaterialPropositionKind.RELATIONAL
                ),
                relations=(
                    MaterialRequirement(
                        kind=(
                            MaterialRequirementKind.RELATION
                        ),
                        text="apply",
                    ),
                ),
            ),
        )
    )

def test_returns_material_question_propositions():
    expected = _propositions()

    result = extract_material_propositions(
        "Can a student apply?",
        _requirements(),
        _parse(),
        FakeExtractor(
            expected
        ),
    )

    assert result is expected

def test_strips_question_before_extraction():
    extractor = FakeExtractor(
        _propositions()
    )

    extract_material_propositions(
        "  Can a student apply?  ",
        _requirements(),
        _parse(),
        extractor,
    )

    assert (
        extractor.received_question
        == "Can a student apply?"
    )

def test_passes_requirements_and_parse_to_extractor():
    requirements = _requirements()
    parse = _parse()

    extractor = FakeExtractor(
        _propositions()
    )

    extract_material_propositions(
        "Can a student apply?",
        requirements,
        parse,
        extractor,
    )

    assert (
        extractor.received_requirements
        is requirements
    )

    assert extractor.received_parse is parse

def test_rejects_empty_question():
    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        extract_material_propositions(
            "   ",
            _requirements(),
            _parse(),
            FakeExtractor(
                _propositions()
            ),
        )

def test_rejects_invalid_requirements():
    with pytest.raises(
        ValueError,
        match=(
            "Requirements must be "
            "MaterialQuestionRequirements."
        ),
    ):
        extract_material_propositions(
            "Question?",
            cast(
                MaterialQuestionRequirements,
                "invalid",
            ),
            _parse(),
            FakeExtractor(
                _propositions()
            ),
        )

def test_rejects_invalid_parse():
    with pytest.raises(
        ValueError,
        match=(
            "Parse must be a QuestionParse."
        ),
    ):
        extract_material_propositions(
            "Question?",
            _requirements(),
            cast(
                QuestionParse,
                "invalid",
            ),
            FakeExtractor(
                _propositions()
            ),
        )

def test_rejects_invalid_extractor_result():
    class InvalidExtractor:
        def extract(
            self,
            question: str,
            requirements: MaterialQuestionRequirements,
            parse: QuestionParse,
        ) -> MaterialQuestionPropositions:
            return cast(
                MaterialQuestionPropositions,
                "invalid",
            )

    with pytest.raises(
        ValueError,
        match=(
            "Material proposition extractor "
            "must return "
            "MaterialQuestionPropositions."
        ),
    ):
        extract_material_propositions(
            "Question?",
            _requirements(),
            _parse(),
            InvalidExtractor(),
        )
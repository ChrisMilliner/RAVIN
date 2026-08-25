import pytest
from backend.routing.dependency_material_requirement_extractor import (
    DependencyMaterialRequirementExtractor,
)
from backend.routing.material_requirements import (
    MaterialRequirementKind,
)
from backend.routing.question_parser import (
    ParsedSpan,
    ParsedToken,
    QuestionParse,
    QuestionParseResult,
)

class FakeQuestionParser:
    def __init__(
        self,
        primary: QuestionParse,
        fallback: QuestionParse | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    def parse(
        self,
        question: str,
    ) -> QuestionParseResult:
        return QuestionParseResult(
            primary=self._primary,
            fallback=self._fallback,
        )

def _token(
    index: int,
    text: str,
    lemma: str,
    pos: str,
    tag: str,
    dependency: str,
    head_index: int,
    *,
    is_stop: bool = False,
    is_punct: bool = False,
    is_alpha: bool = True,
) -> ParsedToken:
    return ParsedToken(
        index=index,
        text=text,
        lemma=lemma,
        pos=pos,
        tag=tag,
        dependency=dependency,
        head_index=head_index,
        is_stop=is_stop,
        is_punct=is_punct,
        is_alpha=is_alpha,
    )

def _pairs(
    extractor: DependencyMaterialRequirementExtractor,
    question: str,
) -> set[
    tuple[
        MaterialRequirementKind,
        str,
    ]
]:
    result = extractor.extract(
        question
    )

    return {
        (
            requirement.kind,
            requirement.text,
        )
        for requirement
        in result.requirements
    }

def test_standalone_question_word_is_not_a_requested_attribute():
    parse = QuestionParse(
        tokens=(
            _token(
                0,
                "Who",
                "who",
                "PRON",
                "WP",
                "nsubj",
                1,
                is_stop=True,
            ),
            _token(
                1,
                "approves",
                "approve",
                "VERB",
                "VBZ",
                "ROOT",
                1,
            ),
            _token(
                2,
                "requests",
                "request",
                "NOUN",
                "NNS",
                "dobj",
                1,
            ),
            _token(
                3,
                "?",
                "?",
                "PUNCT",
                ".",
                "punct",
                1,
                is_punct=True,
                is_alpha=False,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="Who",
                start_index=0,
                end_index=1,
                root_index=0,
            ),
            ParsedSpan(
                text="requests",
                start_index=2,
                end_index=3,
                root_index=2,
            ),
        ),
    )

    extractor = (
        DependencyMaterialRequirementExtractor(
            FakeQuestionParser(
                parse
            )
        )
    )

    requirements = _pairs(
        extractor,
        "Who approves requests?",
    )

    assert (
        MaterialRequirementKind.RELATION,
        "approve",
    ) in requirements

    assert (
        MaterialRequirementKind.CONCEPT,
        "requests",
    ) in requirements

    assert (
        MaterialRequirementKind.REQUESTED_ATTRIBUTE,
        "Who",
    ) not in requirements

def test_coordinated_requested_attributes_are_extracted():
    parse = QuestionParse(
        tokens=(
            _token(
                0,
                "What",
                "what",
                "DET",
                "WDT",
                "det",
                1,
                is_stop=True,
            ),
            _token(
                1,
                "fees",
                "fee",
                "NOUN",
                "NNS",
                "dobj",
                9,
            ),
            _token(
                2,
                ",",
                ",",
                "PUNCT",
                ",",
                "punct",
                1,
                is_punct=True,
                is_alpha=False,
            ),
            _token(
                3,
                "discounts",
                "discount",
                "NOUN",
                "NNS",
                "conj",
                1,
            ),
            _token(
                4,
                "and",
                "and",
                "CCONJ",
                "CC",
                "cc",
                3,
                is_stop=True,
            ),
            _token(
                5,
                "grants",
                "grant",
                "NOUN",
                "NNS",
                "conj",
                3,
            ),
            _token(
                6,
                "does",
                "do",
                "AUX",
                "VBZ",
                "aux",
                9,
                is_stop=True,
            ),
            _token(
                7,
                "the",
                "the",
                "DET",
                "DT",
                "det",
                8,
                is_stop=True,
            ),
            _token(
                8,
                "program",
                "program",
                "NOUN",
                "NN",
                "nsubj",
                9,
            ),
            _token(
                9,
                "provide",
                "provide",
                "VERB",
                "VB",
                "ROOT",
                9,
            ),
            _token(
                10,
                "?",
                "?",
                "PUNCT",
                ".",
                "punct",
                9,
                is_punct=True,
                is_alpha=False,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="What fees",
                start_index=0,
                end_index=2,
                root_index=1,
            ),
            ParsedSpan(
                text="discounts",
                start_index=3,
                end_index=4,
                root_index=3,
            ),
            ParsedSpan(
                text="grants",
                start_index=5,
                end_index=6,
                root_index=5,
            ),
            ParsedSpan(
                text="the program",
                start_index=7,
                end_index=9,
                root_index=8,
            ),
        ),
    )

    extractor = (
        DependencyMaterialRequirementExtractor(
            FakeQuestionParser(
                parse
            )
        )
    )

    requirements = _pairs(
        extractor,
        (
            "What fees, discounts and grants "
            "does the program provide?"
        ),
    )

    assert (
        MaterialRequirementKind.REQUESTED_ATTRIBUTE,
        "fees",
    ) in requirements

    assert (
        MaterialRequirementKind.REQUESTED_ATTRIBUTE,
        "discounts",
    ) in requirements

    assert (
        MaterialRequirementKind.REQUESTED_ATTRIBUTE,
        "grants",
    ) in requirements

    assert (
        MaterialRequirementKind.RELATION,
        "provide",
    ) in requirements

    assert (
        MaterialRequirementKind.CONCEPT,
        "the program",
    ) in requirements

def test_adverbial_modifier_is_a_generic_qualifier():
    parse = QuestionParse(
        tokens=(
            _token(
                0,
                "Does",
                "do",
                "AUX",
                "VBZ",
                "aux",
                4,
                is_stop=True,
            ),
            _token(
                1,
                "a",
                "a",
                "DET",
                "DT",
                "det",
                2,
                is_stop=True,
            ),
            _token(
                2,
                "permit",
                "permit",
                "NOUN",
                "NN",
                "nsubj",
                4,
            ),
            _token(
                3,
                "automatically",
                "automatically",
                "ADV",
                "RB",
                "advmod",
                4,
            ),
            _token(
                4,
                "expire",
                "expire",
                "VERB",
                "VB",
                "ROOT",
                4,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="a permit",
                start_index=1,
                end_index=3,
                root_index=2,
            ),
        ),
    )

    extractor = (
        DependencyMaterialRequirementExtractor(
            FakeQuestionParser(
                parse
            )
        )
    )

    requirements = _pairs(
        extractor,
        "Does a permit automatically expire?",
    )

    assert (
        MaterialRequirementKind.RELATION,
        "expire",
    ) in requirements

    assert (
        MaterialRequirementKind.QUALIFIER,
        "automatically",
    ) in requirements

def test_modal_and_negation_are_preserved():
    parse = QuestionParse(
        tokens=(
            _token(
                0,
                "Can",
                "can",
                "AUX",
                "MD",
                "aux",
                4,
                is_stop=True,
            ),
            _token(
                1,
                "a",
                "a",
                "DET",
                "DT",
                "det",
                2,
                is_stop=True,
            ),
            _token(
                2,
                "member",
                "member",
                "NOUN",
                "NN",
                "nsubj",
                4,
            ),
            _token(
                3,
                "not",
                "not",
                "PART",
                "RB",
                "neg",
                4,
                is_stop=True,
            ),
            _token(
                4,
                "submit",
                "submit",
                "VERB",
                "VB",
                "ROOT",
                4,
            ),
            _token(
                5,
                "a",
                "a",
                "DET",
                "DT",
                "det",
                6,
                is_stop=True,
            ),
            _token(
                6,
                "request",
                "request",
                "NOUN",
                "NN",
                "dobj",
                4,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="a member",
                start_index=1,
                end_index=3,
                root_index=2,
            ),
            ParsedSpan(
                text="a request",
                start_index=5,
                end_index=7,
                root_index=6,
            ),
        ),
    )

    extractor = (
        DependencyMaterialRequirementExtractor(
            FakeQuestionParser(
                parse
            )
        )
    )

    requirements = _pairs(
        extractor,
        "Can a member not submit a request?",
    )

    assert (
        MaterialRequirementKind.MODALITY,
        "can",
    ) in requirements

    assert (
        MaterialRequirementKind.NEGATION,
        "not",
    ) in requirements

    assert (
        MaterialRequirementKind.RELATION,
        "submit",
    ) in requirements

def test_subordinate_clause_is_preserved_as_a_condition():
    parse = QuestionParse(
        tokens=(
            _token(
                0,
                "What",
                "what",
                "PRON",
                "WP",
                "nsubj",
                1,
                is_stop=True,
            ),
            _token(
                1,
                "happens",
                "happen",
                "VERB",
                "VBZ",
                "ROOT",
                1,
            ),
            _token(
                2,
                "when",
                "when",
                "SCONJ",
                "WRB",
                "advmod",
                5,
                is_stop=True,
            ),
            _token(
                3,
                "a",
                "a",
                "DET",
                "DT",
                "det",
                4,
                is_stop=True,
            ),
            _token(
                4,
                "member",
                "member",
                "NOUN",
                "NN",
                "nsubj",
                5,
            ),
            _token(
                5,
                "misses",
                "miss",
                "VERB",
                "VBZ",
                "advcl",
                1,
            ),
            _token(
                6,
                "a",
                "a",
                "DET",
                "DT",
                "det",
                7,
                is_stop=True,
            ),
            _token(
                7,
                "deadline",
                "deadline",
                "NOUN",
                "NN",
                "dobj",
                5,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="What",
                start_index=0,
                end_index=1,
                root_index=0,
            ),
            ParsedSpan(
                text="a member",
                start_index=3,
                end_index=5,
                root_index=4,
            ),
            ParsedSpan(
                text="a deadline",
                start_index=6,
                end_index=8,
                root_index=7,
            ),
        ),
    )

    extractor = (
        DependencyMaterialRequirementExtractor(
            FakeQuestionParser(
                parse
            )
        )
    )

    requirements = _pairs(
        extractor,
        (
            "What happens when a member "
            "misses a deadline?"
        ),
    )

    assert (
        MaterialRequirementKind.CONDITION,
        "when a member misses a deadline",
    ) in requirements

    assert (
        MaterialRequirementKind.REQUESTED_ATTRIBUTE,
        "What",
    ) not in requirements

def test_how_long_is_preserved_as_requested_information():
    parse = QuestionParse(
        tokens=(
            _token(
                0,
                "How",
                "how",
                "SCONJ",
                "WRB",
                "advmod",
                1,
                is_stop=True,
            ),
            _token(
                1,
                "long",
                "long",
                "ADV",
                "RB",
                "advmod",
                5,
            ),
            _token(
                2,
                "can",
                "can",
                "AUX",
                "MD",
                "aux",
                5,
                is_stop=True,
            ),
            _token(
                3,
                "a",
                "a",
                "DET",
                "DT",
                "det",
                4,
                is_stop=True,
            ),
            _token(
                4,
                "member",
                "member",
                "NOUN",
                "NN",
                "nsubj",
                5,
            ),
            _token(
                5,
                "wait",
                "wait",
                "VERB",
                "VB",
                "ROOT",
                5,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="a member",
                start_index=3,
                end_index=5,
                root_index=4,
            ),
        ),
    )

    extractor = (
        DependencyMaterialRequirementExtractor(
            FakeQuestionParser(
                parse
            )
        )
    )

    requirements = _pairs(
        extractor,
        "How long can a member wait?",
    )

    assert (
        MaterialRequirementKind.REQUESTED_ATTRIBUTE,
        "How long",
    ) in requirements

    assert (
        MaterialRequirementKind.MODALITY,
        "can",
    ) in requirements

    assert (
        MaterialRequirementKind.RELATION,
        "wait",
    ) in requirements

def test_primary_and_fallback_requirements_are_deduplicated():
    parse = QuestionParse(
        tokens=(
            _token(
                0,
                "items",
                "item",
                "NOUN",
                "NNS",
                "nsubj",
                1,
            ),
            _token(
                1,
                "expire",
                "expire",
                "VERB",
                "VBP",
                "ROOT",
                1,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="items",
                start_index=0,
                end_index=1,
                root_index=0,
            ),
        ),
    )

    extractor = (
        DependencyMaterialRequirementExtractor(
            FakeQuestionParser(
                primary=parse,
                fallback=parse,
            )
        )
    )

    result = extractor.extract(
        "Items expire?"
    )

    pairs = tuple(
        (
            requirement.kind,
            requirement.text,
        )
        for requirement
        in result.requirements
    )

    assert len(pairs) == len(
        set(pairs)
    )

def test_empty_question_is_rejected():
    parse = QuestionParse(
        tokens=(),
        noun_phrases=(),
    )

    extractor = (
        DependencyMaterialRequirementExtractor(
            FakeQuestionParser(
                parse
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        extractor.extract(
            "   "
        )
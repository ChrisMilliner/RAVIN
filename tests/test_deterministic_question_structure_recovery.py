from backend.routing.deterministic_question_structure_recovery import (
    DeterministicQuestionStructureRecoveryProvider,
)
from backend.routing.question_parser import (
    ParsedSpan,
    ParsedToken,
    QuestionParse,
    QuestionParseResult,
    is_question_parse_suspicious,
)

def _token(
    index: int,
    text: str,
    lemma: str,
    pos: str,
    tag: str,
    dependency: str,
    head_index: int,
) -> ParsedToken:
    return ParsedToken(
        index=index,
        text=text,
        lemma=lemma,
        pos=pos,
        tag=tag,
        dependency=dependency,
        head_index=head_index,
        is_stop=False,
        is_punct=(
            pos == "PUNCT"
        ),
        is_alpha=(
            pos != "PUNCT"
        ),
    )

def _usable_parse(
) -> QuestionParse:
    return QuestionParse(
        tokens=(
            _token(
                0,
                "apply",
                "apply",
                "VERB",
                "VBP",
                "ROOT",
                0,
            ),
        ),
        noun_phrases=(),
    )

def _do_supported_nominal_parse(
) -> QuestionParse:
    return QuestionParse(
        tokens=(
            _token(
                0,
                "How",
                "how",
                "SCONJ",
                "WRB",
                "advmod",
                3,
            ),
            _token(
                1,
                "does",
                "do",
                "AUX",
                "VBZ",
                "aux",
                3,
            ),
            _token(
                2,
                "University",
                "university",
                "PROPN",
                "NNP",
                "nsubj",
                3,
            ),
            _token(
                3,
                "support",
                "support",
                "NOUN",
                "NN",
                "ROOT",
                3,
            ),
            _token(
                4,
                "admission",
                "admission",
                "NOUN",
                "NN",
                "dobj",
                3,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="University",
                start_index=2,
                end_index=3,
                root_index=2,
            ),
            ParsedSpan(
                text="admission",
                start_index=4,
                end_index=5,
                root_index=4,
            ),
        ),
    )

def _modal_nominal_parse(
) -> QuestionParse:
    return QuestionParse(
        tokens=(
            _token(
                0,
                "Can",
                "can",
                "AUX",
                "MD",
                "aux",
                3,
            ),
            _token(
                1,
                "a",
                "a",
                "DET",
                "DT",
                "det",
                3,
            ),
            _token(
                2,
                "member",
                "member",
                "NOUN",
                "NN",
                "compound",
                3,
            ),
            _token(
                3,
                "student",
                "student",
                "NOUN",
                "NN",
                "ROOT",
                3,
            ),
            _token(
                4,
                "on",
                "on",
                "ADP",
                "IN",
                "prep",
                3,
            ),
            _token(
                5,
                "a",
                "a",
                "DET",
                "DT",
                "det",
                7,
            ),
            _token(
                6,
                "student",
                "student",
                "NOUN",
                "NN",
                "compound",
                7,
            ),
            _token(
                7,
                "visa",
                "visa",
                "NOUN",
                "NN",
                "compound",
                8,
            ),
            _token(
                8,
                "enrol",
                "enrol",
                "VERB",
                "VB",
                "pobj",
                4,
            ),
            _token(
                9,
                "in",
                "in",
                "ADP",
                "IN",
                "prep",
                8,
            ),
            _token(
                10,
                "Subject",
                "subject",
                "NOUN",
                "NN",
                "pobj",
                9,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="Can a member student",
                start_index=0,
                end_index=4,
                root_index=3,
            ),
            ParsedSpan(
                text="Subject",
                start_index=10,
                end_index=11,
                root_index=10,
            ),
        ),
    )

def _bare_false_root_parse(
) -> QuestionParse:
    return QuestionParse(
        tokens=(
            _token(
                0,
                "What",
                "what",
                "DET",
                "WDT",
                "det",
                2,
            ),
            _token(
                1,
                "financial",
                "financial",
                "ADJ",
                "JJ",
                "amod",
                2,
            ),
            _token(
                2,
                "benefits",
                "benefit",
                "NOUN",
                "NNS",
                "nsubj",
                4,
            ),
            _token(
                3,
                ",",
                ",",
                "PUNCT",
                ",",
                "punct",
                2,
            ),
            _token(
                4,
                "leave",
                "leave",
                "VERB",
                "VB",
                "ROOT",
                4,
            ),
            _token(
                5,
                "entitlements",
                "entitlement",
                "NOUN",
                "NNS",
                "dobj",
                4,
            ),
            _token(
                6,
                "and",
                "and",
                "CCONJ",
                "CC",
                "cc",
                5,
            ),
            _token(
                7,
                "salary",
                "salary",
                "NOUN",
                "NN",
                "compound",
                8,
            ),
            _token(
                8,
                "changes",
                "change",
                "NOUN",
                "NNS",
                "conj",
                5,
            ),
            _token(
                9,
                "apply",
                "apply",
                "VERB",
                "VBP",
                "conj",
                4,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="What financial benefits",
                start_index=0,
                end_index=3,
                root_index=2,
            ),
            ParsedSpan(
                text="entitlements",
                start_index=5,
                end_index=6,
                root_index=5,
            ),
            ParsedSpan(
                text="salary changes",
                start_index=7,
                end_index=9,
                root_index=8,
            ),
        ),
    )

def test_resolved_parse_is_not_recovered():
    parse = _usable_parse()

    provider = (
        DeterministicQuestionStructureRecoveryProvider()
    )

    result = provider.recover(
        "Members apply?",
        QuestionParseResult(
            primary=parse,
        ),
    )

    assert result is None

def test_recovers_do_supported_nominal_predicate():
    suspicious = (
        _do_supported_nominal_parse()
    )

    provider = (
        DeterministicQuestionStructureRecoveryProvider()
    )

    recovered = provider.recover(
        "How does University support admission?",
        QuestionParseResult(
            primary=suspicious,
            fallback=suspicious,
        ),
    )

    assert recovered is not None
    assert not is_question_parse_suspicious(
        recovered
    )

    roots = recovered.roots

    assert len(roots) == 1
    assert roots[0].text == "support"
    assert roots[0].pos == "VERB"

def test_recovers_modal_nominal_root_with_later_base_verb():
    suspicious = (
        _modal_nominal_parse()
    )

    provider = (
        DeterministicQuestionStructureRecoveryProvider()
    )

    recovered = provider.recover(
        "Can a member student on a student visa enrol in Subject?",
        QuestionParseResult(
            primary=suspicious,
            fallback=suspicious,
        ),
    )

    assert recovered is not None
    assert not is_question_parse_suspicious(
        recovered
    )

    roots = recovered.roots

    assert len(roots) == 1
    assert roots[0].text == "enrol"

    tokens = {
        token.text: token
        for token in recovered.tokens
    }

    assert (
        tokens["Can"].head_index
        == tokens["enrol"].index
    )

    assert any(
        span.text == "a student visa"
        for span in recovered.noun_phrases
    )

def test_recovers_false_bare_root_and_requested_chain():
    suspicious = (
        _bare_false_root_parse()
    )

    provider = (
        DeterministicQuestionStructureRecoveryProvider()
    )

    recovered = provider.recover(
        (
            "What financial benefits, leave "
            "entitlements and salary changes apply?"
        ),
        QuestionParseResult(
            primary=suspicious,
            fallback=suspicious,
        ),
    )

    assert recovered is not None
    assert not is_question_parse_suspicious(
        recovered
    )

    roots = recovered.roots

    assert len(roots) == 1
    assert roots[0].text == "apply"

    leave = next(
        token
        for token in recovered.tokens
        if token.text == "leave"
    )

    assert leave.pos == "NOUN"

    assert any(
        span.text == "leave entitlements"
        for span in recovered.noun_phrases
    )

def test_unknown_unresolved_structure_is_not_recovered():
    suspicious = QuestionParse(
        tokens=(
            _token(
                0,
                "support",
                "support",
                "NOUN",
                "NN",
                "ROOT",
                0,
            ),
        ),
        noun_phrases=(),
    )

    provider = (
        DeterministicQuestionStructureRecoveryProvider()
    )

    recovered = provider.recover(
        "Support?",
        QuestionParseResult(
            primary=suspicious,
            fallback=suspicious,
        ),
    )

    assert recovered is None
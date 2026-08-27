from backend.routing.dependency_material_proposition_extractor import (
    DependencyMaterialPropositionExtractor,
)
from backend.routing.material_propositions import (
    MaterialPropositionKind,
)
from backend.routing.material_requirements import (
    MaterialQuestionRequirements,
    MaterialRequirement,
    MaterialRequirementKind,
)
from backend.routing.question_parser import (
    ParsedSpan,
    ParsedToken,
    QuestionParse,
)

def _requirement(
    kind: MaterialRequirementKind,
    text: str,
) -> MaterialRequirement:
    return MaterialRequirement(
        kind=kind,
        text=text,
    )

def test_extracts_simple_relational_proposition():
    requirements = MaterialQuestionRequirements(
        requirements=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "replace",
            ),
            _requirement(
                MaterialRequirementKind.QUALIFIER,
                "permanently",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "Professional Equivalence",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "the need",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "an academic qualification",
            ),
        )
    )

    parse = QuestionParse(
        tokens=(
            ParsedToken(
                index=0,
                text="Equivalence",
                lemma="Equivalence",
                pos="PROPN",
                tag="NNP",
                dependency="nsubj",
                head_index=2,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=1,
                text="permanently",
                lemma="permanently",
                pos="ADV",
                tag="RB",
                dependency="advmod",
                head_index=2,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=2,
                text="replace",
                lemma="replace",
                pos="VERB",
                tag="VB",
                dependency="ROOT",
                head_index=2,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=3,
                text="need",
                lemma="need",
                pos="NOUN",
                tag="NN",
                dependency="dobj",
                head_index=2,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=4,
                text="qualification",
                lemma="qualification",
                pos="NOUN",
                tag="NN",
                dependency="pobj",
                head_index=3,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="Professional Equivalence",
                start_index=0,
                end_index=1,
                root_index=0,
            ),
            ParsedSpan(
                text="the need",
                start_index=3,
                end_index=4,
                root_index=3,
            ),
            ParsedSpan(
                text="an academic qualification",
                start_index=4,
                end_index=5,
                root_index=4,
            ),
        ),
    )

    result = (
        DependencyMaterialPropositionExtractor()
        .extract(
            "Does Professional Equivalence permanently replace the need for an academic qualification?",
            requirements,
            parse,
        )
    )

    proposition = result.propositions[0]

    assert (
        proposition.kind
        == MaterialPropositionKind.RELATIONAL
    )

    assert tuple(
        item.text
        for item in proposition.subjects
    ) == (
        "Professional Equivalence",
    )

    assert tuple(
        item.text
        for item in proposition.relations
    ) == (
        "replace",
    )

    assert tuple(
        item.text
        for item in proposition.objects
    ) == (
        "the need",
        "an academic qualification",
    )

    assert tuple(
        item.text
        for item in proposition.qualifiers
    ) == (
        "permanently",
    )

def test_extracts_predicate_chain_as_one_proposition():
    requirements = MaterialQuestionRequirements(
        requirements=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "have",
            ),
            _requirement(
                MaterialRequirementKind.RELATION,
                "submit",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "a student",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "a show cause response",
            ),
        )
    )

    parse = QuestionParse(
        tokens=(
            ParsedToken(
                index=0,
                text="student",
                lemma="student",
                pos="NOUN",
                tag="NN",
                dependency="nsubj",
                head_index=1,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=1,
                text="have",
                lemma="have",
                pos="VERB",
                tag="VBP",
                dependency="ROOT",
                head_index=1,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=2,
                text="submit",
                lemma="submit",
                pos="VERB",
                tag="VB",
                dependency="xcomp",
                head_index=1,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=3,
                text="response",
                lemma="response",
                pos="NOUN",
                tag="NN",
                dependency="dobj",
                head_index=2,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="a student",
                start_index=0,
                end_index=1,
                root_index=0,
            ),
            ParsedSpan(
                text="a show cause response",
                start_index=3,
                end_index=4,
                root_index=3,
            ),
        ),
    )

    result = (
        DependencyMaterialPropositionExtractor()
        .extract(
            "How long does a student have to submit a show cause response?",
            requirements,
            parse,
        )
    )

    assert len(result.propositions) == 1

    proposition = result.propositions[0]

    assert tuple(
        item.text
        for item in proposition.relations
    ) == (
        "have",
        "submit",
    )

def test_extracts_relationless_information_request():
    requirements = MaterialQuestionRequirements(
        requirements=(
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "the admission requirements",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "university applicants",
            ),
        )
    )

    parse = QuestionParse(
        tokens=(
            ParsedToken(
                index=0,
                text="requirements",
                lemma="requirement",
                pos="NOUN",
                tag="NNS",
                dependency="nsubj",
                head_index=0,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=1,
                text="applicants",
                lemma="applicant",
                pos="NOUN",
                tag="NNS",
                dependency="pobj",
                head_index=0,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="the admission requirements",
                start_index=0,
                end_index=1,
                root_index=0,
            ),
            ParsedSpan(
                text="university applicants",
                start_index=1,
                end_index=2,
                root_index=1,
            ),
        ),
    )

    result = (
        DependencyMaterialPropositionExtractor()
        .extract(
            "What are the admission requirements for university applicants?",
            requirements,
            parse,
        )
    )

    proposition = result.propositions[0]

    assert (
        proposition.kind
        == MaterialPropositionKind.INFORMATION_REQUEST
    )

    assert tuple(
        item.text
        for item in proposition.subjects
    ) == (
        "university applicants",
    )

    assert tuple(
        item.text
        for item in proposition.objects
    ) == (
        "the admission requirements",
    )

def test_extracts_nested_student_context_as_scope():
    requirements = MaterialQuestionRequirements(
        requirements=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "have",
            ),
            _requirement(
                MaterialRequirementKind.RELATION,
                "submit",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "a student",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "Academic Progression Stage Three",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "a show cause response",
            ),
            _requirement(
                MaterialRequirementKind.REQUESTED_ATTRIBUTE,
                "How long",
            ),
        )
    )

    parse = QuestionParse(
        tokens=(
            ParsedToken(
                index=0,
                text="student",
                lemma="student",
                pos="NOUN",
                tag="NN",
                dependency="nsubj",
                head_index=6,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=1,
                text="at",
                lemma="at",
                pos="ADP",
                tag="IN",
                dependency="prep",
                head_index=0,
                is_stop=True,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=2,
                text="Academic",
                lemma="Academic",
                pos="PROPN",
                tag="NNP",
                dependency="compound",
                head_index=4,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=3,
                text="Progression",
                lemma="Progression",
                pos="PROPN",
                tag="NNP",
                dependency="compound",
                head_index=4,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=4,
                text="Stage",
                lemma="Stage",
                pos="PROPN",
                tag="NNP",
                dependency="pobj",
                head_index=1,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=5,
                text="Three",
                lemma="Three",
                pos="PROPN",
                tag="NNP",
                dependency="nummod",
                head_index=4,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=6,
                text="have",
                lemma="have",
                pos="VERB",
                tag="VBP",
                dependency="ROOT",
                head_index=6,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=7,
                text="submit",
                lemma="submit",
                pos="VERB",
                tag="VB",
                dependency="xcomp",
                head_index=6,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=8,
                text="response",
                lemma="response",
                pos="NOUN",
                tag="NN",
                dependency="dobj",
                head_index=7,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="a student",
                start_index=0,
                end_index=1,
                root_index=0,
            ),
            ParsedSpan(
                text="Academic Progression Stage",
                start_index=2,
                end_index=5,
                root_index=4,
            ),
            ParsedSpan(
                text="a show cause response",
                start_index=8,
                end_index=9,
                root_index=8,
            ),
        ),
    )

    result = (
        DependencyMaterialPropositionExtractor()
        .extract(
            (
                "How long does a student at "
                "Academic Progression Stage Three "
                "have to submit a show cause response?"
            ),
            requirements,
            parse,
        )
    )

    assert len(
        result.propositions
    ) == 1

    proposition = result.propositions[0]

    assert tuple(
        item.text
        for item in proposition.subjects
    ) == (
        "a student",
    )

    assert tuple(
        item.text
        for item in proposition.scopes
    ) == (
        "Academic Progression Stage Three",
    )

    assert tuple(
        item.text
        for item in proposition.relations
    ) == (
        "have",
        "submit",
    )

    assert tuple(
        item.text
        for item in proposition.objects
    ) == (
        "a show cause response",
    )

    assert tuple(
        item.text
        for item
        in proposition.requested_attributes
    ) == (
        "How long",
    )

def test_extracts_nested_prepositional_argument_context_as_scope():
    requirements = MaterialQuestionRequirements(
        requirements=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "apply",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "the maximum enrolment load",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "a student",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "Academic Progression Stage Two",
            ),
        )
    )

    parse = QuestionParse(
        tokens=(
            ParsedToken(
                index=0,
                text="load",
                lemma="load",
                pos="NOUN",
                tag="NN",
                dependency="nsubj",
                head_index=1,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=1,
                text="is",
                lemma="be",
                pos="AUX",
                tag="VBZ",
                dependency="ROOT",
                head_index=1,
                is_stop=True,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=2,
                text="applied",
                lemma="apply",
                pos="VERB",
                tag="VBN",
                dependency="acl",
                head_index=0,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=3,
                text="to",
                lemma="to",
                pos="ADP",
                tag="IN",
                dependency="prep",
                head_index=2,
                is_stop=True,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=4,
                text="student",
                lemma="student",
                pos="NOUN",
                tag="NN",
                dependency="pobj",
                head_index=3,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=5,
                text="at",
                lemma="at",
                pos="ADP",
                tag="IN",
                dependency="prep",
                head_index=4,
                is_stop=True,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=6,
                text="Academic",
                lemma="Academic",
                pos="PROPN",
                tag="NNP",
                dependency="compound",
                head_index=8,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=7,
                text="Progression",
                lemma="Progression",
                pos="PROPN",
                tag="NNP",
                dependency="compound",
                head_index=8,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=8,
                text="Stage",
                lemma="Stage",
                pos="PROPN",
                tag="NNP",
                dependency="pobj",
                head_index=5,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=9,
                text="Two",
                lemma="two",
                pos="NUM",
                tag="CD",
                dependency="nummod",
                head_index=8,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="the maximum enrolment load",
                start_index=0,
                end_index=1,
                root_index=0,
            ),
            ParsedSpan(
                text="a student",
                start_index=4,
                end_index=5,
                root_index=4,
            ),
            ParsedSpan(
                text="Academic Progression Stage",
                start_index=6,
                end_index=9,
                root_index=8,
            ),
        ),
    )

    result = (
        DependencyMaterialPropositionExtractor()
        .extract(
            (
                "What is the maximum enrolment load "
                "applied to a student at "
                "Academic Progression Stage Two?"
            ),
            requirements,
            parse,
        )
    )

    proposition = result.propositions[0]

    assert tuple(
        item.text
        for item in proposition.scopes
    ) == (
        "Academic Progression Stage Two",
    )

    assert (
        "a student"
        in tuple(
            item.text
            for item in proposition.objects
        )
    )

    assert (
        "Academic Progression Stage Two"
        not in tuple(
            item.text
            for item in proposition.objects
        )
    )

def test_splits_coordinated_relations_with_separate_subjects():
    requirements = MaterialQuestionRequirements(
        requirements=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "apply",
            ),
            _requirement(
                MaterialRequirementKind.RELATION,
                "extend",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "students",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "promotion",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "supervisors",
            ),
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "the deadline",
            ),
        )
    )

    parse = QuestionParse(
        tokens=(
            ParsedToken(
                index=0,
                text="students",
                lemma="student",
                pos="NOUN",
                tag="NNS",
                dependency="nsubj",
                head_index=1,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=1,
                text="apply",
                lemma="apply",
                pos="VERB",
                tag="VBP",
                dependency="ROOT",
                head_index=1,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=2,
                text="for",
                lemma="for",
                pos="ADP",
                tag="IN",
                dependency="prep",
                head_index=1,
                is_stop=True,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=3,
                text="promotion",
                lemma="promotion",
                pos="NOUN",
                tag="NN",
                dependency="pobj",
                head_index=2,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=4,
                text="and",
                lemma="and",
                pos="CCONJ",
                tag="CC",
                dependency="cc",
                head_index=6,
                is_stop=True,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=5,
                text="supervisors",
                lemma="supervisor",
                pos="NOUN",
                tag="NNS",
                dependency="nsubj",
                head_index=6,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=6,
                text="extend",
                lemma="extend",
                pos="VERB",
                tag="VBP",
                dependency="conj",
                head_index=1,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=7,
                text="deadline",
                lemma="deadline",
                pos="NOUN",
                tag="NN",
                dependency="dobj",
                head_index=6,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
        ),
        noun_phrases=(
            ParsedSpan(
                text="students",
                start_index=0,
                end_index=1,
                root_index=0,
            ),
            ParsedSpan(
                text="promotion",
                start_index=3,
                end_index=4,
                root_index=3,
            ),
            ParsedSpan(
                text="supervisors",
                start_index=5,
                end_index=6,
                root_index=5,
            ),
            ParsedSpan(
                text="the deadline",
                start_index=7,
                end_index=8,
                root_index=7,
            ),
        ),
    )

    result = (
        DependencyMaterialPropositionExtractor()
        .extract(
            (
                "Students apply for promotion and "
                "supervisors extend the deadline."
            ),
            requirements,
            parse,
        )
    )

    assert len(
        result.propositions
    ) == 2

    first = result.propositions[0]
    second = result.propositions[1]

    assert tuple(
        item.text
        for item in first.subjects
    ) == (
        "students",
    )

    assert tuple(
        item.text
        for item in first.relations
    ) == (
        "apply",
    )

    assert tuple(
        item.text
        for item in first.objects
    ) == (
        "promotion",
    )

    assert tuple(
        item.text
        for item in second.subjects
    ) == (
        "supervisors",
    )

    assert tuple(
        item.text
        for item in second.relations
    ) == (
        "extend",
    )

    assert tuple(
        item.text
        for item in second.objects
    ) == (
        "the deadline",
    )

def test_splits_multiple_requested_attributes_into_separate_propositions():
    requirements = MaterialQuestionRequirements(
        requirements=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "result",
            ),
            _requirement(
                MaterialRequirementKind.CONDITION,
                (
                    "when staff achieve "
                    "Professional Equivalence"
                ),
            ),
            _requirement(
                MaterialRequirementKind.REQUESTED_ATTRIBUTE,
                "employment benefits",
            ),
            _requirement(
                MaterialRequirementKind.REQUESTED_ATTRIBUTE,
                "salary changes",
            ),
            _requirement(
                MaterialRequirementKind.REQUESTED_ATTRIBUTE,
                "career progression rights",
            ),
        )
    )

    parse = QuestionParse(
        tokens=(
            ParsedToken(
                index=0,
                text="benefits",
                lemma="benefit",
                pos="NOUN",
                tag="NNS",
                dependency="nsubj",
                head_index=3,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=1,
                text="changes",
                lemma="change",
                pos="NOUN",
                tag="NNS",
                dependency="conj",
                head_index=0,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=2,
                text="rights",
                lemma="right",
                pos="NOUN",
                tag="NNS",
                dependency="conj",
                head_index=0,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
            ParsedToken(
                index=3,
                text="result",
                lemma="result",
                pos="VERB",
                tag="VBP",
                dependency="ROOT",
                head_index=3,
                is_stop=False,
                is_punct=False,
                is_alpha=True,
            ),
        ),
        noun_phrases=(),
    )

    result = (
        DependencyMaterialPropositionExtractor()
        .extract(
            (
                "What employment benefits, salary changes, "
                "and career progression rights result when "
                "staff achieve Professional Equivalence?"
            ),
            requirements,
            parse,
        )
    )

    assert len(
        result.propositions
    ) == 3

    assert tuple(
        proposition.requested_attributes[0].text
        for proposition
        in result.propositions
    ) == (
        "employment benefits",
        "salary changes",
        "career progression rights",
    )

    assert all(
        tuple(
            condition.text
            for condition
            in proposition.conditions
        )
        == (
            (
                "when staff achieve "
                "Professional Equivalence"
            ),
        )
        for proposition
        in result.propositions
    )

    assert all(
        tuple(
            relation.text
            for relation
            in proposition.relations
        )
        == (
            "result",
        )
        for proposition
        in result.propositions
    )
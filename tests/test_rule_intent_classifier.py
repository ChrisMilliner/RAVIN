import pytest
from backend.routing.models import (
    QuestionIntent,
)
from backend.routing.rule_intent_classifier import (
    RuleBasedQuestionIntentClassifier,
)

def test_classifies_clear_specific_question_as_focused():
    classifier = (
        RuleBasedQuestionIntentClassifier()
    )

    result = classifier.classify(
        "Who approves changes to academic dress?"
    )

    assert result is QuestionIntent.FOCUSED

def test_classifies_specific_show_cause_question_as_focused():
    classifier = (
        RuleBasedQuestionIntentClassifier()
    )

    result = classifier.classify(
        "How long does a student have to "
        "submit a show cause response?"
    )

    assert result is QuestionIntent.FOCUSED

def test_classifies_process_overview_as_broad():
    classifier = (
        RuleBasedQuestionIntentClassifier()
    )

    result = classifier.classify(
        "What happens when a student is not "
        "making satisfactory academic progress?"
    )

    assert result is QuestionIntent.BROAD

def test_classifies_how_does_overview_as_broad():
    classifier = (
        RuleBasedQuestionIntentClassifier()
    )

    result = classifier.classify(
        "How does La Trobe support admission "
        "for people experiencing disadvantage?"
    )

    assert result is QuestionIntent.BROAD

def test_classifies_multi_part_request_as_broad():
    classifier = (
        RuleBasedQuestionIntentClassifier()
    )

    result = classifier.classify(
        "What financial benefits, leave "
        "entitlements and salary changes apply "
        "when academic staff are promoted?"
    )

    assert result is QuestionIntent.BROAD

def test_classifies_missing_topic_as_ambiguous():
    classifier = (
        RuleBasedQuestionIntentClassifier()
    )

    result = classifier.classify(
        "What do I need to submit?"
    )

    assert result is QuestionIntent.AMBIGUOUS

def test_classifies_missing_reference_as_ambiguous():
    classifier = (
        RuleBasedQuestionIntentClassifier()
    )

    result = classifier.classify(
        "Who approves it?"
    )

    assert result is QuestionIntent.AMBIGUOUS

def test_normalizes_case_and_whitespace():
    classifier = (
        RuleBasedQuestionIntentClassifier()
    )

    result = classifier.classify(
        "  WHO   APPROVES changes to "
        "ACADEMIC DRESS?  "
    )

    assert result is QuestionIntent.FOCUSED

def test_classifies_at_each_stage_as_broad():
    classifier = (
        RuleBasedQuestionIntentClassifier()
    )

    result = classifier.classify(
        "What support is provided to students "
        "at each stage of academic progression?"
    )

    assert result is QuestionIntent.BROAD

def test_rejects_empty_question():
    classifier = (
        RuleBasedQuestionIntentClassifier()
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        classifier.classify("   ")

def test_classifies_bounded_compound_question_as_focused():
    classifier = (
        RuleBasedQuestionIntentClassifier()
    )

    result = classifier.classify(
        "Can fixed-term academic staff apply "
        "for promotion, and does a promotion "
        "extend their fixed-term appointment?"
    )

    assert result is QuestionIntent.FOCUSED

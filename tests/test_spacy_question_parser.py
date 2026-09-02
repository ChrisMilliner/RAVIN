from dataclasses import dataclass
from typing import cast
import pytest
from backend.routing.question_parser import (
    QuestionParse,
)
from backend.routing.spacy_question_parser import (
    SpacyPipeline,
    SpacyQuestionParseProvider,
    load_spacy_question_parse_provider,
)

@dataclass
class FakeToken:
    i: int
    text: str
    lemma_: str
    pos_: str
    tag_: str
    dep_: str
    head: "FakeToken | None"
    is_stop: bool
    is_punct: bool
    is_alpha: bool

@dataclass
class FakeChunk:
    text: str
    start: int
    end: int
    root: FakeToken

class FakeDoc:
    def __init__(
        self,
        tokens: tuple[
            FakeToken,
            ...
        ],
        noun_chunks: tuple[
            FakeChunk,
            ...
        ],
    ) -> None:
        self._tokens = tokens
        self.noun_chunks = noun_chunks

    def __iter__(
        self,
    ):
        return iter(
            self._tokens
        )

class FakePipeline:
    def __init__(
        self,
        doc: FakeDoc,
    ) -> None:
        self.doc = doc
        self.received_text: str | None = None

    def __call__(
        self,
        text: str,
    ) -> FakeDoc:
        self.received_text = text

        return self.doc

def _as_spacy_pipeline(
    pipeline: FakePipeline,
) -> SpacyPipeline:
    return cast(
        SpacyPipeline,
        pipeline,
    )

def _make_fake_doc(
) -> FakeDoc:
    students = FakeToken(
        i=0,
        text="Students",
        lemma_="student",
        pos_="NOUN",
        tag_="NNS",
        dep_="nsubj",
        head=None,
        is_stop=False,
        is_punct=False,
        is_alpha=True,
    )

    apply = FakeToken(
        i=1,
        text="apply",
        lemma_="apply",
        pos_="VERB",
        tag_="VBP",
        dep_="ROOT",
        head=None,
        is_stop=False,
        is_punct=False,
        is_alpha=True,
    )

    question_mark = FakeToken(
        i=2,
        text="?",
        lemma_="?",
        pos_="PUNCT",
        tag_=".",
        dep_="punct",
        head=None,
        is_stop=False,
        is_punct=True,
        is_alpha=False,
    )

    students.head = apply
    apply.head = apply
    question_mark.head = apply

    return FakeDoc(
        tokens=(
            students,
            apply,
            question_mark,
        ),
        noun_chunks=(
            FakeChunk(
                text="Students",
                start=0,
                end=1,
                root=students,
            ),
        ),
    )

def test_adapter_converts_tokens_to_question_parse():
    pipeline = FakePipeline(
        _make_fake_doc()
    )

    provider = (
        SpacyQuestionParseProvider(
            pipeline=_as_spacy_pipeline(
                pipeline
            ),
        )
    )

    result = provider.parse(
        "Students apply?"
    )

    assert isinstance(
        result,
        QuestionParse,
    )

    assert len(
        result.tokens
    ) == 3

    subject = result.tokens[0]

    assert subject.index == 0
    assert subject.text == "Students"
    assert subject.lemma == "student"
    assert subject.pos == "NOUN"
    assert subject.tag == "NNS"
    assert subject.dependency == "nsubj"
    assert subject.head_index == 1
    assert not subject.is_stop
    assert not subject.is_punct
    assert subject.is_alpha

def test_adapter_converts_noun_phrases():
    provider = (
        SpacyQuestionParseProvider(
            pipeline=_as_spacy_pipeline(
                FakePipeline(
                    _make_fake_doc()
                )
            ),
        )
    )

    result = provider.parse(
        "Students apply?"
    )

    assert len(
        result.noun_phrases
    ) == 1

    noun_phrase = (
        result.noun_phrases[0]
    )

    assert noun_phrase.text == "Students"
    assert noun_phrase.start_index == 0
    assert noun_phrase.end_index == 1
    assert noun_phrase.root_index == 0

def test_adapter_passes_question_to_pipeline():
    pipeline = FakePipeline(
        _make_fake_doc()
    )

    provider = (
        SpacyQuestionParseProvider(
            pipeline=_as_spacy_pipeline(
                pipeline
            ),
        )
    )

    provider.parse(
        "Can students apply?"
    )

    assert pipeline.received_text == (
        "Can students apply?"
    )

def test_loader_rejects_empty_model_name():
    with pytest.raises(
        ValueError,
        match=(
            "spaCy model name "
            "cannot be empty."
        ),
    ):
        load_spacy_question_parse_provider(
            "   "
        )

def test_loader_uses_requested_model(
    monkeypatch,
):
    pipeline = FakePipeline(
        _make_fake_doc()
    )

    loaded_models: list[str] = []

    def fake_spacy_load(
        model_name: str,
    ):
        loaded_models.append(
            model_name
        )

        return pipeline

    monkeypatch.setattr(
        (
            "backend.routing."
            "spacy_question_parser."
            "spacy.load"
        ),
        fake_spacy_load,
    )

    provider = (
        load_spacy_question_parse_provider(
            "replaceable-parser-model"
        )
    )

    result = provider.parse(
        "Students apply?"
    )

    assert loaded_models == [
        "replaceable-parser-model"
    ]

    assert isinstance(
        result,
        QuestionParse,
    )
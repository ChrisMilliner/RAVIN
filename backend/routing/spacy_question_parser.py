"""
Provide the spaCy adapter for RAVIN question parsing.

This module translates spaCy token, span, and dependency information
into RAVIN's neutral question-parse structures. Library-specific model
loading and parsing remain confined to this adapter.

spaCy provides linguistic analysis only. RAVIN defines the higher-level
material requirements, proposition logic, recovery strategies, evidence
assessment, and routing decisions.
"""

from typing import Protocol
import spacy
from spacy.tokens import Doc
from backend.routing.question_parser import (
    ParsedSpan,
    ParsedToken,
    QuestionParse,
)

class SpacyPipeline(
    Protocol
):
    def __call__(
        self,
        text: str,
    ) -> Doc:
        ...

class SpacyQuestionParseProvider:
    def __init__(
        self,
        pipeline: SpacyPipeline,
    ) -> None:
        self._pipeline = pipeline

    def parse(
        self,
        question: str,
    ) -> QuestionParse:
        doc = self._pipeline(
            question
        )

        tokens = tuple(
            ParsedToken(
                index=token.i,
                text=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                tag=token.tag_,
                dependency=token.dep_,
                head_index=token.head.i,
                is_stop=token.is_stop,
                is_punct=token.is_punct,
                is_alpha=token.is_alpha,
            )
            for token in doc
        )

        noun_phrases = tuple(
            ParsedSpan(
                text=chunk.text,
                start_index=chunk.start,
                end_index=chunk.end,
                root_index=chunk.root.i,
            )
            for chunk in doc.noun_chunks
        )

        return QuestionParse(
            tokens=tokens,
            noun_phrases=noun_phrases,
        )

def load_spacy_question_parse_provider(
    model_name: str,
) -> SpacyQuestionParseProvider:
    model_name = model_name.strip()

    if not model_name:
        raise ValueError(
            "spaCy model name cannot be empty."
        )

    pipeline = spacy.load(
        model_name
    )

    return SpacyQuestionParseProvider(
        pipeline=pipeline,
    )
"""
Recover selected malformed question structures with deterministic rules.

This module contains RAVIN-defined recovery strategies for specific
dependency structures that the configured parser may analyse
incorrectly. Recovery is attempted only when the primary parse is
considered unreliable and the supported structural pattern is clear.

These recovery labels and strategies belong to RAVIN and are not spaCy
dependency labels or generative-model decisions.
"""

from dataclasses import replace
from backend.routing.question_parser import (
    ParsedSpan,
    ParsedToken,
    QuestionParse,
    QuestionParseReliability,
    QuestionParseResult,
    is_question_parse_suspicious,
)

INTERROGATIVE_TAGS = frozenset(
    {
        "WDT",
        "WP",
        "WP$",
    }
)

class DeterministicQuestionStructureRecoveryProvider:
    def recover(
        self,
        question: str,
        parse_result: QuestionParseResult,
    ) -> QuestionParse | None:
        if (
            parse_result.reliability
            != QuestionParseReliability.UNRESOLVED
        ):
            return None

        parses = tuple(
            parse
            for parse in (
                parse_result.fallback,
                parse_result.primary,
            )
            if parse is not None
        )

        strategies = (
            self._recover_do_supported_nominal_predicate,
            self._recover_modal_nominal_root_with_later_base_verb,
            self._recover_bare_root_object_before_conjoined_verb,
        )

        for parse in parses:
            for strategy in strategies:
                recovered = strategy(
                    parse
                )

                if recovered is None:
                    continue

                if is_question_parse_suspicious(
                    recovered
                ):
                    continue

                return recovered

        return None

    def _recover_do_supported_nominal_predicate(
        self,
        parse: QuestionParse,
    ) -> QuestionParse | None:
        roots = parse.roots

        if len(roots) != 1:
            return None

        root = roots[0]

        if root.pos != "NOUN":
            return None

        children = self._children_of(
            parse,
            root.index,
        )

        has_do_auxiliary = any(
            (
                child.pos == "AUX"
                and child.lemma.casefold()
                == "do"
                and child.dependency
                in {
                    "aux",
                    "auxpass",
                }
            )
            for child in children
        )

        has_subject = any(
            child.dependency
            in {
                "nsubj",
                "nsubjpass",
            }
            for child in children
        )

        has_object = any(
            child.dependency
            in {
                "dobj",
                "obj",
            }
            for child in children
        )

        if not (
            has_do_auxiliary
            and has_subject
            and has_object
        ):
            return None

        tokens = tuple(
            (
                replace(
                    token,
                    pos="VERB",
                    tag="VB",
                )
                if token.index == root.index
                else token
            )
            for token in parse.tokens
        )

        return QuestionParse(
            tokens=tokens,
            noun_phrases=parse.noun_phrases,
        )

    def _recover_modal_nominal_root_with_later_base_verb(
        self,
        parse: QuestionParse,
    ) -> QuestionParse | None:
        roots = parse.roots

        if len(roots) != 1:
            return None

        root = roots[0]

        if root.pos != "NOUN":
            return None

        root_children = self._children_of(
            parse,
            root.index,
        )

        modal_auxiliaries = tuple(
            child
            for child in root_children
            if (
                child.pos == "AUX"
                and child.tag == "MD"
                and child.dependency == "aux"
            )
        )

        if not modal_auxiliaries:
            return None

        candidates = tuple(
            token
            for token in parse.tokens
            if (
                token.index > root.index
                and token.pos == "VERB"
                and token.tag == "VB"
                and token.dependency
                not in {
                    "advcl",
                    "relcl",
                    "acl",
                }
            )
        )

        if len(candidates) != 1:
            return None

        candidate = candidates[0]

        tokens_by_index = {
            token.index: token
            for token in parse.tokens
        }

        candidate_head = (
            tokens_by_index.get(
                candidate.head_index
            )
        )

        phrase_head = None

        if (
            candidate_head is not None
            and candidate_head.pos == "ADP"
            and candidate_head.dependency
            == "prep"
            and candidate_head.head_index
            == root.index
        ):
            nominal_tokens = tuple(
                token
                for token in parse.tokens
                if (
                    candidate_head.index
                    < token.index
                    < candidate.index
                    and token.pos
                    in {
                        "NOUN",
                        "PROPN",
                    }
                )
            )

            if nominal_tokens:
                phrase_head = (
                    nominal_tokens[-1]
                )

        replacements: dict[
            int,
            ParsedToken,
        ] = {}

        replacements[
            candidate.index
        ] = replace(
            candidate,
            dependency="ROOT",
            head_index=candidate.index,
        )

        replacements[
            root.index
        ] = replace(
            root,
            dependency="nsubj",
            head_index=candidate.index,
        )

        for modal in modal_auxiliaries:
            replacements[
                modal.index
            ] = replace(
                modal,
                dependency="aux",
                head_index=candidate.index,
            )

        if (
            phrase_head is not None
            and candidate_head is not None
        ):
            replacements[
                phrase_head.index
            ] = replace(
                phrase_head,
                dependency="pobj",
                head_index=(
                    candidate_head.index
                ),
            )

            for token in parse.tokens:
                if not (
                    candidate_head.index
                    < token.index
                    < candidate.index
                ):
                    continue

                if token.index == (
                    phrase_head.index
                ):
                    continue

                if (
                    token.head_index
                    == candidate.index
                ):
                    replacements[
                        token.index
                    ] = replace(
                        token,
                        head_index=(
                            phrase_head.index
                        ),
                    )

        tokens = tuple(
            replacements.get(
                token.index,
                token,
            )
            for token in parse.tokens
        )

        noun_phrases = (
            self._repair_modal_subject_span(
                parse,
                tokens,
                root,
                modal_auxiliaries,
            )
        )

        if (
            phrase_head is not None
            and candidate_head is not None
        ):
            noun_phrases = (
                self._ensure_span(
                    tokens,
                    noun_phrases,
                    start_index=(
                        candidate_head.index
                        + 1
                    ),
                    end_index=(
                        candidate.index
                    ),
                    root_index=(
                        phrase_head.index
                    ),
                )
            )

        return QuestionParse(
            tokens=tokens,
            noun_phrases=noun_phrases,
        )

    def _recover_bare_root_object_before_conjoined_verb(
        self,
        parse: QuestionParse,
    ) -> QuestionParse | None:
        roots = parse.roots

        if len(roots) != 1:
            return None

        root = roots[0]

        if (
            root.pos != "VERB"
            or root.tag != "VB"
        ):
            return None

        root_children = self._children_of(
            parse,
            root.index,
        )

        has_auxiliary = any(
            child.dependency
            in {
                "aux",
                "auxpass",
            }
            for child in root_children
        )

        if has_auxiliary:
            return None

        objects = tuple(
            child
            for child in root_children
            if child.dependency
            in {
                "dobj",
                "obj",
            }
        )

        candidates = tuple(
            token
            for token in parse.tokens
            if (
                token.index > root.index
                and token.pos == "VERB"
                and token.dependency == "conj"
                and token.head_index
                == root.index
            )
        )

        if (
            len(objects) != 1
            or len(candidates) != 1
        ):
            return None

        object_token = objects[0]
        candidate = candidates[0]

        if (
            object_token.index
            != root.index + 1
        ):
            return None

        requested_spans = tuple(
            span
            for span in parse.noun_phrases
            if (
                span.root_index
                < root.index
                and self._span_has_interrogative(
                    parse,
                    span,
                )
            )
        )

        if len(requested_spans) != 1:
            return None

        requested_span = (
            requested_spans[0]
        )

        requested_root = (
            self._token_by_index(
                parse,
                requested_span.root_index,
            )
        )

        if requested_root is None:
            return None

        replacements: dict[
            int,
            ParsedToken,
        ] = {
            candidate.index: replace(
                candidate,
                dependency="ROOT",
                head_index=candidate.index,
            ),
            root.index: replace(
                root,
                pos="NOUN",
                tag="NN",
                dependency="compound",
                head_index=object_token.index,
            ),
            object_token.index: replace(
                object_token,
                dependency="conj",
                head_index=(
                    requested_root.index
                ),
            ),
            requested_root.index: replace(
                requested_root,
                dependency="nsubj",
                head_index=candidate.index,
            ),
        }

        tokens = tuple(
            replacements.get(
                token.index,
                token,
            )
            for token in parse.tokens
        )

        noun_phrases: list[
            ParsedSpan
        ] = []

        object_span_found = False

        for span in parse.noun_phrases:
            if (
                span.root_index
                == object_token.index
            ):
                noun_phrases.append(
                    ParsedSpan(
                        text=self._span_text(
                            tokens,
                            root.index,
                            span.end_index,
                        ),
                        start_index=root.index,
                        end_index=span.end_index,
                        root_index=(
                            object_token.index
                        ),
                    )
                )

                object_span_found = True
                continue

            noun_phrases.append(
                span
            )

        if not object_span_found:
            noun_phrases.append(
                ParsedSpan(
                    text=self._span_text(
                        tokens,
                        root.index,
                        object_token.index
                        + 1,
                    ),
                    start_index=root.index,
                    end_index=(
                        object_token.index
                        + 1
                    ),
                    root_index=(
                        object_token.index
                    ),
                )
            )

        noun_phrases.sort(
            key=lambda span: (
                span.start_index,
                span.end_index,
            )
        )

        return QuestionParse(
            tokens=tokens,
            noun_phrases=tuple(
                noun_phrases
            ),
        )

    def _children_of(
        self,
        parse: QuestionParse,
        head_index: int,
    ) -> tuple[ParsedToken, ...]:
        return tuple(
            token
            for token in parse.tokens
            if (
                token.index != head_index
                and token.head_index
                == head_index
            )
        )

    def _token_by_index(
        self,
        parse: QuestionParse,
        index: int,
    ) -> ParsedToken | None:
        return next(
            (
                token
                for token in parse.tokens
                if token.index == index
            ),
            None,
        )

    def _span_has_interrogative(
        self,
        parse: QuestionParse,
        span: ParsedSpan,
    ) -> bool:
        return any(
            token.tag
            in INTERROGATIVE_TAGS
            for token in parse.tokens
            if (
                span.start_index
                <= token.index
                < span.end_index
            )
        )

    def _repair_modal_subject_span(
        self,
        parse: QuestionParse,
        tokens: tuple[
            ParsedToken,
            ...
        ],
        root: ParsedToken,
        modal_auxiliaries: tuple[
            ParsedToken,
            ...
        ],
    ) -> tuple[ParsedSpan, ...]:
        modal_indexes = {
            modal.index
            for modal in modal_auxiliaries
        }

        spans: list[
            ParsedSpan
        ] = []

        for span in parse.noun_phrases:
            if (
                span.root_index
                != root.index
            ):
                spans.append(
                    span
                )
                continue

            new_start = span.start_index

            while (
                new_start
                < span.end_index
                and new_start
                in modal_indexes
            ):
                new_start += 1

            if new_start >= (
                span.end_index
            ):
                continue

            spans.append(
                ParsedSpan(
                    text=self._span_text(
                        tokens,
                        new_start,
                        span.end_index,
                    ),
                    start_index=new_start,
                    end_index=span.end_index,
                    root_index=span.root_index,
                )
            )

        return tuple(
            spans
        )

    def _ensure_span(
        self,
        tokens: tuple[
            ParsedToken,
            ...
        ],
        spans: tuple[
            ParsedSpan,
            ...
        ],
        *,
        start_index: int,
        end_index: int,
        root_index: int,
    ) -> tuple[ParsedSpan, ...]:
        if any(
            span.root_index == root_index
            for span in spans
        ):
            return spans

        result = list(
            spans
        )

        result.append(
            ParsedSpan(
                text=self._span_text(
                    tokens,
                    start_index,
                    end_index,
                ),
                start_index=start_index,
                end_index=end_index,
                root_index=root_index,
            )
        )

        result.sort(
            key=lambda span: (
                span.start_index,
                span.end_index,
            )
        )

        return tuple(
            result
        )

    def _span_text(
        self,
        tokens: tuple[
            ParsedToken,
            ...
        ],
        start_index: int,
        end_index: int,
    ) -> str:
        return " ".join(
            token.text
            for token in tokens
            if (
                start_index
                <= token.index
                < end_index
                and not token.is_punct
            )
        ).strip()
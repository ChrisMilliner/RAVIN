from backend.routing.material_requirements import (
    MaterialQuestionRequirements,
    MaterialRequirement,
    MaterialRequirementKind,
)
from backend.routing.question_parser import (
    ParsedSpan,
    ParsedToken,
    QuestionParse,
    QuestionParserService,
)

CLAUSE_DEPENDENCIES = frozenset(
    {
        "advcl",
        "relcl",
        "acl",
        "pcomp",
    }
)
INTERROGATIVE_TAGS = frozenset(
    {
        "WDT",
        "WP",
        "WP$",
    }
)

class DependencyMaterialRequirementExtractor:
    def __init__(
        self,
        parser: QuestionParserService,
    ) -> None:
        self._parser = parser

    def extract(
        self,
        question: str,
    ) -> MaterialQuestionRequirements:
        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        parse_result = self._parser.parse(
            question
        )

        requirements = list(
            self._extract_from_parse(
                parse_result.primary
            )
        )

        if parse_result.fallback is not None:
            requirements.extend(
                self._extract_from_parse(
                    parse_result.fallback
                )
            )

        unique_requirements = (
            self._deduplicate_requirements(
                tuple(requirements)
            )
        )

        if not unique_requirements:
            raise ValueError(
                "Question parse did not produce "
                "material requirements."
            )

        return MaterialQuestionRequirements(
            requirements=unique_requirements
        )

    def _extract_from_parse(
        self,
        parse: QuestionParse,
    ) -> tuple[
        MaterialRequirement,
        ...
    ]:
        requirements: list[
            MaterialRequirement
        ] = []

        tokens_by_index = {
            token.index: token
            for token in parse.tokens
        }

        requested_roots = (
            self._requested_attribute_roots(
                parse
            )
        )

        requested_adverbial_heads = (
            self._requested_adverbial_head_indexes(
                parse
            )
        )

        for root in parse.roots:
            if root.pos == "VERB":
                requirements.append(
                    MaterialRequirement(
                        kind=(
                            MaterialRequirementKind
                            .RELATION
                        ),
                        text=root.lemma,
                    )
                )

        for token in parse.tokens:
            if token.dependency == "neg":
                requirements.append(
                    MaterialRequirement(
                        kind=(
                            MaterialRequirementKind
                            .NEGATION
                        ),
                        text=token.lemma,
                    )
                )

            if (
                token.tag == "MD"
                and token.pos == "AUX"
            ):
                requirements.append(
                    MaterialRequirement(
                        kind=(
                            MaterialRequirementKind
                            .MODALITY
                        ),
                        text=token.lemma,
                    )
                )

            if (
                token.dependency == "advmod"
                and not token.is_stop
                and token.is_alpha
                and token.index
                not in requested_adverbial_heads
            ):
                requirements.append(
                    MaterialRequirement(
                        kind=(
                            MaterialRequirementKind
                            .QUALIFIER
                        ),
                        text=token.text,
                    )
                )

            if (
                token.dependency
                in CLAUSE_DEPENDENCIES
            ):
                clause_text = (
                    self._subtree_text(
                        parse,
                        token.index,
                    )
                )

                if clause_text:
                    requirements.append(
                        MaterialRequirement(
                            kind=(
                                MaterialRequirementKind
                                .CONDITION
                            ),
                            text=clause_text,
                        )
                    )

        for span in parse.noun_phrases:
            span_root = tokens_by_index.get(
                span.root_index
            )

            if span_root is None:
                continue

            if (
                span.root_index
                in requested_roots
            ):
                requirement_text = (
                    self._requested_span_text(
                        parse,
                        span,
                    )
                )

                if requirement_text:
                    requirements.append(
                        MaterialRequirement(
                            kind=(
                                MaterialRequirementKind
                                .REQUESTED_ATTRIBUTE
                            ),
                            text=requirement_text,
                        )
                    )

                continue

            if self._span_is_only_interrogative(
                parse,
                span,
            ):
                continue

            requirements.append(
                MaterialRequirement(
                    kind=(
                        MaterialRequirementKind
                        .CONCEPT
                    ),
                    text=self._span_text(
                        parse,
                        span,
                    ),
                )
            )

        requested_adverbial = (
            self._requested_adverbial_text(
                parse
            )
        )

        if requested_adverbial is not None:
            requirements.append(
                MaterialRequirement(
                    kind=(
                        MaterialRequirementKind
                        .REQUESTED_ATTRIBUTE
                    ),
                    text=requested_adverbial,
                )
            )

        return tuple(requirements)

    def _requested_attribute_roots(
        self,
        parse: QuestionParse,
    ) -> set[int]:
        requested_roots: set[int] = set()

        for span in parse.noun_phrases:
            span_tokens = self._span_tokens(
                parse,
                span,
            )

            has_interrogative = any(
                token.tag
                in INTERROGATIVE_TAGS
                for token in span_tokens
            )

            has_requested_content = any(
                (
                    token.tag
                    not in INTERROGATIVE_TAGS
                    and token.is_alpha
                )
                for token in span_tokens
            )

            if (
                has_interrogative
                and has_requested_content
            ):
                requested_roots.add(
                    span.root_index
                )

        changed = True

        while changed:
            changed = False

            for token in parse.tokens:
                if (
                    token.dependency == "conj"
                    and token.head_index
                    in requested_roots
                    and token.index
                    not in requested_roots
                ):
                    requested_roots.add(
                        token.index
                    )

                    changed = True

        return requested_roots

    def _requested_span_text(
        self,
        parse: QuestionParse,
        span: ParsedSpan,
    ) -> str:
        tokens = tuple(
            token
            for token in self._span_tokens(
                parse,
                span,
            )
            if (
                token.tag
                not in INTERROGATIVE_TAGS
                and token.is_alpha
            )
        )

        if tokens:
            return " ".join(
                token.text
                for token in tokens
            )

        return ""

    def _requested_adverbial_head_indexes(
        self,
        parse: QuestionParse,
    ) -> set[int]:
        tokens_by_index = {
            token.index: token
            for token in parse.tokens
        }

        head_indexes: set[int] = set()

        for token in parse.tokens:
            if (
                token.tag != "WRB"
                or token.dependency != "advmod"
            ):
                continue

            head = tokens_by_index.get(
                token.head_index
            )

            if (
                head is None
                or head.pos not in (
                    "ADV",
                    "ADJ",
                )
            ):
                continue

            head_indexes.add(
                head.index
            )

        return head_indexes

    def _requested_adverbial_text(
        self,
        parse: QuestionParse,
    ) -> str | None:
        tokens_by_index = {
            token.index: token
            for token in parse.tokens
        }

        for token in parse.tokens:
            if (
                token.tag != "WRB"
                or token.dependency != "advmod"
            ):
                continue

            head = tokens_by_index.get(
                token.head_index
            )

            if (
                head is None
                or head.pos not in (
                    "ADV",
                    "ADJ",
                )
            ):
                continue

            return (
                f"{token.text} "
                f"{head.text}"
            )

        return None

    def _span_is_only_interrogative(
        self,
        parse: QuestionParse,
        span: ParsedSpan,
    ) -> bool:
        tokens = self._span_tokens(
            parse,
            span,
        )

        return bool(
            tokens
        ) and all(
            (
                token.tag
                in INTERROGATIVE_TAGS
                or not token.is_alpha
            )
            for token in tokens
        )

    def _span_tokens(
        self,
        parse: QuestionParse,
        span: ParsedSpan,
    ) -> tuple[
        ParsedToken,
        ...
    ]:
        return tuple(
            token
            for token in parse.tokens
            if (
                span.start_index
                <= token.index
                < span.end_index
            )
        )

    def _span_text(
        self,
        parse: QuestionParse,
        span: ParsedSpan,
    ) -> str:
        return " ".join(
            token.text
            for token in self._span_tokens(
                parse,
                span,
            )
            if not token.is_punct
        ).strip()

    def _subtree_text(
        self,
        parse: QuestionParse,
        root_index: int,
    ) -> str:
        descendant_indexes = {
            root_index
        }

        changed = True

        while changed:
            changed = False

            for token in parse.tokens:
                if (
                    token.index
                    not in descendant_indexes
                    and token.head_index
                    in descendant_indexes
                ):
                    descendant_indexes.add(
                        token.index
                    )

                    changed = True

        return " ".join(
            token.text
            for token in parse.tokens
            if (
                token.index
                in descendant_indexes
                and not token.is_punct
            )
        ).strip()

    def _deduplicate_requirements(
        self,
        requirements: tuple[
            MaterialRequirement,
            ...
        ],
    ) -> tuple[
        MaterialRequirement,
        ...
    ]:
        unique: list[
            MaterialRequirement
        ] = []

        seen: set[
            tuple[
                MaterialRequirementKind,
                str,
            ]
        ] = set()

        for requirement in requirements:
            key = (
                requirement.kind,
                requirement.text.casefold(),
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            unique.append(
                requirement
            )

        return tuple(
            unique
        )

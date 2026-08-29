"""
Extract material question propositions from resolved dependency structure.

This module converts resolved grammatical structure into the factual
propositions that policy evidence must cover for RAVIN to treat a
question as answerable.

Extraction is deterministic and uses the shared question-structure
resolution result. It does not ask a generative model to infer what
evidence is required.
"""

from dataclasses import replace
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
    ParsedSpan,
    ParsedToken,
    QuestionParse,
)

class DependencyMaterialPropositionExtractor:
    """Derive material propositions from resolved dependency structure.
    """

    def extract(
        self,
        question: str,
        requirements: MaterialQuestionRequirements,
        parse: QuestionParse,
    ) -> MaterialQuestionPropositions:
        """Extract the independent material propositions that evidence must cover.
        """
        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        relations = self._requirements_of_kind(
            requirements,
            MaterialRequirementKind.RELATION,
        )

        if not relations:
            result = self._extract_information_request(
                requirements,
                parse,
            )
        else:
            result = self._extract_relational(
                requirements,
                parse,
                relations,
            )

        return (
            self._expand_requested_attribute_propositions(
                result
            )
        )

    def _expand_requested_attribute_propositions(
        self,
        propositions: MaterialQuestionPropositions,
    ) -> MaterialQuestionPropositions:
        expanded: list[
            MaterialProposition
        ] = []

        for proposition in propositions.propositions:
            if (
                len(
                    proposition.requested_attributes
                )
                <= 1
            ):
                expanded.append(
                    proposition
                )
                continue

            for requested_attribute in (
                proposition.requested_attributes
            ):
                expanded.append(
                    replace(
                        proposition,
                        requested_attributes=(
                            requested_attribute,
                        ),
                    )
                )

        return MaterialQuestionPropositions(
            propositions=tuple(
                expanded
            )
        )

    def _extract_information_request(
        self,
        requirements: MaterialQuestionRequirements,
        parse: QuestionParse,
    ) -> MaterialQuestionPropositions:
        concepts = self._requirements_of_kind(
            requirements,
            MaterialRequirementKind.CONCEPT,
        )

        requested = self._requirements_of_kind(
            requirements,
            MaterialRequirementKind.REQUESTED_ATTRIBUTE,
        )

        subjects: list[
            MaterialRequirement
        ] = []

        objects: list[
            MaterialRequirement
        ] = []

        for requirement in concepts:
            span = self._matching_noun_phrase(
                requirement,
                parse,
            )

            if span is None:
                objects.append(
                    requirement
                )
                continue

            root = self._token_by_index(
                parse,
                span.root_index,
            )

            if root.dependency in {
                "pobj",
                "obj",
                "dobj",
            }:
                subjects.append(
                    requirement
                )
            else:
                objects.append(
                    requirement
                )

        return MaterialQuestionPropositions(
            propositions=(
                MaterialProposition(
                    kind=(
                        MaterialPropositionKind.INFORMATION_REQUEST
                    ),
                    subjects=tuple(
                        subjects
                    ),
                    objects=tuple(
                        objects
                    ),
                    requested_attributes=requested,
                ),
            )
        )

    def _extract_relational(
        self,
        requirements: MaterialQuestionRequirements,
        parse: QuestionParse,
        relations: tuple[
            MaterialRequirement,
            ...
        ],
    ) -> MaterialQuestionPropositions:
        independent_groups = (
            self._independent_relation_groups(
                relations,
                parse,
            )
        )

        if len(
            independent_groups
        ) > 1:
            return (
                self._extract_independent_relation_groups(
                    requirements,
                    parse,
                    independent_groups,
                )
            )
        relation_tokens = tuple(
            token
            for relation in relations
            for token in (
                self._matching_relation_token(
                    relation,
                    parse,
                ),
            )
            if token is not None
        )

        primary_relation_tokens = self._primary_relation_chain(
            relation_tokens,
            parse,
        )

        active_relation_indexes = {
            token.index
            for token in primary_relation_tokens
        }

        concepts = self._requirements_of_kind(
            requirements,
            MaterialRequirementKind.CONCEPT,
        )

        subjects: list[
            MaterialRequirement
        ] = []

        scopes: list[
            MaterialRequirement
        ] = []

        objects: list[
            MaterialRequirement
        ] = []

        matched_concepts = tuple(
            (
                requirement,
                self._matching_noun_phrase(
                    requirement,
                    parse,
                ),
            )
            for requirement in concepts
        )

        direct_subject_root_indexes = {
            span.root_index
            for _, span in matched_concepts
            if (
                span is not None
                and self._is_direct_subject_of_relation_chain(
                    self._token_by_index(
                        parse,
                        span.root_index,
                    ),
                    active_relation_indexes,
                )
            )
        }

        prepositional_argument_root_indexes = {
            span.root_index
            for _, span in matched_concepts
            if (
                span is not None
                and self._is_prepositional_argument_of_relation_chain(
                    self._token_by_index(
                        parse,
                        span.root_index,
                    ),
                    active_relation_indexes,
                    parse,
                )
            )
        }

        context_parent_root_indexes = (
            direct_subject_root_indexes
            | prepositional_argument_root_indexes
        )

        for (
            requirement,
            span,
        ) in matched_concepts:
            if span is None:
                objects.append(
                    requirement
                )
                continue

            root = self._token_by_index(
                parse,
                span.root_index,
            )

            if (
                root.index
                in direct_subject_root_indexes
            ):
                subjects.append(
                    requirement
                )
                continue

            if self._is_nested_under_root(
                root,
                context_parent_root_indexes,
                parse,
            ):
                scopes.append(
                    requirement
                )
                continue

            if self._is_subject_of_relation_chain(
                root,
                active_relation_indexes,
                parse,
            ):
                subjects.append(
                    requirement
                )
            else:
                objects.append(
                    requirement
                )

        return MaterialQuestionPropositions(
            propositions=(
                MaterialProposition(
                    kind=(
                        MaterialPropositionKind.RELATIONAL
                    ),
                    subjects=tuple(
                        subjects
                    ),
                    relations=relations,
                    objects=tuple(
                        objects
                    ),
                    scopes=tuple(
                        scopes
                    ),
                    qualifiers=(
                        self._requirements_of_kind(
                            requirements,
                            MaterialRequirementKind.QUALIFIER,
                        )
                    ),
                    conditions=(
                        self._requirements_of_kind(
                            requirements,
                            MaterialRequirementKind.CONDITION,
                        )
                    ),
                    modalities=(
                        self._requirements_of_kind(
                            requirements,
                            MaterialRequirementKind.MODALITY,
                        )
                    ),
                    negations=(
                        self._requirements_of_kind(
                            requirements,
                            MaterialRequirementKind.NEGATION,
                        )
                    ),
                    requested_attributes=(
                        self._requirements_of_kind(
                            requirements,
                            MaterialRequirementKind.REQUESTED_ATTRIBUTE,
                        )
                    ),
                ),
            )
        )

    def _extract_independent_relation_groups(
        self,
        requirements: MaterialQuestionRequirements,
        parse: QuestionParse,
        relation_groups: tuple[
            tuple[
                MaterialRequirement,
                ...
            ],
            ...
        ],
    ) -> MaterialQuestionPropositions:
        concepts = self._requirements_of_kind(
            requirements,
            MaterialRequirementKind.CONCEPT,
        )

        all_relation_indexes = {
            token.index
            for group in relation_groups
            for relation in group
            for token in (
                self._matching_relation_token(
                    relation,
                    parse,
                ),
            )
            if token is not None
        }

        propositions = []

        for group in relation_groups:
            group_tokens = tuple(
                token
                for relation in group
                for token in (
                    self._matching_relation_token(
                        relation,
                        parse,
                    ),
                )
                if token is not None
            )

            group_indexes = {
                token.index
                for token in group_tokens
            }

            group_concepts = tuple(
                requirement
                for requirement in concepts
                if self._concept_belongs_to_relation_group(
                    requirement,
                    group_indexes,
                    all_relation_indexes,
                    parse,
                )
            )

            subjects = []
            objects = []

            for requirement in group_concepts:
                span = self._matching_noun_phrase(
                    requirement,
                    parse,
                )

                if span is None:
                    objects.append(
                        requirement
                    )
                    continue

                root = self._token_by_index(
                    parse,
                    span.root_index,
                )

                if (
                    self._is_direct_subject_of_relation_chain(
                        root,
                        group_indexes,
                    )
                ):
                    subjects.append(
                        requirement
                    )
                else:
                    objects.append(
                        requirement
                    )

            propositions.append(
                MaterialProposition(
                    kind=(
                        MaterialPropositionKind.RELATIONAL
                    ),
                    subjects=tuple(
                        subjects
                    ),
                    relations=group,
                    objects=tuple(
                        objects
                    ),
                )
            )

        return MaterialQuestionPropositions(
            propositions=tuple(
                propositions
            )
        )

    def _concept_belongs_to_relation_group(
        self,
        requirement: MaterialRequirement,
        group_indexes: set[int],
        all_relation_indexes: set[int],
        parse: QuestionParse,
    ) -> bool:
        span = self._matching_noun_phrase(
            requirement,
            parse,
        )

        if span is None:
            return False

        current = self._token_by_index(
            parse,
            span.root_index,
        )

        visited: set[int] = set()

        while (
            current.index
            not in visited
        ):
            visited.add(
                current.index
            )

            if (
                current.head_index
                in group_indexes
            ):
                return True

            if (
                current.head_index
                in all_relation_indexes
            ):
                return False

            if (
                current.head_index
                == current.index
            ):
                break

            current = self._token_by_index(
                parse,
                current.head_index,
            )

        return False

    def _independent_relation_groups(
        self,
        relations: tuple[
            MaterialRequirement,
            ...
        ],
        parse: QuestionParse,
    ) -> tuple[
        tuple[
            MaterialRequirement,
            ...
        ],
        ...
    ]:
        relation_pairs = tuple(
            (
                relation,
                self._matching_relation_token(
                    relation,
                    parse,
                ),
            )
            for relation in relations
        )

        relation_indexes = {
            token.index
            for _, token in relation_pairs
            if token is not None
        }

        independent_indexes = {
            token.index
            for _, token in relation_pairs
            if (
                token is not None
                and token.dependency == "conj"
                and self._relation_has_own_subject(
                    token,
                    parse,
                )
            )
        }

        if not independent_indexes:
            return (
                relations,
            )

        grouped: dict[
            int | None,
            list[MaterialRequirement],
        ] = {
            None: [],
        }

        for index in sorted(
            independent_indexes
        ):
            grouped[index] = []

        for (
            relation,
            token,
        ) in relation_pairs:
            if token is None:
                grouped[None].append(
                    relation
                )
                continue

            group_index = (
                self._independent_group_for_relation(
                    token,
                    independent_indexes,
                    relation_indexes,
                    parse,
                )
            )

            grouped[
                group_index
            ].append(
                relation
            )

        result = []

        if grouped[None]:
            result.append(
                tuple(
                    grouped[None]
                )
            )

        for index in sorted(
            independent_indexes
        ):
            if grouped[index]:
                result.append(
                    tuple(
                        grouped[index]
                    )
                )

        return tuple(
            result
        )

    def _relation_has_own_subject(
        self,
        relation: ParsedToken,
        parse: QuestionParse,
    ) -> bool:
        return any(
            token.dependency
            in {
                "nsubj",
                "nsubjpass",
            }
            and token.head_index
            == relation.index
            for token in parse.tokens
        )

    def _independent_group_for_relation(
        self,
        relation: ParsedToken,
        independent_indexes: set[int],
        relation_indexes: set[int],
        parse: QuestionParse,
    ) -> int | None:
        current = relation
        visited: set[int] = set()

        while (
            current.index
            not in visited
        ):
            visited.add(
                current.index
            )

            if (
                current.index
                in independent_indexes
            ):
                return current.index

            if (
                current.head_index
                not in relation_indexes
                or current.head_index
                == current.index
            ):
                break

            current = self._token_by_index(
                parse,
                current.head_index,
            )

        return None

    def _requirements_of_kind(
        self,
        requirements: MaterialQuestionRequirements,
        kind: MaterialRequirementKind,
    ) -> tuple[
        MaterialRequirement,
        ...
    ]:
        return tuple(
            requirement
            for requirement
            in requirements.requirements
            if requirement.kind == kind
        )

    def _matching_relation_token(
        self,
        requirement: MaterialRequirement,
        parse: QuestionParse,
    ) -> ParsedToken | None:
        for token in parse.tokens:
            if (
                token.pos in {
                    "VERB",
                    "ADJ",
                }
                and token.lemma.lower()
                == requirement.text.lower()
            ):
                return token

        return None

    def _matching_noun_phrase(
        self,
        requirement: MaterialRequirement,
        parse: QuestionParse,
    ) -> ParsedSpan | None:
        target = self._normalise_text(
            requirement.text
        )

        for span in parse.noun_phrases:
            span_text = self._normalise_text(
                span.text
            )

            expanded_span_text = (
                self._normalise_text(
                    self._expanded_noun_phrase_text(
                        span,
                        parse,
                    )
                )
            )

            if (
                span_text == target
                or expanded_span_text
                == target
            ):
                return span

        return None

    def _expanded_noun_phrase_text(
        self,
        span: ParsedSpan,
        parse: QuestionParse,
    ) -> str:
        span_tokens = tuple(
            token
            for token in parse.tokens
            if (
                span.start_index
                <= token.index
                < span.end_index
            )
        )

        included_indexes = {
            token.index
            for token in span_tokens
        }

        numeric_modifiers = tuple(
            token
            for token in parse.tokens
            if (
                token.head_index
                == span.root_index
                and token.dependency
                == "nummod"
                and token.index
                not in included_indexes
            )
        )

        combined_tokens = tuple(
            sorted(
                (
                    *span_tokens,
                    *numeric_modifiers,
                ),
                key=lambda token: token.index,
            )
        )

        return " ".join(
            token.text
            for token in combined_tokens
            if not token.is_punct
        ).strip()

    def _normalise_text(
        self,
        text: str,
    ) -> str:
        return " ".join(
            text.lower()
            .replace("-", " ")
            .split()
        )

    def _token_by_index(
        self,
        parse: QuestionParse,
        index: int,
    ) -> ParsedToken:
        return next(
            token
            for token in parse.tokens
            if token.index == index
        )

    def _primary_relation_chain(
        self,
        relation_tokens: tuple[
            ParsedToken,
            ...
        ],
        parse: QuestionParse,
    ) -> tuple[
        ParsedToken,
        ...
    ]:
        if not relation_tokens:
            return ()

        relation_indexes = {
            token.index
            for token in relation_tokens
        }

        roots = tuple(
            token
            for token in relation_tokens
            if token.dependency == "ROOT"
        )

        if roots:
            start = roots[0]
        else:
            start = relation_tokens[0]

        chain = {
            start.index
        }

        changed = True

        while changed:
            changed = False

            for token in relation_tokens:
                if token.index in chain:
                    continue

                if (
                    token.head_index in chain
                    and token.dependency
                    in {
                        "xcomp",
                        "ccomp",
                    }
                ):
                    chain.add(
                        token.index
                    )
                    changed = True

        return tuple(
            token
            for token in parse.tokens
            if (
                token.index in chain
                and token.index
                in relation_indexes
            )
        )

    def _is_direct_subject_of_relation_chain(
        self,
        token: ParsedToken,
        relation_indexes: set[int],
    ) -> bool:
        return (
            token.dependency
            in {
                "nsubj",
                "nsubjpass",
            }
            and token.head_index
            in relation_indexes
        )

    def _is_prepositional_argument_of_relation_chain(
        self,
        token: ParsedToken,
        relation_indexes: set[int],
        parse: QuestionParse,
    ) -> bool:
        if token.dependency != "pobj":
            return False

        preposition = self._token_by_index(
            parse,
            token.head_index,
        )

        return (
            preposition.dependency == "prep"
            and preposition.head_index
            in relation_indexes
        )

    def _is_nested_under_root(
        self,
        token: ParsedToken,
        root_indexes: set[int],
        parse: QuestionParse,
    ) -> bool:
        current = token
        visited: set[int] = set()

        while (
            current.index
            not in visited
        ):
            visited.add(
                current.index
            )

            if (
                current.head_index
                in root_indexes
            ):
                return True

            if (
                current.head_index
                == current.index
            ):
                break

            current = self._token_by_index(
                parse,
                current.head_index,
            )

        return False

    def _is_subject_of_relation_chain(
        self,
        token: ParsedToken,
        relation_indexes: set[int],
        parse: QuestionParse,
    ) -> bool:
        if (
            token.dependency
            in {
                "nsubj",
                "nsubjpass",
            }
            and token.head_index
            in relation_indexes
        ):
            return True

        current = token
        visited: set[int] = set()

        while (
            current.index
            not in visited
        ):
            visited.add(
                current.index
            )

            if (
                current.head_index
                in relation_indexes
                and current.dependency
                in {
                    "nsubj",
                    "nsubjpass",
                }
            ):
                return True

            if (
                current.head_index
                == current.index
            ):
                break

            current = self._token_by_index(
                parse,
                current.head_index,
            )

        return False
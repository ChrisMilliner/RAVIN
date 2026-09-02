"""
Define the user-facing answer behaviours produced by RAVIN.

This module contains the framework-neutral behaviour vocabulary used
after question intent and evidence sufficiency have been determined.
The behaviour value describes what RAVIN is permitted to return:
a focused grounded answer, a grounded overview, a clarification
request, or a controlled no-grounded-answer response.

Behaviour selection is a deterministic control decision. Generative
language models do not choose these values.
"""

from enum import Enum

class AnswerBehavior(str, Enum):
    """Represent the four answer behaviours RAVIN may expose to a user.

    The value is selected by deterministic routing after question intent
    and evidence sufficiency have been assessed. A generative language
    model is not permitted to choose the answer behaviour.
    """

    DIRECT_ANSWER = "direct_answer"
    GROUNDED_OVERVIEW = "grounded_overview"
    CLARIFY = "clarify"
    NO_GROUNDED_ANSWER = "no_grounded_answer"
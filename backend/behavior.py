from enum import Enum

class AnswerBehavior(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    GROUNDED_OVERVIEW = "grounded_overview"
    CLARIFY = "clarify"
    NO_GROUNDED_ANSWER = "no_grounded_answer"
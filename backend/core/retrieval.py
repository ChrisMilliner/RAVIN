import re
from backend.core.models import PolicyDocument, RetrievedEvidence

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "can",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
    "what",
    "when",
    "where",
    "with",
}

def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())

    return {
        word
        for word in words
        if len(word) >= 3 and word not in STOP_WORDS
    }

def retrieve_evidence(
    question: str,
    policies: tuple[PolicyDocument, ...],
) -> tuple[RetrievedEvidence, ...]:
    question_tokens = _tokenize(question)

    if not question_tokens:
        return ()

    matches: list[RetrievedEvidence] = []

    for policy in policies:
        policy_tokens = _tokenize(
            f"{policy.title} {policy.text}"
        )

        matching_tokens = question_tokens & policy_tokens

        if not matching_tokens:
            continue

        relevance_score = len(matching_tokens) / len(question_tokens)

        matches.append(
            RetrievedEvidence(
                policy_id=policy.policy_id,
                policy_title=policy.title,
                source_url=policy.source_url,
                text=policy.text,
                relevance_score=relevance_score,
            )
        )

    return tuple(
        sorted(
            matches,
            key=lambda evidence: evidence.relevance_score,
            reverse=True,
        )
    )
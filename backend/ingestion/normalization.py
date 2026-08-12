import re

def normalize_policy_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text)
    return normalized.strip()
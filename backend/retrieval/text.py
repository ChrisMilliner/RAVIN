from backend.ingestion.models import PolicyChunk

def build_retrieval_text(
    chunk: PolicyChunk,
) -> str:
    parts = [chunk.policy_title]

    parts.extend(chunk.heading_path)
    parts.append(chunk.text)

    return "\n".join(parts)
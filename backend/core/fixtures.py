from backend.core.models import PolicyDocument

POLICY_FIXTURES: tuple[PolicyDocument, ...] = (
    PolicyDocument(
        policy_id="POL-EXT-001",
        title="Example Assessment Extension Policy",
        source_url="https://example.invalid/policies/assessment-extension",
        status="current",
        text=(
            "Students may request an assessment extension when circumstances "
            "outside their control affect their ability to complete an assessment "
            "by the due date. An extension request should be submitted before the "
            "assessment deadline where reasonably possible."
        ),
    ),
    PolicyDocument(
        policy_id="POL-AI-001",
        title="Example Academic Integrity Policy",
        source_url="https://example.invalid/policies/academic-integrity",
        status="current",
        text=(
            "Students must submit academic work that represents their own work "
            "and must acknowledge sources appropriately. Plagiarism, unauthorised "
            "collaboration and other forms of academic misconduct may be investigated "
            "under the academic integrity process."
        ),
    ),
)
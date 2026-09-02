"""
Provide deterministic sample material for ingestion development.

The fixtures in this module support repeatable ingestion tests and
development checks without requiring every test to depend on live
policy-source availability.

Fixture content is development support data and must not be confused
with the current production policy corpus.
"""

from backend.ingestion.models import RawPolicyContent

INGESTION_FIXTURES: tuple[RawPolicyContent, ...] = (
    RawPolicyContent(
        policy_id="POL-EXT-001",
        title="Example Assessment Extension Policy",
        source_url="https://example.invalid/policies/assessment-extension",
        status="Current",
        effective_date="14 November 2025",
        review_date="13 November 2028",
        raw_text=(
            "Students may request an assessment extension when circumstances "
            "outside their control affect their ability to complete an assessment "
            "by the due date. Requests should normally be submitted before the "
            "assessment deadline where reasonably possible.\n\n"
            "The request should include sufficient information to support the "
            "circumstances described by the student. Approved extensions may "
            "change the submission due date for the affected assessment."
        ),
    ),
    RawPolicyContent(
        policy_id="POL-OLD-001",
        title="Example Superseded Assessment Policy",
        source_url="https://example.invalid/policies/superseded-assessment",
        status="Historic",
        effective_date="1 January 2020",
        review_date=None,
        raw_text=(
            "This representative policy contains valid text but is marked "
            "as historic and must not enter the active searchable corpus."
        ),
    ),
)
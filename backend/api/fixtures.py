"""
Additional policy fixtures for demo/API variety, layered on top of Chris's
original POLICY_FIXTURES (backend/core/fixtures.py) without modifying that
file directly - avoids merge conflicts and keeps his original 2 fixtures
untouched as the reference set his own demo.py uses.

Used by the API service (backend/api) only.
"""

from backend.core.fixtures import POLICY_FIXTURES
from backend.core.models import PolicyDocument

_ADDITIONAL_FIXTURES: tuple[PolicyDocument, ...] = (
    PolicyDocument(
        policy_id="POL-HDR-001",
        title="Example HDR Candidature Leave Policy",
        source_url="https://example.invalid/policies/hdr-candidature",
        status="current",
        text=(
            "Higher Degree by Research candidates may apply for a leave of "
            "absence from candidature for documented medical, personal, or "
            "professional reasons. Approved leave suspends candidature "
            "milestones and expected completion dates."
        ),
    ),
    PolicyDocument(
        policy_id="POL-CONDUCT-001",
        title="Example Student Conduct Policy",
        source_url="https://example.invalid/policies/student-conduct",
        status="current",
        text=(
            "Students are expected to behave respectfully towards staff and "
            "other students. Breaches of the code of conduct may be referred "
            "to the student conduct process for investigation and resolution."
        ),
    ),
    PolicyDocument(
        policy_id="POL-PRIVACY-001",
        title="Example Data Privacy Policy",
        source_url="https://example.invalid/policies/data-privacy",
        status="current",
        text=(
            "Personal information collected from students is handled in "
            "accordance with privacy legislation and is only used for "
            "purposes directly related to enrolment, assessment, and "
            "university administration."
        ),
    ),
    PolicyDocument(
        policy_id="POL-EMPLOY-001",
        title="Example Casual Employment Policy",
        source_url="https://example.invalid/policies/casual-employment",
        status="current",
        text=(
            "Casual staff are engaged on an as-needed basis and paid for "
            "actual hours worked. Casual employment does not guarantee "
            "ongoing or regular work and conditions are set out in the "
            "relevant enterprise agreement."
        ),
    ),
)

# Combined set used by the API. Chris's original fixtures stay first so
# his demo.py (which imports POLICY_FIXTURES directly, unaffected by this
# file) remains the authoritative minimal reference set.
API_POLICY_FIXTURES: tuple[PolicyDocument, ...] = POLICY_FIXTURES + _ADDITIONAL_FIXTURES

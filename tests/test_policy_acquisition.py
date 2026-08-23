import pytest
from backend.ingestion.acquisition import (
    PolicyLink,
    PolicyMetadata,
    PolicyPageContent,
    build_raw_policy_content,
    build_status_url,
    discover_policy_links,
    extract_policy_id,
    extract_policy_metadata,
    extract_policy_page,
)
from backend.ingestion.models import PolicyContentUnit

def test_extract_policy_id_from_document_url():
    policy_id = extract_policy_id(
        "https://policies.latrobe.edu.au/document/view.php?id=208"
    )

    assert policy_id == "208"

def test_extract_policy_id_rejects_missing_id():
    with pytest.raises(
        ValueError,
        match="Policy URL does not contain an id parameter.",
    ):
        extract_policy_id(
            "https://policies.latrobe.edu.au/document/view.php"
        )

def test_discover_policy_links_from_browse_html():
    html = """
    <html>
        <body>
            <a href="/document/view.php?id=208">
                Academic Dress Policy
            </a>

            <a href="/document/view.php?id=300">
                Example Assessment Policy
            </a>
        </body>
    </html>
    """

    links = discover_policy_links(html)

    assert links == (
        PolicyLink(
            policy_id="208",
            title="Academic Dress Policy",
            url=(
                "https://policies.latrobe.edu.au/"
                "document/view.php?id=208"
            ),
        ),
        PolicyLink(
            policy_id="300",
            title="Example Assessment Policy",
            url=(
                "https://policies.latrobe.edu.au/"
                "document/view.php?id=300"
            ),
        ),
    )

def test_discovery_ignores_unrelated_links():
    html = """
    <html>
        <body>
            <a href="/">Home</a>

            <a href="/search">Search</a>

            <a href="/document/view.php?id=208">
                Academic Dress Policy
            </a>
        </body>
    </html>
    """

    links = discover_policy_links(html)

    assert len(links) == 1
    assert links[0].policy_id == "208"

def test_discovery_removes_duplicate_policy_ids():
    html = """
    <html>
        <body>
            <a href="/document/view.php?id=208">
                Academic Dress Policy
            </a>

            <a href="/document/view.php?id=208">
                Academic Dress Policy
            </a>
        </body>
    </html>
    """

    links = discover_policy_links(html)

    assert len(links) == 1

def test_build_status_url():
    url = build_status_url("208")

    assert url == (
        "https://policies.latrobe.edu.au/"
        "document/status-and-details.php?id=208"
    )

def test_extract_policy_metadata():
    html = """
    <html>
        <body>
            <table>
                <tr>
                    <th>Status</th>
                    <td>Current</td>
                    <td>Status description.</td>
                </tr>
                <tr>
                    <th>Effective Date</th>
                    <td>14th November 2025</td>
                    <td>Effective date description.</td>
                </tr>
                <tr>
                    <th>Review Date</th>
                    <td>13th November 2028</td>
                    <td>Review date description.</td>
                </tr>
            </table>
        </body>
    </html>
    """

    metadata = extract_policy_metadata(html)

    assert metadata == PolicyMetadata(
        status="Current",
        effective_date="14th November 2025",
        review_date="13th November 2028",
    )

def test_extract_policy_metadata_allows_missing_optional_dates():
    html = """
    <table>
        <tr>
            <th>Status</th>
            <td>Current</td>
            <td>Status description.</td>
        </tr>
    </table>
    """

    metadata = extract_policy_metadata(html)

    assert metadata.status == "Current"
    assert metadata.effective_date is None
    assert metadata.review_date is None

def test_extract_policy_metadata_requires_status():
    html = """
    <table>
        <tr>
            <th>Effective Date</th>
            <td>14th November 2025</td>
        </tr>
    </table>
    """

    with pytest.raises(
        ValueError,
        match="Policy status metadata could not be found.",
    ):
        extract_policy_metadata(html)

def test_extract_policy_page():
    html = """
    <html>
        <body>
            <nav>
                <a href="/">Home</a>
                <a href="/browse">Browse A-Z</a>
            </nav>

            <h1 id="sliph-document-title">
                Academic Dress Policy
            </h1>

            <div id="sliph-document-content">
                <h1 id="section1">
                    Section 1 - Key Information
                </h1>

                <p>
                    This is authoritative policy content.
                </p>

                <h1 id="section2">
                    Section 2 - Purpose
                </h1>

                <p>
                    This section describes the purpose.
                </p>
            </div>

            <footer>
                Copyright information.
            </footer>
        </body>
    </html>
    """

    page = extract_policy_page(html)

    assert page.title == "Academic Dress Policy"

    assert "Section 1 - Key Information" in page.raw_text
    assert "This is authoritative policy content." in page.raw_text
    assert "Section 2 - Purpose" in page.raw_text

    assert "Home" not in page.raw_text
    assert "Browse A-Z" not in page.raw_text
    assert "Copyright information." not in page.raw_text

def test_extract_policy_page_requires_title():
    html = """
    <div id="sliph-document-content">
        Policy content.
    </div>
    """

    with pytest.raises(
        ValueError,
        match="Policy document title could not be found.",
    ):
        extract_policy_page(html)

def test_extract_policy_page_requires_content():
    html = """
    <h1 id="sliph-document-title">
        Academic Dress Policy
    </h1>
    """

    with pytest.raises(
        ValueError,
        match="Policy document content could not be found.",
    ):
        extract_policy_page(html)

def test_build_raw_policy_content_combines_document_and_metadata():
    link = PolicyLink(
        policy_id="208",
        title="Academic Dress Policy",
        url=(
            "https://policies.latrobe.edu.au/"
            "document/view.php?id=208"
        ),
    )

    policy_html = """
    <html>
        <body>
            <h1 id="sliph-document-title">
                Academic Dress Policy
            </h1>

            <div id="sliph-document-content">
                <h1>Section 1 - Key Information</h1>
                <p>Authoritative policy content.</p>
            </div>
        </body>
    </html>
    """

    status_html = """
    <table>
        <tr>
            <th>Status</th>
            <td>Current</td>
        </tr>
        <tr>
            <th>Effective Date</th>
            <td>14th November 2025</td>
        </tr>
        <tr>
            <th>Review Date</th>
            <td>13th November 2028</td>
        </tr>
    </table>
    """

    policy = build_raw_policy_content(
        link=link,
        policy_html=policy_html,
        status_html=status_html,
    )

    assert policy.policy_id == "208"
    assert policy.title == "Academic Dress Policy"
    assert policy.status == "Current"
    assert policy.effective_date == "14th November 2025"
    assert policy.review_date == "13th November 2028"

    assert (
        policy.source_url
        == "https://policies.latrobe.edu.au/"
        "document/view.php?id=208"
    )

    assert "Section 1 - Key Information" in policy.raw_text
    assert "Authoritative policy content." in policy.raw_text

def test_extract_policy_page_removes_embedded_navigation_content():
    html = """
    <html>
        <body>
            <h1 id="sliph-document-title">
                Academic Dress Policy
            </h1>

            <div id="sliph-document-content">
                <div class="sliph-document-status current">
                    This is the current version of this document.
                </div>

                <h1 id="section1">
                    Section 1 - Key Information
                </h1>

                <p>
                    Authoritative policy content.
                </p>

                <a href="#document-top">
                    Top of Page
                </a>

                <h1 id="section2">
                    Section 2 - Purpose
                </h1>

                <p>
                    More authoritative policy content.
                </p>
            </div>
        </body>
    </html>
    """

    page = extract_policy_page(html)

    assert "Section 1 - Key Information" in page.raw_text
    assert "Authoritative policy content." in page.raw_text
    assert "Section 2 - Purpose" in page.raw_text

    assert "This is the current version" not in page.raw_text
    assert "Top of Page" not in page.raw_text

def test_extract_policy_page_preserves_heading_hierarchy():
    html = """
    <html>
        <body>
            <h1 id="sliph-document-title">
                Example Policy
            </h1>

            <div id="sliph-document-content">
                <h1 id="section6">
                    Section 6 - Procedures
                </h1>

                <p>
                    General procedure information.
                </p>

                <h2 id="part1">
                    Part A -&nbsp;First Procedure
                </h2>

                <p>
                    Part A content.
                </p>

                <h3 id="major1">
                    Special Cases
                </h3>

                <p>
                    Special case content.
                </p>

                <h2 id="part2">
                    Part B - Second Procedure
                </h2>

                <p>
                    Part B content.
                </p>

                <h1 id="section7">
                    Section 7 - Definitions
                </h1>

                <p>
                    Definition content.
                </p>
            </div>
        </body>
    </html>
    """

    page = extract_policy_page(html)

    assert page.content_units == (
        PolicyContentUnit(
            heading_path=(
                "Section 6 - Procedures",
            ),
            text="General procedure information.",
        ),
        PolicyContentUnit(
            heading_path=(
                "Section 6 - Procedures",
                "Part A - First Procedure",
            ),
            text="Part A content.",
        ),
        PolicyContentUnit(
            heading_path=(
                "Section 6 - Procedures",
                "Part A - First Procedure",
                "Special Cases",
            ),
            text="Special case content.",
        ),
        PolicyContentUnit(
            heading_path=(
                "Section 6 - Procedures",
                "Part B - Second Procedure",
            ),
            text="Part B content.",
        ),
        PolicyContentUnit(
            heading_path=(
                "Section 7 - Definitions",
            ),
            text="Definition content.",
        ),
    )
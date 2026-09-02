"""
Acquire current policy content and metadata from policy web sources.

This module owns the external policy-source boundary. It discovers or
accepts policy links, fetches policy HTML, extracts identifying
metadata and structured content units, and produces raw policy content
for downstream processing.

Network acquisition remains separate from normalization, chunking, and
retrieval so those stages can be tested independently.
"""

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from backend.ingestion.models import (
    PolicyContentUnit,
    RawPolicyContent,
)

BASE_URL = "https://policies.latrobe.edu.au/"
BROWSE_URL = urljoin(BASE_URL, "browse")
HEADING_LEVELS = {
    "h1": 1,
    "h2": 2,
    "h3": 3,
    "h4": 4,
}

@dataclass(frozen=True)
class PolicyLink:
    """Represent a discoverable policy document and its source location.

    The policy identifier is retained separately from the title and URL so
    acquisition, ingestion, retrieval, and later source citation can refer
    to the same policy consistently.
    """

    policy_id: str
    title: str
    url: str

@dataclass(frozen=True)
class PolicyMetadata:
    """Represent status metadata acquired separately from policy body content.

    Status is required by the ingestion quality controls, while effective
    and review dates preserve source metadata that can be carried into
    retrieval chunks.
    """

    status: str
    effective_date: str | None
    review_date: str | None

@dataclass(frozen=True)
class PolicyPageContent:
    """Represent the policy body extracted from one policy document page.

    The model retains the page title, complete raw text, and heading-aware
    content units used by downstream normalization and chunking.
    """

    title: str
    raw_text: str
    content_units: tuple[PolicyContentUnit, ...]

def extract_policy_id(url: str) -> str:
    """Extract the policy identifier from a policy-database URL.

    A ValueError is raised when the URL has no id query parameter because
    policy identity is required for traceable ingestion and retrieval.
    """
    query = parse_qs(urlparse(url).query)

    policy_ids = query.get("id")

    if not policy_ids:
        raise ValueError("Policy URL does not contain an id parameter.")

    return policy_ids[0]

def discover_policy_links(html: str) -> tuple[PolicyLink, ...]:
    """Discover unique policy-document links from policy-database HTML.

    Only links matching the policy document URL pattern and containing
    usable link text are retained. Duplicate policy identifiers are reduced
    to one PolicyLink before the immutable result is returned.
    """
    soup = BeautifulSoup(html, "html.parser")

    links: list[PolicyLink] = []

    for anchor in soup.find_all(
        "a",
        href=re.compile(r"document/view\.php\?id=\d+"),
    ):
        href = anchor.get("href")

        if not isinstance(href, str):
            continue

        title = anchor.get_text(
            " ",
            strip=True,
        )

        if not title:
            continue

        url = urljoin(
            BASE_URL,
            href,
        )

        links.append(
            PolicyLink(
                policy_id=extract_policy_id(url),
                title=title,
                url=url,
            )
        )

    unique_links = {
        link.policy_id: link
        for link in links
    }

    return tuple(
        unique_links.values()
    )

def fetch_html(
    url: str,
    timeout_seconds: float = 15.0,
) -> str:
    """Fetch HTML from a policy source with a bounded network timeout.

    HTTP error responses are surfaced through Requests rather than being
    converted into successful acquisition results, allowing startup and
    ingestion callers to fail explicitly.
    """
    response = requests.get(
        url,
        timeout=timeout_seconds,
    )

    response.raise_for_status()

    return response.text

def extract_policy_metadata(html: str) -> PolicyMetadata:
    """Extract policy status and date metadata from a status-details page.

    Policy status is mandatory because RAVIN only promotes current policy
    content through the normal ingestion pipeline. Effective and review
    dates are retained when available.
    """
    soup = BeautifulSoup(html, "html.parser")

    metadata: dict[str, str] = {}

    for row in soup.find_all("tr"):
        cells = [
            cell.get_text(" ", strip=True)
            for cell in row.find_all(["th", "td"])
        ]

        if len(cells) < 2:
            continue

        field_name = cells[0]
        field_value = cells[1]

        metadata[field_name] = field_value

    status = metadata.get("Status")

    if not status:
        raise ValueError(
            "Policy status metadata could not be found."
        )

    return PolicyMetadata(
        status=status,
        effective_date=metadata.get("Effective Date"),
        review_date=metadata.get("Review Date"),
    )

def build_status_url(policy_id: str) -> str:
    """Build the policy status-and-details URL for a policy identifier.

    Keeping status URL construction explicit allows policy body acquisition
    and policy metadata acquisition to remain separate source operations.
    """
    return urljoin(
        BASE_URL,
        f"document/status-and-details.php?id={policy_id}",
    )

def extract_policy_content_units(
    content_element: Tag,
) -> tuple[PolicyContentUnit, ...]:
    """Extract heading-aware content units from a policy document element.

    Top-level child elements are traversed in source order. Heading elements
    update the active heading path, while subsequent content is grouped
    under that structural path until another heading is encountered.

    The resulting units preserve policy structure for chunking, retrieval,
    context expansion, and evidence citation.
    """
    units: list[PolicyContentUnit] = []

    heading_path: list[str] = []
    current_text_parts: list[str] = []

    for child in content_element.find_all(recursive=False):
        if child.name in HEADING_LEVELS:
            if current_text_parts:
                text = " ".join(current_text_parts).strip()

                if text:
                    units.append(
                        PolicyContentUnit(
                            heading_path=tuple(heading_path),
                            text=text,
                        )
                    )

                current_text_parts = []

            level = HEADING_LEVELS[child.name]

            heading_path = heading_path[: level - 1]
            heading_text = " ".join(
                child.get_text(" ", strip=True).split()
            )

            heading_path.append(heading_text)

            continue

        text = child.get_text(" ", strip=True)

        if text:
            current_text_parts.append(text)

    if current_text_parts:
        text = " ".join(current_text_parts).strip()

        if text:
            units.append(
                PolicyContentUnit(
                    heading_path=tuple(heading_path),
                    text=text,
                )
            )

    return tuple(units)

def extract_policy_page(html: str) -> PolicyPageContent:
    """Extract title, text, and structured content from policy page HTML.

    Required title and document-content elements are validated before
    content is returned. Presentation-only status and navigation elements
    are removed so they do not contaminate retrievable policy evidence.
    """
    soup = BeautifulSoup(html, "html.parser")

    title_element = soup.find(id="sliph-document-title")

    if title_element is None:
        raise ValueError(
            "Policy document title could not be found."
        )

    content_element = soup.find(id="sliph-document-content")

    if content_element is None:
        raise ValueError(
            "Policy document content could not be found."
        )

    for unwanted_element in content_element.select(
        ".sliph-document-status, .top-link, a[href='#document-top']"
        ):
        unwanted_element.decompose()

    content_units = extract_policy_content_units(
    content_element
    )

    title = title_element.get_text(
        " ",
        strip=True,
    )

    raw_text = content_element.get_text(
        "\n",
        strip=True,
    )

    if not title:
        raise ValueError(
            "Policy document title was empty."
        )

    if not raw_text:
        raise ValueError(
            "Policy document content was empty."
        )

    return PolicyPageContent(
        title=title,
        raw_text=raw_text,
        content_units=content_units,
    )

def build_raw_policy_content(
    link: PolicyLink,
    policy_html: str,
    status_html: str,
) -> RawPolicyContent:
    """Combine acquired policy body and status metadata into RawPolicyContent.

    The function joins the independently acquired source documents while
    preserving the original policy identifier, URL, dates, status, raw
    text, and heading-aware content units.
    """
    page = extract_policy_page(policy_html)
    metadata = extract_policy_metadata(status_html)

    return RawPolicyContent(
        policy_id=link.policy_id,
        title=page.title,
        source_url=link.url,
        status=metadata.status,
        effective_date=metadata.effective_date,
        review_date=metadata.review_date,
        raw_text=page.raw_text,
        content_units=page.content_units,
    )

def acquire_policy(
    link: PolicyLink,
    timeout_seconds: float = 15.0,
) -> RawPolicyContent:
    """Acquire one complete raw policy record from its policy link.

    The policy document and status-details pages are fetched separately and
    then combined into a traceable RawPolicyContent instance. Network or
    source-validation failures are allowed to propagate to the caller.
    """
    policy_html = fetch_html(
        link.url,
        timeout_seconds=timeout_seconds,
    )

    status_html = fetch_html(
        build_status_url(link.policy_id),
        timeout_seconds=timeout_seconds,
    )

    return build_raw_policy_content(
        link=link,
        policy_html=policy_html,
        status_html=status_html,
    )
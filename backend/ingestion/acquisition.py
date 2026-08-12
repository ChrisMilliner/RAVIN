import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
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
    policy_id: str
    title: str
    url: str

@dataclass(frozen=True)
class PolicyMetadata:
    status: str
    effective_date: str | None
    review_date: str | None

@dataclass(frozen=True)
class PolicyPageContent:
    title: str
    raw_text: str
    content_units: tuple[PolicyContentUnit, ...]

def extract_policy_id(url: str) -> str:
    query = parse_qs(urlparse(url).query)

    policy_ids = query.get("id")

    if not policy_ids:
        raise ValueError("Policy URL does not contain an id parameter.")

    return policy_ids[0]

def discover_policy_links(html: str) -> tuple[PolicyLink, ...]:
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
    response = requests.get(
        url,
        timeout=timeout_seconds,
    )

    response.raise_for_status()

    return response.text

def extract_policy_metadata(html: str) -> PolicyMetadata:
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
    return urljoin(
        BASE_URL,
        f"document/status-and-details.php?id={policy_id}",
    )

def extract_policy_content_units(
    content_element,
) -> tuple[PolicyContentUnit, ...]:
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
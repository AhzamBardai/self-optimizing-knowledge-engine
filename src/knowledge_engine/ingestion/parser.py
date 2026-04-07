# src/knowledge_engine/ingestion/parser.py
"""
Extract structured sections from SEC 10-K HTML filings.

Sections targeted: Business (Item 1), Risk Factors (Item 1A),
MD&A (Item 7), Financial Highlights.
"""
from __future__ import annotations

import re

import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger()

SECTION_PATTERNS: dict[str, list[str]] = {
    "business": [r"item\s*1\b(?!a)", r"business\s*overview"],
    "risk_factors": [r"item\s*1a", r"risk\s*factors"],
    "mda": [r"item\s*7\b(?!a)", r"management.{0,20}discussion"],
    "financial_highlights": [r"financial\s*highlights", r"selected\s*financial\s*data"],
    "quantitative_disclosures": [r"item\s*7a", r"quantitative.*qualitative"],
}


def extract_text_from_html(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    # Remove script/style noise
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_sections(raw_text: str) -> dict[str, str]:
    """
    Split 10-K text into labeled sections.

    Uses regex anchors to find section boundaries. Falls back to
    the full text for 'full_document' if sections are not detected.
    """
    # Normalize: lowercase for matching, preserve original for extraction
    lower = raw_text.lower()
    sections: dict[str, str] = {}

    # Find start positions for each section
    positions: dict[str, int] = {}
    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                positions[section_name] = match.start()
                break

    if not positions:
        log.warning("parser.no_sections_found", text_length=len(raw_text))
        return {"full_document": raw_text}

    # Sort by position to extract text between consecutive section starts
    sorted_sections = sorted(positions.items(), key=lambda x: x[1])
    for idx, (section_name, start_pos) in enumerate(sorted_sections):
        end_pos = sorted_sections[idx + 1][1] if idx + 1 < len(sorted_sections) else len(raw_text)
        section_text = raw_text[start_pos:end_pos].strip()
        if len(section_text) > 100:  # filter out spurious matches
            sections[section_name] = section_text
            log.debug(
                "parser.section_extracted",
                section=section_name,
                length=len(section_text),
            )

    if not sections:
        log.warning("parser.no_sections_extracted", text_length=len(raw_text))
        return {"full_document": raw_text}

    return sections

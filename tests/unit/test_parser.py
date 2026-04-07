# tests/unit/test_parser.py
"""Tests for the 10-K section parser."""
from pathlib import Path

from knowledge_engine.ingestion.parser import extract_text_from_html, parse_sections

SAMPLE_HTML = """
<html><body>
<p>APPLE INC. FORM 10-K</p>
<p>ITEM 1. BUSINESS</p>
<p>Apple designs smartphones and computers.</p>
<p>ITEM 1A. RISK FACTORS</p>
<p>Competition in the technology sector is intense.</p>
<p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</p>
<p>Net sales were $383,285 million in 2023.</p>
</body></html>
"""


def test_extract_text_strips_html() -> None:
    text = extract_text_from_html(SAMPLE_HTML)
    assert "<html>" not in text
    assert "Apple designs" in text


def test_parse_sections_identifies_known_sections() -> None:
    text = extract_text_from_html(SAMPLE_HTML)
    sections = parse_sections(text)
    assert "business" in sections or "full_document" in sections


def test_parse_sections_with_fixture() -> None:
    fixture = Path("tests/fixtures/sample_10k.txt").read_text()
    sections = parse_sections(fixture)
    assert len(sections) >= 1
    for _name, content in sections.items():
        assert len(content) > 50, "Section should not be empty"


def test_extract_text_removes_scripts() -> None:
    html = "<html><body><script>alert(1)</script><p>Real content</p></body></html>"
    text = extract_text_from_html(html)
    assert "alert" not in text
    assert "Real content" in text

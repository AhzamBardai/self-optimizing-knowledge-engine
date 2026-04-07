# src/knowledge_engine/ingestion/edgar_client.py
"""
Fetch 10-K filings from the SEC EDGAR full-text search API.

EDGAR does not require authentication. The User-Agent header is mandatory
per SEC fair-access policy — see https://www.sec.gov/developer
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger()

EDGAR_BASE = "https://data.sec.gov"
EDGAR_SUBMISSIONS = f"{EDGAR_BASE}/submissions"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


@dataclass
class Filing:
    """A parsed SEC 10-K filing."""

    ticker: str
    company_name: str
    cik: str
    accession_number: str
    fiscal_year_end: str
    raw_text: str
    sections: dict[str, str] = field(default_factory=dict)


class EdgarClient:
    """Client for the SEC EDGAR API."""

    def __init__(self, user_agent: str, request_delay_seconds: float = 0.1) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.session.headers["Accept-Encoding"] = "gzip, deflate"
        self.delay = request_delay_seconds

    def _get(self, url: str) -> Any:
        time.sleep(self.delay)  # respect EDGAR rate limits (10 req/sec max)
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _get_text(self, url: str) -> str:
        time.sleep(self.delay)
        resp = self.session.get(url, timeout=60)
        resp.raise_for_status()
        return resp.text

    def get_cik(self, ticker: str) -> str:
        """Resolve ticker symbol to CIK (padded to 10 digits)."""
        data = self._get(COMPANY_TICKERS_URL)
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                log.info("edgar.cik_resolved", ticker=ticker, cik=cik)
                return cik
        raise ValueError(f"Ticker {ticker!r} not found in EDGAR company tickers")

    def get_latest_10k(self, ticker: str) -> Filing:
        """Fetch the most recent 10-K filing for a ticker."""
        cik = self.get_cik(ticker)
        submissions = self._get(f"{EDGAR_SUBMISSIONS}/CIK{cik}.json")
        company_name: str = submissions.get("name", ticker)

        recent = submissions.get("filings", {}).get("recent", {})
        form_types: list[str] = recent.get("form", [])
        accession_numbers: list[str] = recent.get("accessionNumber", [])
        dates: list[str] = recent.get("filingDate", [])

        for i, form in enumerate(form_types):
            if form == "10-K":
                accession = accession_numbers[i]
                fiscal_year_end = dates[i]
                filing_text = self._fetch_filing_text(cik, accession)
                log.info(
                    "edgar.filing_fetched",
                    ticker=ticker,
                    accession=accession,
                    length=len(filing_text),
                )
                return Filing(
                    ticker=ticker,
                    company_name=company_name,
                    cik=cik,
                    accession_number=accession,
                    fiscal_year_end=fiscal_year_end,
                    raw_text=filing_text,
                )

        raise ValueError(f"No 10-K filing found for {ticker}")

    def _fetch_filing_text(self, cik: str, accession_number: str) -> str:
        """Fetch the primary document text from a filing."""
        clean_accession = accession_number.replace("-", "")
        index_url = (
            f"{EDGAR_ARCHIVES}/{cik.lstrip('0')}/{clean_accession}/"
            f"{accession_number}-index.json"
        )
        try:
            index = self._get(index_url)
        except Exception:
            # Fall back to older index format
            index_url = (
                f"{EDGAR_ARCHIVES}/{cik.lstrip('0')}/{clean_accession}/index.json"
            )
            index = self._get(index_url)

        # Find the primary .htm document
        documents: list[dict[str, Any]] = index.get("documents", [])
        for doc in documents:
            doc_type = doc.get("type", "")
            doc_name: str = doc.get("document", "")
            if doc_type == "10-K" and doc_name.endswith(".htm"):
                text_url = (
                    f"{EDGAR_ARCHIVES}/{cik.lstrip('0')}/{clean_accession}/{doc_name}"
                )
                return self._get_text(text_url)

        # Last resort: try to get the first .htm file
        for doc in documents:
            doc_name = doc.get("document", "")
            if doc_name.endswith((".htm", ".txt")):
                text_url = (
                    f"{EDGAR_ARCHIVES}/{cik.lstrip('0')}/{clean_accession}/{doc_name}"
                )
                return self._get_text(text_url)

        raise ValueError(f"Could not find primary document for {accession_number}")

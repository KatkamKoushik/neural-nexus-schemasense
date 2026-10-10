"""Business Rules Ingestion & Grounding Engine — PDF extraction via pypdf."""

from __future__ import annotations

import io
import logging
from typing import BinaryIO

from pypdf import PdfReader
from pypdf.errors import PyPdfError

logger = logging.getLogger("schemasense.rules")


class PDFIngestionError(Exception):
    """Raised when PDF extraction fails."""


def extract_rules_from_pdf(
    pdf_file: BinaryIO | bytes, max_pages: int = 20
) -> list[str]:
    """
    Extracts text sections and business rules from an uploaded PDF document.

    Returns:
        list[str]: Extracted paragraphs / rule statements.
    """
    if isinstance(pdf_file, bytes):
        stream = io.BytesIO(pdf_file)
    else:
        stream = pdf_file

    try:
        reader = PdfReader(stream)
        if len(reader.pages) == 0:
            return []

        rules: list[str] = []
        pages_to_read = min(len(reader.pages), max_pages)

        for page_idx in range(pages_to_read):
            page = reader.pages[page_idx]
            text = page.extract_text() or ""
            # Split into meaningful paragraphs
            for block in text.split("\n\n"):
                clean = " ".join(block.split()).strip()
                if clean and len(clean) > 15:  # Ignore headers/footers/page numbers
                    rules.append(clean)

        return rules
    except PyPdfError as pe:
        logger.warning(f"PyPDF extraction error: {pe}")
        raise PDFIngestionError(f"Corrupted or unreadable PDF: {pe}") from pe
    except Exception as e:
        logger.warning(f"Unexpected error parsing PDF: {e}")
        raise PDFIngestionError(f"Failed to extract PDF text: {e}") from e


def format_rules_for_prompt(rules: list[str]) -> str:
    """Formats business rules into an XML-delimited prompt block."""
    if not rules:
        return ""

    formatted_lines = ["<enterprise_business_rules>"]
    for idx, rule in enumerate(
        rules[:30], 1
    ):  # Cap to top 30 rules to preserve token budgets
        formatted_lines.append(f"{idx}. {rule}")
    formatted_lines.append("</enterprise_business_rules>")
    return "\n".join(formatted_lines)

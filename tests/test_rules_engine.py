"""Tests for PDF business rules extraction and prompt grounding."""

import io

import pytest
from pypdf import PdfWriter

from schemasense.rules_engine import (
    PDFIngestionError,
    extract_rules_from_pdf,
    format_rules_for_prompt,
)


def test_extract_rules_from_blank_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    pdf_bytes = io.BytesIO()
    writer.write(pdf_bytes)
    pdf_bytes.seek(0)

    rules = extract_rules_from_pdf(pdf_bytes)
    assert rules == []


def test_extract_rules_corrupted_stream():
    corrupt_bytes = io.BytesIO(b"this is not a valid pdf file")
    with pytest.raises(PDFIngestionError):
        extract_rules_from_pdf(corrupt_bytes)


def test_format_rules_for_prompt():
    sample_rules = [
        "Rule 1: Orders with status 'cancelled' must not be included in revenue calculations.",
        "Rule 2: Every customer must have a valid country code.",
    ]
    prompt_str = format_rules_for_prompt(sample_rules)
    assert "<enterprise_business_rules>" in prompt_str
    assert "</enterprise_business_rules>" in prompt_str
    assert "Rule 1" in prompt_str
    assert "Rule 2" in prompt_str


def test_format_rules_empty():
    assert format_rules_for_prompt([]) == ""

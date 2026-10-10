"""Regression tests for dependency hygiene, modern SDK imports, and SBOM consistency."""

import json
import sys
from pathlib import Path


def test_modern_sdk_imports():
    """Verifies that the modern google-genai SDK imports cleanly and exposes Client."""
    import google.genai

    assert hasattr(google.genai, "Client")


def test_production_code_does_not_require_deprecated_sdk():
    """Verifies that schemasense modules import and execute without importing google.generativeai."""
    import importlib

    modules_to_test = [
        "schemasense.ai_agent",
        "schemasense.benchmark",
        "schemasense.connection",
        "schemasense.profiler",
        "schemasense.rules_engine",
        "schemasense.schema_inspector",
        "schemasense.sql_guard",
    ]
    for mod_name in modules_to_test:
        importlib.import_module(mod_name)

    # Verify that none of the schemasense modules imported google.generativeai
    assert "google.generativeai" not in sys.modules


def test_requirements_file_hygiene():
    """Verifies that requirements.txt declares google-genai and omits google-generativeai."""
    req_path = Path(__file__).resolve().parent.parent / "requirements.txt"
    content = req_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert any(line.startswith("google-genai") for line in lines)
    assert not any(line.startswith("google-generativeai") for line in lines)


def test_constraints_file_hygiene():
    """Verifies that constraints.txt pins google-genai and omits google-generativeai."""
    constraints_path = Path(__file__).resolve().parent.parent / "constraints.txt"
    content = constraints_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert any(line.startswith("google-genai==") for line in lines)
    assert not any(line.startswith("google-generativeai") for line in lines)


def test_setup_script_checks_modern_sdk():
    """Verifies that setup.sh checks google.genai rather than deprecated SDK."""
    setup_path = Path(__file__).resolve().parent.parent / "setup.sh"
    content = setup_path.read_text(encoding="utf-8")

    assert "import google.genai" in content
    assert "import google.generativeai" not in content


def test_sbom_includes_modern_sdk_and_omits_deprecated():
    """Verifies that sbom.cyclonedx.json is valid CycloneDX and reflects modern google-genai."""
    sbom_path = Path(__file__).resolve().parent.parent / "sbom.cyclonedx.json"
    assert sbom_path.exists()

    data = json.loads(sbom_path.read_text(encoding="utf-8"))
    assert data.get("bomFormat") == "CycloneDX"
    assert "components" in data

    component_names = {c["name"] for c in data["components"]}
    assert "google-genai" in component_names
    assert "google-generativeai" not in component_names

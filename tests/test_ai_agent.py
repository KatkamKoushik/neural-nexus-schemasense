"""Tests for SchemaSense AI Agent — google-genai migration, structured output, and error resilience."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from schemasense.ai_agent import (
    AIQuotaExceededError,
    SchemaSenseAIAgent,
    SQLAgentResponse,
    extract_sql_from_text,
    verify_schema_fidelity,
)


# ──────────────────────────────────────────────
# UNIT TESTS: UTILITIES & PARSERS
# ──────────────────────────────────────────────
def test_extract_sql_from_markdown_fences():
    """Verifies that SQL is cleanly extracted from markdown code fences."""
    raw_text = (
        "Here is the requested analytical query:\n\n"
        "```sql\n"
        "SELECT id, name, city FROM customers WHERE city = 'Delhi';\n"
        "```\n\n"
        "This filters customers by location."
    )
    extracted = extract_sql_from_text(raw_text)
    assert extracted == "SELECT id, name, city FROM customers WHERE city = 'Delhi';"


def test_extract_sql_from_bare_select():
    """Verifies that bare SELECT statements ending with a semicolon are extracted."""
    raw_text = "You can run this query: SELECT count(*) FROM orders WHERE amount > 500; Hope this helps!"
    extracted = extract_sql_from_text(raw_text)
    assert extracted == "SELECT count(*) FROM orders WHERE amount > 500;"


def test_extract_sql_from_cte_query():
    """Verifies that CTE queries with WITH clauses are extracted."""
    raw_text = "```sql\nWITH top_customers AS (SELECT customer_id FROM orders) SELECT * FROM top_customers;\n```"
    extracted = extract_sql_from_text(raw_text)
    assert extracted is not None
    assert "WITH top_customers AS" in extracted


def test_extract_sql_returns_none_for_non_sql():
    """Verifies that purely descriptive text returns None."""
    raw_text = "The customers table stores user demographics including city, email, and sign up dates."
    assert extract_sql_from_text(raw_text) is None
    assert extract_sql_from_text("") is None


def test_verify_schema_fidelity_detects_hallucinations():
    """Verifies that out-of-schema table names are identified as hallucinations."""
    allowed = {"customers", "orders", "products"}
    extracted = ["customers", "orders", "payments", "credit_cards"]

    hallucinations = verify_schema_fidelity(extracted, allowed)
    assert "payments" in hallucinations
    assert "credit_cards" in hallucinations
    assert "customers" not in hallucinations


def test_verify_schema_fidelity_clean_on_valid_tables():
    """Verifies that fully valid table references produce zero hallucinations."""
    allowed = {"customers", "orders"}
    extracted = ["customers", "orders"]

    assert verify_schema_fidelity(extracted, allowed) == []


# ──────────────────────────────────────────────
# UNIT TESTS: STRUCTURED MODEL OUTPUT SCHEMA
# ──────────────────────────────────────────────
def test_sql_agent_response_model():
    """Verifies that SQLAgentResponse instantiates and validates correctly."""
    resp = SQLAgentResponse(
        explanation="Calculates top spending customers.",
        sql_query="SELECT customer_id, sum(amount) FROM orders GROUP BY customer_id;",
        target_tables=["orders"],
        is_executable=True,
    )
    assert resp.is_executable is True
    assert resp.target_tables == ["orders"]
    assert "orders" in resp.sql_query

    # JSON roundtrip
    json_data = resp.model_dump_json()
    reconstructed = SQLAgentResponse.model_validate_json(json_data)
    assert reconstructed == resp


def test_sql_agent_response_requires_explanation():
    """Verifies that missing explanation raises a ValidationError."""
    with pytest.raises(ValidationError):
        SQLAgentResponse.model_validate({"is_executable": True})


# ──────────────────────────────────────────────
# UNIT TESTS: RESILIENCE & KEY ROTATION
# ──────────────────────────────────────────────
def test_ai_agent_key_rotation():
    """Verifies that key rotation cycles through available API keys."""
    keys = ["KEY_ALPHA", "KEY_BETA", "KEY_GAMMA"]
    agent = SchemaSenseAIAgent(api_keys=keys)

    assert agent.active_key == "KEY_ALPHA"
    assert agent.rotate_key() == "KEY_BETA"
    assert agent.rotate_key() == "KEY_GAMMA"
    # Wraparound
    assert agent.rotate_key() == "KEY_ALPHA"


@patch("schemasense.ai_agent.genai.Client")
def test_ai_agent_rate_limit_rotation_and_recovery(mock_client_class):
    """Verifies that encountering 429 quota errors triggers key rotation and retries."""
    keys = ["KEY_FAIL", "KEY_SUCCESS"]
    agent = SchemaSenseAIAgent(api_keys=keys)

    # First client call raises 429 quota error, second client call succeeds
    mock_response = MagicMock()
    mock_response.text = '{"explanation": "Success", "sql_query": "SELECT 1;", "target_tables": [], "is_executable": true}'

    mock_client_fail = MagicMock()
    mock_client_fail.models.generate_content.side_effect = Exception(
        "429 Resource has been exhausted (quota limit)"
    )

    mock_client_success = MagicMock()
    mock_client_success.models.generate_content.return_value = mock_response

    mock_client_class.side_effect = [mock_client_fail, mock_client_success]

    response = agent.generate_structured("test prompt")
    assert response.is_executable is True
    assert response.sql_query == "SELECT 1;"
    assert agent.active_key == "KEY_SUCCESS"


@patch("schemasense.ai_agent.genai.Client")
def test_ai_agent_quota_exhaustion_raises_clean_error(mock_client_class):
    """Verifies that when all keys return 429, AIQuotaExceededError is raised."""
    keys = ["KEY_1", "KEY_2"]
    agent = SchemaSenseAIAgent(api_keys=keys)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 RateLimitExceeded")
    mock_client_class.return_value = mock_client

    with pytest.raises(AIQuotaExceededError):
        agent.generate_structured("test prompt")


@patch("schemasense.ai_agent.genai.Client")
def test_ai_agent_malformed_json_fallback(mock_client_class):
    """Verifies that malformed JSON responses fall back cleanly to regex extraction."""
    agent = SchemaSenseAIAgent(api_keys=["KEY_1"])

    mock_response = MagicMock()
    # Non-JSON text with embedded SQL fence
    mock_response.text = (
        "Here is your answer without JSON:\n"
        "```sql\nSELECT id, name FROM customers;\n```"
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    result = agent.generate_structured("Show me customers")
    assert result.sql_query == "SELECT id, name FROM customers;"
    assert result.is_executable is True


@patch("schemasense.ai_agent.genai.Client")
def test_ai_agent_hallucinated_table_disables_executable(mock_client_class):
    """Verifies that referenced tables outside the allowed schema set mark is_executable as False."""
    agent = SchemaSenseAIAgent(api_keys=["KEY_1"])

    mock_response = MagicMock()
    mock_response.text = (
        '{"explanation": "Query", "sql_query": "SELECT * FROM secret_audit;", '
        '"target_tables": ["secret_audit"], "is_executable": true}'
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    result = agent.generate_structured(
        prompt="Audit query",
        allowed_tables={"customers", "orders"},
    )
    assert result.is_executable is False
    assert result.safety_notes is not None
    assert "Hallucination Warning" in result.safety_notes

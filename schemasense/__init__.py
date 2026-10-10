"""SchemaSense AI — Secure Database Intelligence & Analytics Engine."""

from schemasense.ai_agent import (
    AIAgentError,
    AIProviderError,
    AIQuotaExceededError,
    AISchemaHallucinationError,
    AITimeoutError,
    SchemaSenseAIAgent,
    SQLAgentResponse,
    create_agent,
    extract_sql_from_text,
    get_session_agent,
    verify_schema_fidelity,
)

__version__ = "1.0.0"

__all__ = [
    "AIAgentError",
    "AIProviderError",
    "AIQuotaExceededError",
    "AISchemaHallucinationError",
    "AITimeoutError",
    "SQLAgentResponse",
    "SchemaSenseAIAgent",
    "create_agent",
    "extract_sql_from_text",
    "get_session_agent",
    "verify_schema_fidelity",
]

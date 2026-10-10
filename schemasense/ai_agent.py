"""SchemaSense AI Agent — backend-neutral agent with structured outputs and schema grounding."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from schemasense.runtime.backends.base import (
    BackendInfo,
    BackendNotAvailableError,
    BackendResourceError,
    BackendTimeoutError,
    GenerationRequest,
    InferenceBackend,
    InferenceBackendError,
)
from schemasense.runtime.backends.gemini import (
    GOOGLE_GENAI_AVAILABLE,
    GeminiBackend,
    load_api_keys,
)
from schemasense.runtime.backends.llama_cpp import (
    LlamaCppBackend,
)

if TYPE_CHECKING:
    from collections.abc import Generator

# Backward-compatibility alias for Google SDK references
from schemasense.runtime.backends import gemini as _gemini_module

genai = getattr(_gemini_module, "genai", None)
types = getattr(_gemini_module, "types", None)

logger = logging.getLogger("schemasense.ai_agent")


# ──────────────────────────────────────────────
# STRUCTURED MODEL OUTPUT SCHEMA
# ──────────────────────────────────────────────
class SQLAgentResponse(BaseModel):
    """Structured analytical response produced by the SchemaSense AI Agent."""

    explanation: str = Field(
        description="Clear, professional explanation of the analytical logic, schema reasoning, or query intent."
    )
    sql_query: str | None = Field(
        default=None,
        description=(
            "Single executable SELECT query referencing only discovered schema tables."
            " Must not include markdown backticks or commentary."
        ),
    )
    target_tables: list[str] = Field(
        default_factory=list,
        description="List of database tables referenced in the sql_query.",
    )
    is_executable: bool = Field(
        default=False,
        description="Set to true if sql_query contains a complete, runnable SELECT statement.",
    )
    safety_notes: str | None = Field(
        default=None,
        description="Safety observations, such as joins used, row limits applied, or security considerations.",
    )


# ──────────────────────────────────────────────
# EXCEPTIONS
# ──────────────────────────────────────────────
class AIAgentError(Exception):
    """Base exception for SchemaSense AI Agent failures."""


class AITimeoutError(AIAgentError):
    """Raised when an AI API call exceeds its execution deadline."""


class AIQuotaExceededError(AIAgentError):
    """Raised when API quota or rate limits are exhausted across all available keys."""


class AIProviderError(AIAgentError):
    """Raised when the AI model provider returns an unrecoverable error."""


class AISchemaHallucinationError(AIAgentError):
    """Raised when the generated SQL query references tables absent from the schema context."""


# ──────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────
def extract_sql_from_text(text: str) -> str | None:
    """Robust regex fallback for extracting SQL statements from unstructured model output."""
    if not text:
        return None

    # Priority 1: Markdown fenced code block
    match = re.search(r"```(?:sql|SQL)\s*([\s\S]+?)```", text)
    if match:
        return match.group(1).strip()

    # Priority 2: Generic code fence with SELECT
    match = re.search(r"```\s*(SELECT[\s\S]+?)```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Priority 3: Bare SELECT statement ending with semicolon
    match = re.search(r"(SELECT\s[\s\S]+?;)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Priority 4: WITH clause CTE statement ending with semicolon
    match = re.search(r"(WITH\s[\s\S]+?;)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def verify_schema_fidelity(
    extracted_tables: Sequence[str], allowed_tables: set[str] | Sequence[str]
) -> list[str]:
    """Identifies tables referenced in a query that do not exist in the allowed schema context.

    Returns a list of hallucinated/unauthorized table names.
    """
    if not allowed_tables:
        return []

    allowed_normalized = {t.lower().strip() for t in allowed_tables}
    hallucinated = [
        t for t in extracted_tables if t.lower().strip() not in allowed_normalized
    ]
    return hallucinated


# ──────────────────────────────────────────────
# CORE AGENT CLASS
# ──────────────────────────────────────────────
class SchemaSenseAIAgent:
    """Enterprise AI Agent delegating inference to an InferenceBackend.

    Preserves structured outputs, schema grounding, multi-key rotation,
    and timeout bounds. Supports explicit backend selection between 'gemini'
    and 'llama_cpp'. Defaults to GeminiBackend when no backend is specified.
    """

    def __init__(
        self,
        api_keys: list[str] | None = None,
        model_name: str = "gemini-2.5-flash",
        timeout_seconds: float = 20.0,
        backend: InferenceBackend | None = None,
        backend_type: str = "gemini",
        model_path: str | Path | None = None,
    ) -> None:
        self.api_keys = api_keys if api_keys else load_api_keys()
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.backend_type = backend_type.lower().strip() if backend_type else "gemini"
        self.model_path = Path(model_path) if model_path is not None else None
        self._active_key_index = 0

        if backend is not None:
            self.backend = backend
        elif self.backend_type in ("llama_cpp", "local"):
            self.backend = LlamaCppBackend(
                model_path=self.model_path,
                n_ctx=2048,
                n_gpu_layers=0,
            )
        elif self.backend_type in ("gemini", "cloud"):
            self.backend = GeminiBackend(
                api_keys=self.api_keys,
                model_id=self.model_name,
                timeout_seconds=self.timeout_seconds,
            )
        else:
            raise ValueError(
                f"Unsupported backend_type '{backend_type}'. Supported values are 'gemini' and 'llama_cpp'."
            )

    @property
    def is_local(self) -> bool:
        """Returns True if the active backend performs in-process inference."""
        return self.backend.get_info().is_local

    @property
    def backend_info(self) -> BackendInfo:
        """Static identity and declared capabilities of the active backend."""
        return self.backend.get_info()

    @property
    def active_key_index(self) -> int:
        if hasattr(self.backend, "active_key_index"):
            return self.backend.active_key_index
        return self._active_key_index

    @active_key_index.setter
    def active_key_index(self, val: int) -> None:
        self._active_key_index = val
        if hasattr(self.backend, "active_key_index"):
            self.backend.active_key_index = val

    @property
    def active_key(self) -> str:
        if hasattr(self.backend, "active_key"):
            return self.backend.active_key
        if not self.api_keys:
            return "NO_KEY"
        return self.api_keys[self.active_key_index % len(self.api_keys)]

    def rotate_key(self) -> str:
        """Rotates to the next credential key in the pool."""
        if hasattr(self.backend, "rotate_key"):
            key = self.backend.rotate_key()
            self._active_key_index = getattr(self.backend, "active_key_index", 0)
            return key
        self._active_key_index = (self._active_key_index + 1) % len(self.api_keys)
        logger.info("Rotated to API key index %d", self._active_key_index)
        return self.active_key

    def close(self) -> None:
        """Explicitly release native model resources held by the active backend."""
        if hasattr(self.backend, "close"):
            self.backend.close()

    def _build_system_instruction(
        self,
        schema_context: str = "",
        business_rules_context: str = "",
        dialect: str = "postgres",
    ) -> str:
        """Builds strict, schema-grounded system instructions."""
        schema_block = (
            schema_context
            if schema_context.strip()
            else "No database connected. Provide general SQL guidance without referencing specific tables."
        )
        rules_block = (
            f"\n\nEnterprise Business Constraints:\n{business_rules_context}"
            if business_rules_context.strip()
            else ""
        )

        return (
            f"You are SchemaSense, a Principal Enterprise Data Architect and Analytics Agent.\n"
            f"Target Database Dialect: {dialect}\n\n"
            f"Database Schema Context:\n{schema_block}"
            f"{rules_block}\n\n"
            f"Mandatory Operational Guardrails:\n"
            f"1. SCHEMA GROUNDING: Reference ONLY tables and columns explicitly present in the schema context above.\n"
            f"   Never invent, hallucinate, or assume tables that are not listed.\n"
            f"2. READ-ONLY CONSTRAINT: Generate strictly single-statement SELECT queries.\n"
            f"   Never produce DROP, DELETE, INSERT, UPDATE, ALTER, TRUNCATE, or stacked semicolon queries.\n"
            f"3. BUSINESS RULE GROUNDING: Honor all enterprise business constraints.\n"
            f"4. OUTPUT FORMAT: When structured output is requested, provide explanation, valid sql_query, and target_tables.\n"
            f"   Wrap raw SQL in ```sql fences when generating unstructured stream text."
        )

    def generate_structured(
        self,
        prompt: str,
        schema_context: str = "",
        business_rules_context: str = "",
        dialect: str = "postgres",
        allowed_tables: set[str] | Sequence[str] | None = None,
    ) -> SQLAgentResponse:
        """Generates a validated structured response using the underlying backend and Pydantic."""
        if not self.backend.is_available():
            info = self.backend.get_info()
            if info.is_local:
                raise AIProviderError(
                    "Local inference backend (llama.cpp) is not available: either llama-cpp-python "
                    "is not installed, or the configured model_path is missing or invalid. "
                    "Cloud fallback is strictly disabled for local inference requests."
                )
            if not GOOGLE_GENAI_AVAILABLE:
                raise AIProviderError(
                    "The google-genai SDK is not installed. Install via: pip install google-genai"
                )
            raise AIProviderError(
                "Gemini cloud inference is not available: no valid API keys configured."
            )

        system_instruction = self._build_system_instruction(
            schema_context=schema_context,
            business_rules_context=business_rules_context,
            dialect=dialect,
        )

        request = GenerationRequest(
            prompt=prompt,
            system_instruction=system_instruction,
            structured_schema=SQLAgentResponse,
            temperature=0.1,
            timeout_seconds=self.timeout_seconds,
        )

        try:
            result = self.backend.generate(request)
        except BackendTimeoutError as te:
            raise AITimeoutError(f"AI API request timed out: {te}") from te
        except BackendResourceError as re:
            raise AIQuotaExceededError(
                f"Exhausted all available API keys due to rate limit/quota errors: {re}"
            ) from re
        except (BackendNotAvailableError, InferenceBackendError) as be:
            raise AIProviderError(f"AI provider error: {be}") from be

        if not result or not result.text:
            return SQLAgentResponse(
                explanation="No response received from the AI model.",
                is_executable=False,
            )

        # Parse into Pydantic model
        try:
            structured_output = SQLAgentResponse.model_validate_json(result.text)
        except Exception as parse_err:  # noqa: BLE001
            logger.warning(
                "Failed to parse JSON response directly: %s. Attempting fallback.",
                parse_err,
            )
            extracted_sql = extract_sql_from_text(result.text)
            structured_output = SQLAgentResponse(
                explanation=result.text,
                sql_query=extracted_sql,
                is_executable=extracted_sql is not None,
            )

        # Schema fidelity validation: catch hallucinated tables
        if structured_output.sql_query and allowed_tables:
            hallucinated = verify_schema_fidelity(
                structured_output.target_tables, allowed_tables
            )
            if hallucinated:
                structured_output.safety_notes = f"Hallucination Warning: Tables {hallucinated} were referenced but not present in allowed schema."
                structured_output.is_executable = False

        return structured_output

    def generate_stream(
        self,
        prompt: str,
        schema_context: str = "",
        business_rules_context: str = "",
        dialect: str = "postgres",
        history: list[dict[str, Any]] | None = None,
    ) -> Generator[str, None, None]:
        """Streams text chunks using the underlying inference backend."""
        if not self.backend.is_available():
            info = self.backend.get_info()
            if info.is_local:
                raise AIProviderError(
                    "Local inference backend (llama.cpp) is not available: either llama-cpp-python "
                    "is not installed, or the configured model_path is missing or invalid. "
                    "Cloud fallback is strictly disabled for local inference requests."
                )
            if not GOOGLE_GENAI_AVAILABLE:
                raise AIProviderError(
                    "The google-genai SDK is not installed. Install via: pip install google-genai"
                )
            raise AIProviderError(
                "Gemini cloud inference is not available: no valid API keys configured."
            )

        system_instruction = self._build_system_instruction(
            schema_context=schema_context,
            business_rules_context=business_rules_context,
            dialect=dialect,
        )

        formatted_history: list[dict[str, str]] = []
        if history:
            for msg in history:
                role = "user" if msg.get("role") == "user" else "assistant"
                content_text = str(msg.get("content", ""))
                if content_text:
                    formatted_history.append({"role": role, "content": content_text})

        request = GenerationRequest(
            prompt=prompt,
            system_instruction=system_instruction,
            history=formatted_history,
            temperature=0.2,
            timeout_seconds=self.timeout_seconds,
        )

        try:
            stream_iter = self.backend.generate_stream(request)
            for chunk in stream_iter:
                if chunk:
                    yield chunk
        except BackendTimeoutError as te:
            raise AITimeoutError(f"AI streaming request timed out: {te}") from te
        except BackendResourceError as re:
            raise AIQuotaExceededError(
                f"Exhausted all available API keys during streaming: {re}"
            ) from re
        except (BackendNotAvailableError, InferenceBackendError) as be:
            raise AIProviderError(f"AI streaming provider error: {be}") from be


def create_agent(
    backend_type: str = "gemini",
    model_name: str = "gemini-2.5-flash",
    model_path: str | Path | None = None,
    api_keys: list[str] | None = None,
    timeout_seconds: float = 20.0,
    backend: InferenceBackend | None = None,
) -> SchemaSenseAIAgent:
    """Factory creating a SchemaSenseAIAgent with explicit backend selection."""
    return SchemaSenseAIAgent(
        api_keys=api_keys,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
        backend=backend,
        backend_type=backend_type,
        model_path=model_path,
    )


def get_session_agent(
    model_name: str = "gemini-2.5-flash",
    backend_type: str | None = None,
    model_path: str | Path | None = None,
    custom_api_key: str | None = None,
    default_api_keys: list[str] | None = None,
    cache_store: Any = None,
    timeout_seconds: float = 20.0,
) -> SchemaSenseAIAgent:
    """Retrieve or instantiate a SchemaSenseAIAgent with session-isolated caching.

    Reuses existing agent instances when configuration (backend type, model name, model path)
    has not changed, avoiding redundant weight mapping into memory. When local mode is selected,
    fails closed with zero cloud fallback.
    """
    store = cache_store if cache_store is not None else {}

    if backend_type is None:
        mode_select = ""
        if hasattr(store, "get"):
            mode_select = str(store.get("backend_mode_select", ""))
        backend_choice = "llama_cpp" if "Local" in mode_select else "gemini"
    else:
        backend_choice = backend_type.lower().strip()

    if backend_choice in ("llama_cpp", "local"):
        chosen_path = model_path
        if chosen_path is None and hasattr(store, "get"):
            stored_path = store.get("local_model_path", "")
            if stored_path:
                chosen_path = str(stored_path).strip()
        elif chosen_path is not None:
            chosen_path = str(chosen_path).strip()

        if chosen_path:
            try:
                resolved_path = str(Path(chosen_path).resolve())
            except (ValueError, OSError):
                resolved_path = str(chosen_path)
        else:
            resolved_path = ""

        cache_key = ("llama_cpp", resolved_path)
        if (
            hasattr(store, "get")
            and store.get("_active_agent_cache_key") == cache_key
            and store.get("_active_agent") is not None
        ):
            return store.get("_active_agent")

        # Explicitly close previously cached agent before replacing
        old_agent = store.get("_active_agent") if hasattr(store, "get") else None
        if old_agent is not None and hasattr(old_agent, "close"):
            old_agent.close()

        agent = SchemaSenseAIAgent(
            backend_type="llama_cpp",
            model_path=resolved_path or None,
            timeout_seconds=timeout_seconds,
        )
        if hasattr(store, "__setitem__"):
            store["_active_agent"] = agent
            store["_active_agent_cache_key"] = cache_key
        return agent

    # Default backend: Google Gemini
    custom = custom_api_key
    if custom is None and hasattr(store, "get"):
        custom = str(store.get("custom_api_key", "")).strip()

    keys = [custom] if custom else (default_api_keys or load_api_keys())
    cache_key = ("gemini", model_name, str(custom or ""))
    if (
        hasattr(store, "get")
        and store.get("_active_agent_cache_key") == cache_key
        and store.get("_active_agent") is not None
    ):
        cached_agent = store.get("_active_agent")
        active_idx = store.get("active_key_index", 0) if hasattr(store, "get") else 0
        cached_agent.active_key_index = active_idx
        return cached_agent

    # Explicitly close previous agent native memory when switching to Gemini
    old_agent = store.get("_active_agent") if hasattr(store, "get") else None
    if old_agent is not None and hasattr(old_agent, "close"):
        old_agent.close()

    agent = SchemaSenseAIAgent(
        api_keys=keys,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
        backend_type="gemini",
    )
    if hasattr(store, "get"):
        agent.active_key_index = store.get("active_key_index", 0)
    if hasattr(store, "__setitem__"):
        store["_active_agent"] = agent
        store["_active_agent_cache_key"] = cache_key
    return agent

"""Comprehensive tests for GeminiBackend and SchemaSenseAIAgent integration (Step 2).

Test Areas
----------
* Gemini backend construction and availability without live API calls.
* Text generation and streaming through mocked SDK responses.
* Structured-output compatibility with SQLAgentResponse.
* API-key rotation and placeholder-key rejection (credentials never logged).
* SDK error mapping and timeout/deadline behavior.
* Distinction between caller timeout and underlying worker cancellation.
* Backward compatibility of SchemaSenseAIAgent public API.
* Deterministic SQLGuard safety boundary on all generated-SQL paths.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from schemasense.ai_agent import (
    AIProviderError,
    AIQuotaExceededError,
    AITimeoutError,
    SchemaSenseAIAgent,
    SQLAgentResponse,
)
from schemasense.runtime.backends.base import (
    BackendCancelledError,
    BackendCapability,
    BackendInfo,
    BackendResourceError,
    BackendTimeoutError,
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
)
from schemasense.runtime.backends.gemini import (
    GeminiBackend,
    is_placeholder_key,
)
from schemasense.sql_guard import execute_safe_query

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONSTRUCTION & AVAILABILITY (NO LIVE CALLS)
# ──────────────────────────────────────────────────────────────────────────────


class TestGeminiBackendConstruction:
    """Verify GeminiBackend construction, metadata, and zero-IO checks."""

    def test_construction_defaults(self):
        backend = GeminiBackend(api_keys=["TEST_KEY_ALPHA_123"])
        assert backend.model_id == "gemini-2.5-flash"
        assert backend.timeout_seconds == 20.0
        assert backend.active_key_index == 0
        assert backend.active_key == "TEST_KEY_ALPHA_123"

    def test_get_info_metadata(self):
        backend = GeminiBackend(
            api_keys=["TEST_KEY_ALPHA_123"],
            model_id="gemini-2.5-flash",
        )
        info = backend.get_info()
        assert isinstance(info, BackendInfo)
        assert info.backend_id == "gemini"
        assert info.display_name == "Google Gemini (cloud)"
        assert info.is_local is False
        assert info.requires_api_key is True
        assert info.model_id == "gemini-2.5-flash"
        assert info.engine_version == "google-genai"
        assert BackendCapability.STREAMING in info.capabilities
        assert BackendCapability.STRUCTURED_OUTPUT in info.capabilities
        # Single-shot in-flight RPCs cannot be hard-cancelled; must NOT declare CANCELLATION
        assert BackendCapability.CANCELLATION not in info.capabilities

    def test_availability_with_valid_keys(self):
        backend = GeminiBackend(api_keys=["VALID_KEY_12345678"])
        assert backend.is_available() is True

    def test_availability_fails_with_empty_keys(self):
        backend = GeminiBackend(api_keys=[])
        assert backend.is_available() is False

    def test_availability_fails_with_only_placeholders(self):
        for placeholder in [
            "DEMO_KEY_STANDBY",
            "NO_KEY",
            "CHANGE_ME",
            "YOUR_API_KEY",
            "YOUR_GEMINI_API_KEY_HERE",
            "",
            "   ",
        ]:
            backend = GeminiBackend(api_keys=[placeholder])
            assert backend.is_available() is False, (
                f"Failed for placeholder: {placeholder}"
            )

    def test_availability_latency_under_100ms(self):
        backend = GeminiBackend(api_keys=["VALID_KEY_12345678"])
        start = time.perf_counter()
        for _ in range(100):
            _ = backend.is_available()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"100 calls took {elapsed:.4f}s"

    @patch("schemasense.runtime.backends.gemini.GOOGLE_GENAI_AVAILABLE", False)
    def test_availability_false_when_sdk_missing(self):
        backend = GeminiBackend(api_keys=["VALID_KEY_12345678"])
        assert backend.is_available() is False

    def test_placeholder_detector_patterns(self):
        assert is_placeholder_key(None) is True
        assert is_placeholder_key("") is True
        assert is_placeholder_key("   ") is True
        assert is_placeholder_key("DEMO_KEY_STANDBY") is True
        assert is_placeholder_key("demo_key_standby") is True
        assert is_placeholder_key("NO_KEY") is True
        assert is_placeholder_key("YOUR_API_KEY") is True
        assert is_placeholder_key("YOUR-GEMINI-API-KEY-HERE") is True
        # Real/synthetic test keys are not placeholders
        assert is_placeholder_key("AIzaSyD-RandomRealLookingKey123") is False
        assert is_placeholder_key("KEY_ALPHA_1234") is False


# ──────────────────────────────────────────────────────────────────────────────
# 2. KEY ROTATION & CREDENTIAL HYGIENE
# ──────────────────────────────────────────────────────────────────────────────


class TestGeminiBackendKeyRotation:
    """Verify key rotation, placeholder skipping, and credential redaction."""

    def test_key_rotation_cycles_cleanly(self):
        keys = ["KEY_A_1", "KEY_B_2", "KEY_C_3"]
        backend = GeminiBackend(api_keys=keys)
        assert backend.active_key == "KEY_A_1"
        assert backend.rotate_key() == "KEY_B_2"
        assert backend.active_key_index == 1
        assert backend.rotate_key() == "KEY_C_3"
        assert backend.rotate_key() == "KEY_A_1"

    def test_active_key_index_setter_modulo(self):
        keys = ["KEY_A_1", "KEY_B_2"]
        backend = GeminiBackend(api_keys=keys)
        backend.active_key_index = 5
        assert backend.active_key_index == 1
        assert backend.active_key == "KEY_B_2"

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_rotation_on_429_quota_exhaustion(self, mock_client_class):
        keys = ["KEY_FAIL_1", "KEY_SUCCESS_2"]
        backend = GeminiBackend(api_keys=keys)

        mock_resp = MagicMock()
        mock_resp.text = "Analysis complete."
        mock_resp.usage_metadata.prompt_token_count = 10
        mock_resp.usage_metadata.candidates_token_count = 5

        client_fail = MagicMock()
        client_fail.models.generate_content.side_effect = Exception(
            "429 Resource has been exhausted (quota limit)"
        )
        client_ok = MagicMock()
        client_ok.models.generate_content.return_value = mock_resp

        mock_client_class.side_effect = [client_fail, client_ok]

        request = GenerationRequest(prompt="Show summary")
        result = backend.generate(request)

        assert result.text == "Analysis complete."
        assert backend.active_key == "KEY_SUCCESS_2"
        assert result.tokens_prompt == 10
        assert result.tokens_completion == 5

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_quota_exhaustion_raises_resource_error(self, mock_client_class):
        keys = ["KEY_1", "KEY_2"]
        backend = GeminiBackend(api_keys=keys)

        client_mock = MagicMock()
        client_mock.models.generate_content.side_effect = Exception(
            "429 RateLimitExceeded quota limit reached"
        )
        mock_client_class.return_value = client_mock

        with pytest.raises(BackendResourceError) as exc_info:
            backend.generate(GenerationRequest(prompt="Test"))

        # Verify key credentials are not leaked in the exception message
        assert "KEY_1" not in str(exc_info.value)
        assert "KEY_2" not in str(exc_info.value)

    def test_telemetry_does_not_expose_credentials(self):
        backend = GeminiBackend(api_keys=["SECRET_KEY_ALPHA_999"])
        telemetry = backend.get_resource_telemetry()
        assert "SECRET_KEY_ALPHA_999" not in str(telemetry)
        assert telemetry["backend_id"] == "gemini"
        assert telemetry["total_keys"] == 1
        assert telemetry["active_key_index"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# 3. TEXT GENERATION & STRUCTURED OUTPUT
# ──────────────────────────────────────────────────────────────────────────────


class TestGeminiBackendGeneration:
    """Verify single-shot generation and structured output enforcement."""

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_generate_text_success(self, mock_client_class):
        backend = GeminiBackend(api_keys=["VALID_KEY_123"])
        mock_resp = MagicMock()
        mock_resp.text = "SELECT * FROM orders;"
        mock_resp.usage_metadata.prompt_token_count = 15
        mock_resp.usage_metadata.candidates_token_count = 8

        client = MagicMock()
        client.models.generate_content.return_value = mock_resp
        mock_client_class.return_value = client

        request = GenerationRequest(
            prompt="Get all orders",
            system_instruction="You are a SQL assistant.",
            temperature=0.1,
            max_tokens=256,
        )
        result = backend.generate(request)

        assert isinstance(result, GenerationResult)
        assert result.text == "SELECT * FROM orders;"
        assert result.backend_id == "gemini"
        assert result.model_id == "gemini-2.5-flash"
        assert result.finish_reason == "stop"
        assert result.tokens_prompt == 15
        assert result.tokens_completion == 8
        assert result.elapsed_ms >= 0.0

        # Verify SDK call parameters
        client.models.generate_content.assert_called_once()
        call_kwargs = client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["contents"] == "Get all orders"

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_generate_with_history(self, mock_client_class):
        backend = GeminiBackend(api_keys=["VALID_KEY_123"])
        mock_resp = MagicMock()
        mock_resp.text = "Second turn answer."
        client = MagicMock()
        client.models.generate_content.return_value = mock_resp
        mock_client_class.return_value = client

        request = GenerationRequest(
            prompt="Follow up question",
            history=[
                {"role": "user", "content": "First turn question"},
                {"role": "assistant", "content": "First turn answer"},
            ],
        )
        result = backend.generate(request)
        assert result.text == "Second turn answer."

        # Verify history was formatted into contents list
        call_kwargs = client.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]
        assert isinstance(contents, list)
        assert len(contents) == 3

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_generate_structured_schema_config(self, mock_client_class):
        backend = GeminiBackend(api_keys=["VALID_KEY_123"])
        mock_resp = MagicMock()
        mock_resp.text = (
            '{"explanation": "Valid", "sql_query": "SELECT 1;", '
            '"target_tables": [], "is_executable": true}'
        )
        client = MagicMock()
        client.models.generate_content.return_value = mock_resp
        mock_client_class.return_value = client

        request = GenerationRequest(
            prompt="Generate structured query",
            structured_schema=SQLAgentResponse,
        )
        result = backend.generate(request)
        assert result.text == mock_resp.text

        # Verify config received response_schema
        call_kwargs = client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.response_schema == SQLAgentResponse
        assert config.response_mime_type == "application/json"


# ──────────────────────────────────────────────────────────────────────────────
# 4. STREAMING GENERATION
# ──────────────────────────────────────────────────────────────────────────────


class TestGeminiBackendStreaming:
    """Verify streaming generation, chunk delivery, and stream cancellation."""

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_generate_stream_success(self, mock_client_class):
        backend = GeminiBackend(api_keys=["VALID_KEY_123"])

        chunk1 = MagicMock()
        chunk1.text = "SELECT id, "
        chunk2 = MagicMock()
        chunk2.text = "name FROM "
        chunk3 = MagicMock()
        chunk3.text = "users;"

        client = MagicMock()
        client.models.generate_content_stream.return_value = [chunk1, chunk2, chunk3]
        mock_client_class.return_value = client

        request = GenerationRequest(prompt="Stream query")
        chunks = list(backend.generate_stream(request))

        assert chunks == ["SELECT id, ", "name FROM ", "users;"]
        assert "".join(chunks) == "SELECT id, name FROM users;"

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_generate_stream_cancellation_during_yield(self, mock_client_class):
        backend = GeminiBackend(api_keys=["VALID_KEY_123"])

        cancel_token = threading.Event()

        def stream_generator():
            c1 = MagicMock()
            c1.text = "Chunk 1"
            yield c1
            # Signal cancellation before chunk 2
            cancel_token.set()
            c2 = MagicMock()
            c2.text = "Chunk 2"
            yield c2

        client = MagicMock()
        client.models.generate_content_stream.return_value = stream_generator()
        mock_client_class.return_value = client

        request = GenerationRequest(prompt="Test stream cancel")
        gen = backend.generate_stream(request, cancel_token=cancel_token)

        first_chunk = next(gen)
        assert first_chunk == "Chunk 1"

        with pytest.raises(BackendCancelledError) as exc_info:
            next(gen)
        assert "incomplete" in str(exc_info.value).lower()

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_stream_timeout_deadline(self, mock_client_class):
        backend = GeminiBackend(api_keys=["VALID_KEY_123"])

        def slow_stream():
            c1 = MagicMock()
            c1.text = "Fast chunk"
            yield c1
            time.sleep(0.05)
            c2 = MagicMock()
            c2.text = "Late chunk"
            yield c2

        client = MagicMock()
        client.models.generate_content_stream.return_value = slow_stream()
        mock_client_class.return_value = client

        # 20ms timeout
        request = GenerationRequest(prompt="Slow stream", timeout_seconds=0.02)
        gen = backend.generate_stream(request)

        first = next(gen)
        assert first == "Fast chunk"

        with pytest.raises(BackendTimeoutError):
            next(gen)


# ──────────────────────────────────────────────────────────────────────────────
# 5. TIMEOUTS, CANCELLATION & WORKER DISTINCTION
# ──────────────────────────────────────────────────────────────────────────────


class TestGeminiTimeoutSemantics:
    """Verify explicit timeout semantics and worker termination distinctions."""

    def test_pre_set_cancel_token_aborts_without_io(self):
        backend = GeminiBackend(api_keys=["VALID_KEY_123"])
        cancel_token = threading.Event()
        cancel_token.set()

        request = GenerationRequest(prompt="Never runs")
        with pytest.raises(BackendCancelledError):
            backend.generate(request, cancel_token=cancel_token)

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_sdk_timeout_mapped_to_backend_timeout_error(self, mock_client_class):
        backend = GeminiBackend(api_keys=["VALID_KEY_123"])

        client = MagicMock()
        client.models.generate_content.side_effect = Exception(
            "504 Deadline Exceeded (request timed out)"
        )
        mock_client_class.return_value = client

        request = GenerationRequest(prompt="Will time out", timeout_seconds=5.0)
        with pytest.raises(BackendTimeoutError) as exc_info:
            backend.generate(request)

        # Confirm documentation limitation is present
        assert "deadline does not guarantee termination" in str(exc_info.value)

    def test_caller_side_timeout_does_not_imply_worker_dead(self):
        """Demonstrate that an abandoned background thread is NOT forcibly terminated,

        and late results must be discarded.
        """
        worker_finished = threading.Event()
        late_result_captured = []

        def slow_mock_worker():
            time.sleep(0.05)
            worker_finished.set()
            return "Late finished result"

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(slow_mock_worker)

        # Caller deadline is 0.01s
        with pytest.raises(concurrent.futures.TimeoutError):
            future.result(timeout=0.01)

        # Crucial invariant: The worker is STILL RUNNING despite caller timeout
        assert not worker_finished.is_set(), "Worker should still be in-flight"

        # Wait for worker to finish in background
        worker_finished.wait(timeout=0.2)
        assert worker_finished.is_set()

        # The result exists on the completed future, but caller discarded it
        assert future.result() == "Late finished result"
        # Caller never assigned it to late_result_captured
        assert late_result_captured == []

        executor.shutdown(wait=False)


# ──────────────────────────────────────────────────────────────────────────────
# 6. SCHEMASENSE AI AGENT BACKWARD COMPATIBILITY
# ──────────────────────────────────────────────────────────────────────────────


class TestSchemaSenseAIAgentIntegration:
    """Verify SchemaSenseAIAgent operates seamlessly with GeminiBackend."""

    def test_agent_initialization_defaults(self):
        agent = SchemaSenseAIAgent(api_keys=["VALID_KEY_ABC"])
        assert isinstance(agent.backend, GeminiBackend)
        assert agent.active_key == "VALID_KEY_ABC"
        assert agent.timeout_seconds == 20.0

    def test_agent_custom_backend_injection(self):
        mock_backend = MagicMock(spec=InferenceBackend)
        mock_backend.is_available.return_value = True
        mock_backend.active_key = "INJECTED_KEY"
        mock_backend.active_key_index = 0

        agent = SchemaSenseAIAgent(backend=mock_backend)
        assert agent.backend is mock_backend
        assert agent.active_key == "INJECTED_KEY"

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_agent_generate_structured_success(self, mock_client_class):
        agent = SchemaSenseAIAgent(api_keys=["VALID_KEY_123"])

        mock_resp = MagicMock()
        mock_resp.text = (
            '{"explanation": "Lists customers.", '
            '"sql_query": "SELECT id, name FROM customers;", '
            '"target_tables": ["customers"], "is_executable": true}'
        )
        client = MagicMock()
        client.models.generate_content.return_value = mock_resp
        mock_client_class.return_value = client

        response = agent.generate_structured(
            prompt="List customers",
            allowed_tables={"customers"},
        )
        assert isinstance(response, SQLAgentResponse)
        assert response.is_executable is True
        assert response.sql_query == "SELECT id, name FROM customers;"
        assert response.target_tables == ["customers"]

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_agent_translates_timeout_exception(self, mock_client_class):
        agent = SchemaSenseAIAgent(api_keys=["VALID_KEY_123"])

        client = MagicMock()
        client.models.generate_content.side_effect = Exception("Deadline Exceeded")
        mock_client_class.return_value = client

        with pytest.raises(AITimeoutError):
            agent.generate_structured("Prompt")

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_agent_translates_quota_exception(self, mock_client_class):
        agent = SchemaSenseAIAgent(api_keys=["VALID_KEY_123"])

        client = MagicMock()
        client.models.generate_content.side_effect = Exception("429 Quota Exceeded")
        mock_client_class.return_value = client

        with pytest.raises(AIQuotaExceededError):
            agent.generate_structured("Prompt")

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_agent_translates_provider_exception(self, mock_client_class):
        agent = SchemaSenseAIAgent(api_keys=["VALID_KEY_123"])

        client = MagicMock()
        client.models.generate_content.side_effect = Exception("500 Internal error")
        mock_client_class.return_value = client

        with pytest.raises(AIProviderError):
            agent.generate_structured("Prompt")

    @patch("schemasense.runtime.backends.gemini.genai.Client")
    def test_agent_generate_stream_delegation(self, mock_client_class):
        agent = SchemaSenseAIAgent(api_keys=["VALID_KEY_123"])

        c1 = MagicMock()
        c1.text = "Hello "
        c2 = MagicMock()
        c2.text = "world!"
        client = MagicMock()
        client.models.generate_content_stream.return_value = [c1, c2]
        mock_client_class.return_value = client

        chunks = list(agent.generate_stream("Say hello"))
        assert chunks == ["Hello ", "world!"]


# ──────────────────────────────────────────────────────────────────────────────
# 7. DETERMINISTIC SQLGUARD BOUNDARY
# ──────────────────────────────────────────────────────────────────────────────


class TestSQLGuardBoundaryIntegration:
    """Verify generated SQL cannot bypass deterministic SQLGuard validation."""

    def test_adversarial_backend_sql_blocked_by_sqlguard(self, sqlite_engine):
        """Even if the model generates a malicious DROP statement, SQLGuard blocks it."""
        malicious_generated_sql = "DROP TABLE customers;"

        with pytest.raises(Exception) as exc_info:
            execute_safe_query(
                engine=sqlite_engine,
                query=malicious_generated_sql,
                allowed_tables={"customers"},
                dialect="sqlite",
            )
        assert (
            "security violation" in str(exc_info.value).lower()
            or "read-only" in str(exc_info.value).lower()
        )

    def test_stacked_multi_statement_blocked(self, sqlite_engine):
        stacked_sql = "SELECT * FROM customers; DROP TABLE orders;"

        with pytest.raises(Exception) as exc_info:
            execute_safe_query(
                engine=sqlite_engine,
                query=stacked_sql,
                allowed_tables={"customers", "orders"},
                dialect="sqlite",
            )
        assert (
            "multi-statement" in str(exc_info.value).lower()
            or "security" in str(exc_info.value).lower()
        )

    def test_unauthorized_table_blocked(self, sqlite_engine):
        unauthorized_sql = "SELECT * FROM secret_passwords;"

        with pytest.raises(Exception) as exc_info:
            execute_safe_query(
                engine=sqlite_engine,
                query=unauthorized_sql,
                allowed_tables={"customers"},
                dialect="sqlite",
            )
        assert "unauthorized table" in str(exc_info.value).lower()

    def test_legitimate_generated_query_executes_safely(self, sqlite_engine):
        valid_sql = "SELECT id, name FROM customers;"
        df, is_truncated = execute_safe_query(
            engine=sqlite_engine,
            query=valid_sql,
            allowed_tables={"customers"},
            dialect="sqlite",
        )
        assert not df.empty
        assert is_truncated is False
        assert "name" in df.columns

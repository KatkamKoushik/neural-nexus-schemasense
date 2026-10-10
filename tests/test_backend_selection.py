"""Unit, integration, and UI tests for SchemaSense Backend Selection (Milestone 8A Step 4).

Covers:
* Backward-compatible default Gemini agent construction.
* Explicit local (llama.cpp) backend selection and model-path configuration.
* Fail-closed behavior: local inference unavailability raises actionable errors
  with strictly zero silent fallback to Gemini or any cloud provider.
* Validation and actionable error messages for missing dependencies / invalid model paths.
* Session-state agent caching and isolation in Streamlit runtime.
* Streamlit AppTest integration verifying backend switching and UI indicators.
* End-to-end SQLGuard enforcement along the execution path for generated queries.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from schemasense.ai_agent import (
    AIProviderError,
    SchemaSenseAIAgent,
    SQLAgentResponse,
    create_agent,
    get_session_agent,
)
from schemasense.runtime.backends.base import (
    BackendCapability,
    BackendInfo,
    BackendNotAvailableError,
    GenerationResult,
    InferenceBackend,
)
from schemasense.runtime.backends.gemini import GeminiBackend
from schemasense.runtime.backends.llama_cpp import LlamaCppBackend
from schemasense.sql_guard import (
    SQLSecurityViolation,
    execute_safe_query,
)

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


# ──────────────────────────────────────────────────────────────────────────────
# 1. AGENT INITIALIZATION & BACKWARD COMPATIBILITY
# ──────────────────────────────────────────────────────────────────────────────


class TestAgentBackendSelectionConstruction:
    """Verifies construction semantics, defaulting, and explicit configuration."""

    def test_default_construction_preserves_gemini_backend(self):
        """Omitting backend_type defaults to Gemini with backward-compatible defaults."""
        agent = SchemaSenseAIAgent(api_keys=["TEST_KEY_123"])
        assert agent.backend_type == "gemini"
        assert agent.is_local is False
        assert isinstance(agent.backend, GeminiBackend)
        assert agent.backend_info.backend_id == "gemini"
        assert agent.backend_info.is_local is False

    def test_explicit_gemini_backend_type(self):
        """Explicitly requesting 'gemini' configures GeminiBackend."""
        agent = SchemaSenseAIAgent(
            api_keys=["TEST_KEY_ABC"],
            model_name="gemini-2.5-pro",
            backend_type="gemini",
        )
        assert agent.backend_type == "gemini"
        assert agent.is_local is False
        assert agent.backend.model_id == "gemini-2.5-pro"

    def test_explicit_llama_cpp_backend_type(self, tmp_path):
        """Explicitly requesting 'llama_cpp' configures LlamaCppBackend."""
        model_file = tmp_path / "sql_coder.gguf"
        model_file.write_bytes(b"GGUF\x03\x00\x00\x00")

        agent = SchemaSenseAIAgent(
            backend_type="llama_cpp",
            model_path=model_file,
        )
        assert agent.backend_type == "llama_cpp"
        assert agent.is_local is True
        assert isinstance(agent.backend, LlamaCppBackend)
        assert agent.backend_info.backend_id == "llama_cpp"
        assert agent.backend_info.is_local is True
        assert agent.model_path == model_file

    def test_backend_type_aliases_normalize_correctly(self, tmp_path):
        """Aliases 'cloud' and 'local' map cleanly to gemini and llama_cpp."""
        agent_cloud = SchemaSenseAIAgent(
            api_keys=["TEST_KEY"],
            backend_type="cloud",
        )
        assert isinstance(agent_cloud.backend, GeminiBackend)

        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")
        agent_local = SchemaSenseAIAgent(
            backend_type="local",
            model_path=dummy_gguf,
        )
        assert isinstance(agent_local.backend, LlamaCppBackend)

    def test_unsupported_backend_type_raises_value_error(self):
        """Unsupported backend types raise ValueError with actionable guidance."""
        with pytest.raises(ValueError, match="Unsupported backend_type 'vllm'"):
            SchemaSenseAIAgent(backend_type="vllm")

    def test_create_agent_factory_function(self, tmp_path):
        """create_agent factory mirrors SchemaSenseAIAgent parameters."""
        dummy_gguf = tmp_path / "test.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        agent = create_agent(
            backend_type="llama_cpp",
            model_path=dummy_gguf,
            timeout_seconds=15.0,
        )
        assert agent.backend_type == "llama_cpp"
        assert agent.timeout_seconds == 15.0
        assert agent.is_local is True

    def test_custom_backend_instance_injection_bypasses_factory(self):
        """Passing an explicit backend instance overrides backend_type reflection."""
        mock_backend = MagicMock(spec=InferenceBackend)
        mock_backend.get_info.return_value = BackendInfo(
            backend_id="custom_mock",
            display_name="Custom Mock",
            is_local=True,
            capabilities=frozenset({BackendCapability.STRUCTURED_OUTPUT}),
        )
        agent = SchemaSenseAIAgent(backend=mock_backend)
        assert agent.backend is mock_backend
        assert agent.is_local is True
        assert agent.backend_info.backend_id == "custom_mock"


# ──────────────────────────────────────────────────────────────────────────────
# 2. PRIVACY & FAIL-CLOSED LOCAL INFERENCE GUARANTEES
# ──────────────────────────────────────────────────────────────────────────────


class TestLocalBackendFailClosedPolicy:
    """Verifies that local backend failures fail closed with zero cloud fallback."""

    def test_missing_llama_cpp_dependency_fails_closed_without_cloud_fallback(
        self, tmp_path
    ):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        with (
            patch("schemasense.runtime.backends.llama_cpp.LLAMA_CPP_AVAILABLE", False),
            patch("schemasense.runtime.backends.llama_cpp.llama_cpp", None),
            patch.object(GeminiBackend, "generate") as mock_gemini_generate,
        ):
            agent = SchemaSenseAIAgent(
                backend_type="llama_cpp",
                model_path=dummy_gguf,
            )

            with pytest.raises(AIProviderError) as exc_info:
                agent.generate_structured(prompt="SELECT 1;")

            assert "Cloud fallback is strictly disabled" in str(exc_info.value)
            assert "llama-cpp-python is not installed" in str(exc_info.value)
            # Guarantee Gemini was never called
            mock_gemini_generate.assert_not_called()

    def test_missing_model_path_fails_closed(self):
        agent = SchemaSenseAIAgent(
            backend_type="llama_cpp",
            model_path=None,
        )
        with pytest.raises(AIProviderError) as exc_info:
            agent.generate_structured(prompt="SELECT 1;")

        assert "Cloud fallback is strictly disabled" in str(exc_info.value)

    def test_nonexistent_model_path_fails_closed(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.gguf"
        agent = SchemaSenseAIAgent(
            backend_type="llama_cpp",
            model_path=missing_path,
        )
        with pytest.raises(AIProviderError) as exc_info:
            agent.generate_structured(prompt="SELECT 1;")

        assert "Cloud fallback is strictly disabled" in str(exc_info.value)

    def test_streaming_fails_closed_when_local_unavailable(self, tmp_path):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        with (
            patch("schemasense.runtime.backends.llama_cpp.LLAMA_CPP_AVAILABLE", False),
            patch("schemasense.runtime.backends.llama_cpp.llama_cpp", None),
            patch.object(GeminiBackend, "generate_stream") as mock_gemini_stream,
        ):
            agent = SchemaSenseAIAgent(
                backend_type="llama_cpp",
                model_path=dummy_gguf,
            )

            with pytest.raises(AIProviderError) as exc_info:
                list(agent.generate_stream(prompt="Show tables"))

            assert "Cloud fallback is strictly disabled" in str(exc_info.value)
            mock_gemini_stream.assert_not_called()

    def test_backend_error_does_not_silently_fall_back(self):
        """Backend runtime exceptions in local mode fail directly rather than falling back."""
        mock_backend = MagicMock(spec=InferenceBackend)
        mock_backend.is_available.return_value = True
        mock_backend.get_info.return_value = BackendInfo(
            backend_id="llama_cpp",
            display_name="llama.cpp",
            is_local=True,
            capabilities=frozenset({BackendCapability.STRUCTURED_OUTPUT}),
        )
        mock_backend.generate.side_effect = BackendNotAvailableError(
            "Out of VRAM allocating context"
        )

        with patch.object(GeminiBackend, "generate") as mock_gemini_generate:
            agent = SchemaSenseAIAgent(backend=mock_backend, backend_type="llama_cpp")
            with pytest.raises(AIProviderError, match="Out of VRAM"):
                agent.generate_structured(prompt="SELECT 1;")

            mock_gemini_generate.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# 3. STRUCTURED GENERATION & STREAMING THROUGH LOCAL BACKEND
# ──────────────────────────────────────────────────────────────────────────────


class TestSelectedBackendExecution:
    """Verifies that SchemaSense structured reasoning and streaming work through the selected backend."""

    def test_structured_generation_via_local_backend(self):
        mock_backend = MagicMock(spec=InferenceBackend)
        mock_backend.is_available.return_value = True
        mock_backend.get_info.return_value = BackendInfo(
            backend_id="llama_cpp",
            display_name="llama.cpp",
            is_local=True,
            capabilities=frozenset({BackendCapability.STRUCTURED_OUTPUT}),
        )
        mock_backend.generate.return_value = GenerationResult(
            text='{"explanation": "Queries customers table.", "sql_query": "SELECT id, name FROM customers;", "target_tables": ["customers"], "is_executable": true}',
            model_id="test-model.gguf",
            backend_id="llama_cpp",
        )

        agent = SchemaSenseAIAgent(backend=mock_backend, backend_type="llama_cpp")
        response = agent.generate_structured(
            prompt="List customers",
            allowed_tables=["customers"],
        )
        assert isinstance(response, SQLAgentResponse)
        assert response.explanation == "Queries customers table."
        assert response.sql_query == "SELECT id, name FROM customers;"
        assert response.target_tables == ["customers"]
        assert response.is_executable is True

    def test_schema_hallucination_detected_on_local_output(self):
        mock_backend = MagicMock(spec=InferenceBackend)
        mock_backend.is_available.return_value = True
        mock_backend.get_info.return_value = BackendInfo(
            backend_id="llama_cpp",
            display_name="llama.cpp",
            is_local=True,
            capabilities=frozenset({BackendCapability.STRUCTURED_OUTPUT}),
        )
        mock_backend.generate.return_value = GenerationResult(
            text='{"explanation": "Joining orders and unauthorized_log.", "sql_query": "SELECT * FROM unauthorized_log;", "target_tables": ["unauthorized_log"], "is_executable": true}',
            model_id="test-model.gguf",
            backend_id="llama_cpp",
        )

        agent = SchemaSenseAIAgent(backend=mock_backend, backend_type="llama_cpp")
        response = agent.generate_structured(
            prompt="Show audit logs",
            allowed_tables=["customers", "orders"],
        )
        assert response.is_executable is False
        assert "Hallucination Warning" in response.safety_notes
        assert "unauthorized_log" in response.safety_notes

    def test_streaming_generation_via_local_backend(self):
        mock_backend = MagicMock(spec=InferenceBackend)
        mock_backend.is_available.return_value = True
        mock_backend.get_info.return_value = BackendInfo(
            backend_id="llama_cpp",
            display_name="llama.cpp",
            is_local=True,
            capabilities=frozenset({BackendCapability.STREAMING}),
        )
        mock_backend.generate_stream.return_value = iter(
            ["SELECT ", "* FROM ", "orders;"]
        )

        agent = SchemaSenseAIAgent(backend=mock_backend, backend_type="llama_cpp")
        chunks = list(agent.generate_stream(prompt="Show orders"))
        assert chunks == ["SELECT ", "* FROM ", "orders;"]


# ──────────────────────────────────────────────────────────────────────────────
# 4. STREAMLIT APP AGENT REUSE & SESSION ISOLATION
# ──────────────────────────────────────────────────────────────────────────────


class TestStreamlitAgentLifecycle:
    """Verifies that get_session_agent reuses models safely within a session without cross-configuration leaks."""

    def test_agent_reuse_within_session(self):
        cache_store = {
            "backend_mode_select": "Google Gemini (Cloud)",
            "custom_api_key": "MOCK_KEY_1",
            "active_key_index": 0,
        }
        agent1 = get_session_agent(
            model_name="gemini-2.5-flash",
            cache_store=cache_store,
        )
        agent2 = get_session_agent(
            model_name="gemini-2.5-flash",
            cache_store=cache_store,
        )
        assert agent1 is agent2

    def test_agent_cache_invalidated_when_backend_changes(self, tmp_path):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        cache_store = {
            "backend_mode_select": "Google Gemini (Cloud)",
            "custom_api_key": "MOCK_KEY_1",
            "active_key_index": 0,
        }
        agent_gemini = get_session_agent(
            model_name="gemini-2.5-flash",
            cache_store=cache_store,
        )
        assert agent_gemini.backend_type == "gemini"

        # Switch session to local mode
        cache_store["backend_mode_select"] = "llama.cpp (Local GGUF)"
        cache_store["local_model_path"] = str(dummy_gguf)

        agent_local = get_session_agent(
            model_name="local-model",
            cache_store=cache_store,
        )
        assert agent_local is not agent_gemini
        assert agent_local.backend_type == "llama_cpp"

    def test_agent_cache_invalidated_when_local_path_changes(self, tmp_path):
        model1 = tmp_path / "model1.gguf"
        model1.write_bytes(b"GGUF\x03\x00\x00\x00")
        model2 = tmp_path / "model2.gguf"
        model2.write_bytes(b"GGUF\x03\x00\x00\x00")

        cache_store = {
            "backend_mode_select": "llama.cpp (Local GGUF)",
            "local_model_path": str(model1),
            "active_key_index": 0,
        }
        agent1 = get_session_agent(
            model_name="model1.gguf",
            cache_store=cache_store,
        )
        assert agent1.model_path == model1

        cache_store["local_model_path"] = str(model2)
        agent2 = get_session_agent(
            model_name="model2.gguf",
            cache_store=cache_store,
        )
        assert agent2 is not agent1
        assert agent2.model_path == model2

    def test_session_isolation_independent_caches(self):
        cache1 = {"custom_api_key": "KEY_1"}
        cache2 = {"custom_api_key": "KEY_2"}
        agent1 = get_session_agent(cache_store=cache1)
        agent2 = get_session_agent(cache_store=cache2)
        assert agent1 is not agent2

    def test_path_normalization_reuses_cached_agent(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF\x03\x00\x00\x00")

        cache_store = {
            "backend_mode_select": "llama.cpp (Local GGUF)",
            "local_model_path": str(model_file),
        }
        agent1 = get_session_agent(
            model_name="model.gguf",
            cache_store=cache_store,
        )

        # Access using resolved path equivalent
        cache_store["local_model_path"] = str(model_file.resolve())
        agent2 = get_session_agent(
            model_name="model.gguf",
            cache_store=cache_store,
        )
        assert agent1 is agent2

    def test_explicit_agent_close_releases_native_model(self):
        mock_backend = MagicMock(spec=InferenceBackend)
        mock_backend.close = MagicMock()
        agent = SchemaSenseAIAgent(backend=mock_backend)
        agent.close()
        mock_backend.close.assert_called_once()

    def test_cache_invalidation_closes_superseded_agent(self, tmp_path):
        model1 = tmp_path / "m1.gguf"
        model1.write_bytes(b"GGUF\x03\x00\x00\x00")
        model2 = tmp_path / "m2.gguf"
        model2.write_bytes(b"GGUF\x03\x00\x00\x00")

        cache_store = {
            "backend_mode_select": "llama.cpp (Local GGUF)",
            "local_model_path": str(model1),
        }
        agent1 = get_session_agent(
            model_name="m1.gguf",
            cache_store=cache_store,
        )
        # Mock close on backend of agent1
        agent1.backend.close = MagicMock()

        # Switch to model2
        cache_store["local_model_path"] = str(model2)
        agent2 = get_session_agent(
            model_name="m2.gguf",
            cache_store=cache_store,
        )
        assert agent2 is not agent1
        agent1.backend.close.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# 5. STREAMLIT APPTEST UI VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────


class TestStreamlitAppTestUIBackendSwitching:
    """Verifies backend selection UI controls, status badges, and telemetry via AppTest."""

    def test_initial_render_defaults_to_cloud_gemini(self):
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        assert not at.exception

        backend_select = next(
            s for s in at.sidebar.selectbox if "Backend Selection" in s.label
        )
        assert backend_select.value == "Google Gemini (Cloud)"

        # Text input at index 0 must be Gemini API Key
        text_inputs = list(at.sidebar.text_input)
        assert len(text_inputs) >= 6
        assert "Gemini API Key" in text_inputs[0].label

        # Telemetry strip reflects Cloud Gemini
        chat_markdowns = [m.value for m in at.markdown]
        assert any("Inference: <strong>Cloud (Gemini" in val for val in chat_markdowns)

    def test_switch_to_local_backend_renders_model_path_input(self):
        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        assert not at.exception

        backend_select = next(
            s for s in at.sidebar.selectbox if "Backend Selection" in s.label
        )
        backend_select.select("llama.cpp (Local GGUF)")
        at.run()
        assert not at.exception

        # In local mode, text input at index 0 is Local GGUF Model Path
        text_inputs = list(at.sidebar.text_input)
        assert "Local GGUF Model Path" in text_inputs[0].label

        # Verify chat telemetry strip reflects local inference
        chat_markdowns = [m.value for m in at.markdown]
        assert any(
            "Inference: <strong>Local (llama.cpp" in val for val in chat_markdowns
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6. SQLGUARD ENFORCEMENT ON ACTUAL EXECUTION PATH
# ──────────────────────────────────────────────────────────────────────────────


class TestSQLGuardActualExecutionPath:
    """Verifies that generated SQL from any backend is strictly enforced via SQLGuard."""

    @pytest.fixture
    def test_sqlite_engine(self):
        from sqlalchemy import create_engine, text

        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(
                text("CREATE TABLE customers (id INT PRIMARY KEY, name TEXT);")
            )
            conn.execute(text("INSERT INTO customers VALUES (1, 'Alice'), (2, 'Bob');"))
            conn.commit()
        return engine

    def test_safe_select_executes_successfully(self, test_sqlite_engine):
        safe_sql = "SELECT id, name FROM customers;"
        df, is_truncated = execute_safe_query(
            engine=test_sqlite_engine,
            query=safe_sql,
            allowed_tables={"customers"},
            dialect="sqlite",
        )
        assert len(df) == 2
        assert is_truncated is False
        assert list(df["name"]) == ["Alice", "Bob"]

    def test_malicious_drop_query_blocked_by_sql_guard(self, test_sqlite_engine):
        malicious_sql = "DROP TABLE customers;"
        with pytest.raises(
            SQLSecurityViolation, match="Only read-only SELECT queries are permitted"
        ):
            execute_safe_query(
                engine=test_sqlite_engine,
                query=malicious_sql,
                allowed_tables={"customers"},
                dialect="sqlite",
            )

    def test_unauthorized_table_access_blocked(self, test_sqlite_engine):
        from sqlalchemy import text

        with test_sqlite_engine.connect() as conn:
            conn.execute(text("CREATE TABLE private_salaries (id INT, salary INT);"))
            conn.commit()

        unauthorized_sql = "SELECT salary FROM private_salaries;"
        with pytest.raises(SQLSecurityViolation, match="Unauthorized table reference"):
            execute_safe_query(
                engine=test_sqlite_engine,
                query=unauthorized_sql,
                allowed_tables={"customers"},  # private_salaries NOT allowed
                dialect="sqlite",
            )

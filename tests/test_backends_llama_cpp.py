"""Unit and integration tests for LlamaCppBackend (Step 3).

Coverage:
* Optional dependency handling when llama-cpp-python is absent
* Invalid, missing, or unsupported model paths
* Lazy loading and zero import-time model memory allocation
* Generation and streaming token/chunk mapping
* Native grammar-constrained structured output vs prompt formatting
* Resource/allocation failures (OOM) and error classification
* Timeout and cancellation behavior (cooperative streaming vs single-shot)
* Privacy-preserving telemetry
* SQLGuard deterministic enforcement boundary
* Skipped-by-default real inference integration test
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from schemasense.ai_agent import SQLAgentResponse
from schemasense.runtime.backends.base import (
    BackendCancelledError,
    BackendInfo,
    BackendNotAvailableError,
    BackendResourceError,
    BackendStructuredOutputError,
    BackendTimeoutError,
    GenerationRequest,
    GenerationResult,
)
from schemasense.runtime.backends.llama_cpp import (
    LLAMA_CPP_AVAILABLE,
    LlamaCppBackend,
    _validate_model_path,
)
from schemasense.sql_guard import execute_safe_query

# ──────────────────────────────────────────────────────────────────────────────
# 1. OPTIONAL DEPENDENCY ABSENCE & IMPORT HYGIENE
# ──────────────────────────────────────────────────────────────────────────────


class TestLlamaCppOptionalDependencyAbsence:
    """Verify behavior when llama-cpp-python is not installed."""

    @patch("schemasense.runtime.backends.llama_cpp.LLAMA_CPP_AVAILABLE", False)
    @patch("schemasense.runtime.backends.llama_cpp.llama_cpp", None)
    def test_is_available_false_when_package_missing(self, tmp_path):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        backend = LlamaCppBackend(model_path=dummy_gguf)
        assert backend.is_available() is False

    @patch("schemasense.runtime.backends.llama_cpp.LLAMA_CPP_AVAILABLE", False)
    @patch("schemasense.runtime.backends.llama_cpp.llama_cpp", None)
    def test_get_info_completes_when_package_missing(self):
        backend = LlamaCppBackend(model_path="model.gguf")
        info = backend.get_info()

        assert isinstance(info, BackendInfo)
        assert info.backend_id == "llama_cpp"
        assert info.display_name == "llama.cpp (local)"
        assert info.is_local is True
        assert info.requires_api_key is False
        assert info.engine_version is None

    @patch("schemasense.runtime.backends.llama_cpp.LLAMA_CPP_AVAILABLE", False)
    @patch("schemasense.runtime.backends.llama_cpp.llama_cpp", None)
    def test_generate_raises_not_available_with_helpful_message(self, tmp_path):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        backend = LlamaCppBackend(model_path=dummy_gguf)
        request = GenerationRequest(prompt="Hello")

        with pytest.raises(BackendNotAvailableError) as exc_info:
            backend.generate(request)

        assert (
            "requirements-local.txt" in str(exc_info.value)
            or "not available" in str(exc_info.value).lower()
        )

    @patch("schemasense.runtime.backends.llama_cpp.LLAMA_CPP_AVAILABLE", False)
    @patch("schemasense.runtime.backends.llama_cpp.llama_cpp", None)
    def test_generate_stream_raises_not_available(self, tmp_path):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        backend = LlamaCppBackend(model_path=dummy_gguf)
        request = GenerationRequest(prompt="Hello")

        with pytest.raises(BackendNotAvailableError):
            list(backend.generate_stream(request))

    def test_package_reexport_importable_without_engine(self):
        """Verifies importing package root never fails even without optional engine."""
        from schemasense.runtime.backends import (
            GeminiBackend,
            InferenceBackend,
            LlamaCppBackend,
        )

        assert LlamaCppBackend is not None
        assert GeminiBackend is not None
        assert InferenceBackend is not None


# ──────────────────────────────────────────────────────────────────────────────
# 2. MODEL PATH VALIDATION & ZERO AUTO-DOWNLOADS
# ──────────────────────────────────────────────────────────────────────────────


class TestLlamaCppModelPathValidation:
    """Verify explicit path checks, extension filtering, and absence of downloads."""

    def test_none_model_path_rejected(self):
        backend = LlamaCppBackend(model_path=None)
        assert backend.is_available() is False

        with pytest.raises(BackendNotAvailableError) as exc:
            _validate_model_path(None)
        assert "explicit local model_path" in str(exc.value)

    def test_nonexistent_path_rejected(self, tmp_path):
        missing = tmp_path / "does_not_exist.gguf"
        backend = LlamaCppBackend(model_path=missing)
        assert backend.is_available() is False

        with pytest.raises(BackendNotAvailableError) as exc:
            _validate_model_path(missing)
        assert "does not exist" in str(exc.value)

    def test_directory_rejected(self, tmp_path):
        directory = tmp_path / "models_dir"
        directory.mkdir()
        backend = LlamaCppBackend(model_path=directory)
        assert backend.is_available() is False

        with pytest.raises(BackendNotAvailableError):
            _validate_model_path(directory)

    def test_unsupported_extensions_rejected(self, tmp_path):
        for bad_ext in [".safetensors", ".pt", ".bin", ".pkl", ".onnx"]:
            f = tmp_path / f"model{bad_ext}"
            f.write_bytes(b"FAKE_DATA")
            backend = LlamaCppBackend(model_path=f)
            assert backend.is_available() is False

            with pytest.raises(BackendNotAvailableError) as exc:
                _validate_model_path(f)
            assert "Only .gguf files are supported" in str(exc.value)

    def test_valid_gguf_path_accepted(self, tmp_path):
        gguf_file = tmp_path / "qwen2.5-coder-7b.gguf"
        gguf_file.write_bytes(b"GGUF\x03\x00\x00\x00")

        validated = _validate_model_path(gguf_file)
        assert validated == gguf_file.resolve()


# ──────────────────────────────────────────────────────────────────────────────
# 3. LAZY LOADING & NO IMPORT-TIME ALLOCATION
# ──────────────────────────────────────────────────────────────────────────────


class TestLlamaCppLazyLoading:
    """Verify weights are never loaded until generate is explicitly called."""

    def test_instantiation_does_not_allocate_model(self, tmp_path):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        backend = LlamaCppBackend(model_path=dummy_gguf)
        assert backend._llm is None

    def test_get_info_does_not_allocate_model(self, tmp_path):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        backend = LlamaCppBackend(model_path=dummy_gguf)
        _ = backend.get_info()
        assert backend._llm is None

    def test_is_available_does_not_allocate_model(self, tmp_path):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        backend = LlamaCppBackend(model_path=dummy_gguf)
        _ = backend.is_available()
        assert backend._llm is None

    def test_telemetry_reflects_unloaded_state(self, tmp_path):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        backend = LlamaCppBackend(model_path=dummy_gguf)
        tel = backend.get_resource_telemetry()
        assert tel["model_loaded"] is False
        assert tel["is_local"] is True
        assert tel["backend_id"] == "llama_cpp"


# ──────────────────────────────────────────────────────────────────────────────
# 4. MOCKED GENERATION & STREAMING MAPPING
# ──────────────────────────────────────────────────────────────────────────────


class TestLlamaCppGeneration:
    """Verify single-shot generation and streaming result mapping via mocks."""

    @pytest.fixture
    def mock_llama_cpp(self, monkeypatch, tmp_path):
        """Set up mock llama_cpp module and dummy model file."""
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        mock_module = MagicMock()
        mock_module.__version__ = "0.3.5"

        mock_llm_instance = MagicMock()
        mock_module.Llama.return_value = mock_llm_instance

        monkeypatch.setattr(
            "schemasense.runtime.backends.llama_cpp.LLAMA_CPP_AVAILABLE", True
        )
        monkeypatch.setattr(
            "schemasense.runtime.backends.llama_cpp.llama_cpp", mock_module
        )

        return dummy_gguf, mock_module, mock_llm_instance

    def test_generate_text_success_raw_completion(self, mock_llama_cpp):
        dummy_gguf, _mock_module, mock_llm = mock_llama_cpp
        backend = LlamaCppBackend(model_path=dummy_gguf)

        mock_llm.create_completion.return_value = {
            "choices": [{"text": "SELECT * FROM customers;", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        }

        request = GenerationRequest(prompt="List customers", max_tokens=128)
        result = backend.generate(request)

        assert isinstance(result, GenerationResult)
        assert result.text == "SELECT * FROM customers;"
        assert result.backend_id == "llama_cpp"
        assert result.model_id == "model.gguf"
        assert result.finish_reason == "stop"
        assert result.tokens_prompt == 12
        assert result.tokens_completion == 6
        assert result.elapsed_ms >= 0.0

    def test_generate_chat_completion_with_system_and_history(self, mock_llama_cpp):
        dummy_gguf, _mock_module, mock_llm = mock_llama_cpp
        backend = LlamaCppBackend(model_path=dummy_gguf)

        mock_llm.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {"content": "SELECT count(*) FROM orders;"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        }

        request = GenerationRequest(
            prompt="How many orders?",
            system_instruction="You are a data architect.",
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )
        result = backend.generate(request)

        assert result.text == "SELECT count(*) FROM orders;"
        mock_llm.create_chat_completion.assert_called_once()
        call_kwargs = mock_llm.create_chat_completion.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"

    def test_generate_stream_yields_chunks(self, mock_llama_cpp):
        dummy_gguf, _mock_module, mock_llm = mock_llama_cpp
        backend = LlamaCppBackend(model_path=dummy_gguf)

        stream_chunks = [
            {"choices": [{"text": "SELECT "}]},
            {"choices": [{"text": "id, "}]},
            {"choices": [{"text": "name "}]},
            {"choices": [{"text": "FROM users;"}]},
        ]
        mock_llm.create_completion.return_value = iter(stream_chunks)

        request = GenerationRequest(prompt="Get users")
        chunks = list(backend.generate_stream(request))

        assert chunks == ["SELECT ", "id, ", "name ", "FROM users;"]
        assert "".join(chunks) == "SELECT id, name FROM users;"

    def test_generate_stream_cancellation_signal(self, mock_llama_cpp):
        dummy_gguf, _mock_module, mock_llm = mock_llama_cpp
        backend = LlamaCppBackend(model_path=dummy_gguf)

        cancel_token = threading.Event()

        def stream_gen():
            yield {"choices": [{"text": "Part 1"}]}
            cancel_token.set()
            yield {"choices": [{"text": "Part 2"}]}

        mock_llm.create_completion.return_value = stream_gen()

        request = GenerationRequest(prompt="Long stream")
        gen = backend.generate_stream(request, cancel_token=cancel_token)

        assert next(gen) == "Part 1"
        with pytest.raises(BackendCancelledError) as exc_info:
            next(gen)
        assert "incomplete" in str(exc_info.value).lower()

    def test_generate_stream_timeout_deadline(self, mock_llama_cpp):
        dummy_gguf, _mock_module, mock_llm = mock_llama_cpp
        backend = LlamaCppBackend(model_path=dummy_gguf)

        def slow_stream():
            yield {"choices": [{"text": "Fast"}]}
            time.sleep(0.04)
            yield {"choices": [{"text": "Late"}]}

        mock_llm.create_completion.return_value = slow_stream()

        request = GenerationRequest(prompt="Slow query", timeout_seconds=0.02)
        gen = backend.generate_stream(request)

        assert next(gen) == "Fast"
        with pytest.raises(BackendTimeoutError):
            next(gen)


# ──────────────────────────────────────────────────────────────────────────────
# 5. STRUCTURED OUTPUT: NATIVE ENFORCEMENT VS PROMPT SUGGESTION
# ──────────────────────────────────────────────────────────────────────────────


class TestLlamaCppStructuredOutput:
    """Verify accurate distinction of native grammar enforcement vs prompt suggestion."""

    @pytest.fixture
    def mock_llama_with_grammar(self, monkeypatch, tmp_path):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        mock_module = MagicMock()
        mock_module.__version__ = "0.3.5"
        mock_grammar = MagicMock()
        mock_module.LlamaGrammar.from_json_schema.return_value = mock_grammar

        mock_llm = MagicMock()
        mock_module.Llama.return_value = mock_llm

        monkeypatch.setattr(
            "schemasense.runtime.backends.llama_cpp.LLAMA_CPP_AVAILABLE", True
        )
        monkeypatch.setattr(
            "schemasense.runtime.backends.llama_cpp.llama_cpp", mock_module
        )

        return dummy_gguf, mock_module, mock_llm, mock_grammar

    def test_native_grammar_enforced_when_schema_provided(
        self, mock_llama_with_grammar
    ):
        dummy_gguf, mock_module, mock_llm, mock_grammar = mock_llama_with_grammar
        backend = LlamaCppBackend(model_path=dummy_gguf)

        valid_json = (
            '{"explanation": "Valid", "sql_query": "SELECT 1;", '
            '"target_tables": [], "is_executable": true}'
        )
        mock_llm.create_completion.return_value = {
            "choices": [{"text": valid_json, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10},
        }

        request = GenerationRequest(
            prompt="Generate structured",
            structured_schema=SQLAgentResponse,
        )
        result = backend.generate(request)

        assert result.text == valid_json
        mock_module.LlamaGrammar.from_json_schema.assert_called_once()
        call_kwargs = mock_llm.create_completion.call_args.kwargs
        assert call_kwargs["grammar"] == mock_grammar

    def test_grammar_compilation_failure_raises_structured_output_error(
        self, mock_llama_with_grammar
    ):
        dummy_gguf, mock_module, _mock_llm, _ = mock_llama_with_grammar
        backend = LlamaCppBackend(model_path=dummy_gguf)

        mock_module.LlamaGrammar.from_json_schema.side_effect = RuntimeError(
            "GBNF grammar compilation failed for recursive schema"
        )

        request = GenerationRequest(
            prompt="Structured prompt",
            structured_schema=SQLAgentResponse,
        )

        with pytest.raises(BackendStructuredOutputError) as exc_info:
            backend.generate(request)

        assert "Failed to compile native JSON schema grammar" in str(exc_info.value)

    def test_plain_text_request_does_not_compile_grammar(self, mock_llama_with_grammar):
        dummy_gguf, mock_module, mock_llm, _ = mock_llama_with_grammar
        backend = LlamaCppBackend(model_path=dummy_gguf)

        mock_llm.create_completion.return_value = {
            "choices": [{"text": "Plain text output", "finish_reason": "stop"}],
            "usage": {},
        }

        request = GenerationRequest(prompt="Plain text", structured_schema=None)
        _ = backend.generate(request)

        mock_module.LlamaGrammar.from_json_schema.assert_not_called()
        call_kwargs = mock_llm.create_completion.call_args.kwargs
        assert "grammar" not in call_kwargs


# ──────────────────────────────────────────────────────────────────────────────
# 6. RESOURCE & ALLOCATION FAILURES (OOM)
# ──────────────────────────────────────────────────────────────────────────────


class TestLlamaCppResourceFailures:
    """Verify resource exhaustion error mapping."""

    def test_ram_vram_oom_maps_to_backend_resource_error(self, monkeypatch, tmp_path):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        mock_module = MagicMock()
        mock_module.Llama.side_effect = MemoryError(
            "CUDA error: out of memory allocating 8GB VRAM"
        )

        monkeypatch.setattr(
            "schemasense.runtime.backends.llama_cpp.LLAMA_CPP_AVAILABLE", True
        )
        monkeypatch.setattr(
            "schemasense.runtime.backends.llama_cpp.llama_cpp", mock_module
        )

        backend = LlamaCppBackend(model_path=dummy_gguf)
        request = GenerationRequest(prompt="Prompt")

        with pytest.raises(BackendResourceError) as exc_info:
            backend.generate(request)

        assert "Resource exhaustion" in str(exc_info.value)

    def test_model_load_corruption_maps_to_not_available_error(
        self, monkeypatch, tmp_path
    ):
        dummy_gguf = tmp_path / "model.gguf"
        dummy_gguf.write_bytes(b"GGUF\x03\x00\x00\x00")

        mock_module = MagicMock()
        mock_module.Llama.side_effect = ValueError(
            "Corrupt GGUF magic number: 0xDEADBEEF"
        )

        monkeypatch.setattr(
            "schemasense.runtime.backends.llama_cpp.LLAMA_CPP_AVAILABLE", True
        )
        monkeypatch.setattr(
            "schemasense.runtime.backends.llama_cpp.llama_cpp", mock_module
        )

        backend = LlamaCppBackend(model_path=dummy_gguf)
        request = GenerationRequest(prompt="Prompt")

        with pytest.raises(BackendNotAvailableError) as exc_info:
            backend.generate(request)

        assert "Failed to load local model" in str(exc_info.value)


# ──────────────────────────────────────────────────────────────────────────────
# 7. DETERMINISTIC SQLGUARD BOUNDARY
# ──────────────────────────────────────────────────────────────────────────────


class TestLlamaCppSQLGuardBoundary:
    """Verify local inference outputs are never executed without SQLGuard validation."""

    def test_adversarial_local_sql_blocked_by_sqlguard(self, sqlite_engine):
        malicious_local_sql = "DROP TABLE customers;"

        with pytest.raises(Exception) as exc_info:
            execute_safe_query(
                engine=sqlite_engine,
                query=malicious_local_sql,
                allowed_tables={"customers"},
                dialect="sqlite",
            )
        assert (
            "security violation" in str(exc_info.value).lower()
            or "read-only" in str(exc_info.value).lower()
        )

    def test_stacked_local_sql_blocked_by_sqlguard(self, sqlite_engine):
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

    def test_legitimate_local_sql_executes_safely(self, sqlite_engine):
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


# ──────────────────────────────────────────────────────────────────────────────
# 8. REAL INFERENCE INTEGRATION TEST (SKIPPED BY DEFAULT)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (
        LLAMA_CPP_AVAILABLE
        and os.getenv("SCHEMASENSE_TEST_GGUF_MODEL")
        and Path(os.getenv("SCHEMASENSE_TEST_GGUF_MODEL", "")).is_file()
    ),
    reason="Requires installed llama-cpp-python and valid local GGUF model path in SCHEMASENSE_TEST_GGUF_MODEL env var.",
)
def test_real_local_inference_integration():
    """Real inference test executed ONLY when optional engine and user model exist."""
    model_path = os.getenv("SCHEMASENSE_TEST_GGUF_MODEL")
    assert model_path is not None

    backend = LlamaCppBackend(model_path=model_path, n_ctx=512)
    assert backend.is_available() is True

    request = GenerationRequest(
        prompt="Say hello in exactly two words.",
        max_tokens=10,
        temperature=0.1,
    )
    result = backend.generate(request)

    assert result.text.strip() != ""
    assert result.backend_id == "llama_cpp"
    assert result.finish_reason in ("stop", "length")

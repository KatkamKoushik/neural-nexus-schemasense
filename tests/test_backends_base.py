"""Tests for the backend-neutral inference contract (Step 1).

Coverage
--------
* :class:`BackendCapability` enum completeness
* :class:`BackendInfo` construction and immutability
* :class:`GenerationRequest` construction, defaults, and field semantics
* :class:`GenerationResult` construction and defaults
* Exception hierarchy (base / subclass relationships)
* :class:`FakeBackend` — protocol conformance and all specified behaviours:
  - ``get_info`` returns ``BackendInfo`` without I/O
  - ``is_available`` reflects the constructed state
  - ``generate`` returns ``GenerationResult`` with correct ``backend_id``
  - ``generate`` raises ``BackendTimeoutError`` when ``simulate_timeout=True``
  - ``generate`` raises ``BackendCancelledError`` when cancel token is pre-set
  - ``generate_stream`` yields chunks from ``stream_chunks``
  - ``generate_stream`` raises ``BackendCancelledError`` when cancel token is
    set mid-stream
  - ``get_resource_telemetry`` returns a dict and never raises

Design constraints honoured
---------------------------
* ``FakeBackend`` is defined in this test module only — it is not importable
  from production code.
* No Gemini SDK or llama_cpp import is required or performed.
* No model is loaded; no network call is made.
* SQLGuard is not involved — that enforcement belongs to
  ``execute_safe_query``, not to the backend contract.
* ``BackendTimeoutError`` tests only assert that the exception is raised;
  they do not assert that a worker thread has been terminated, consistent with
  the module-level timeout semantics.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from schemasense.runtime.backends import (
    BackendCancelledError,
    BackendCapability,
    BackendInfo,
    BackendNotAvailableError,
    BackendResourceError,
    BackendStructuredOutputError,
    BackendTimeoutError,
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
    InferenceBackendError,
)

# ──────────────────────────────────────────────────────────────────────────────
# FAKE BACKEND — test-only implementation
# ──────────────────────────────────────────────────────────────────────────────


class FakeBackend(InferenceBackend):
    """Minimal in-memory backend for protocol-conformance testing.

    ``FakeBackend`` must **never** be used to present output as real inference.
    It exists only to validate that the ``InferenceBackend`` contract is
    correctly defined and that callers can rely on its specified behaviours.

    Parameters
    ----------
    available:
        Return value of ``is_available()``.
    fixed_text:
        Text included in ``GenerationResult.text`` by ``generate()``.
    stream_chunks:
        Sequence of strings yielded by ``generate_stream()``.
    simulate_timeout:
        When ``True``, ``generate()`` raises ``BackendTimeoutError`` instead
        of returning a result.
    simulate_resource_error:
        When ``True``, ``generate()`` raises ``BackendResourceError``.
    extra_telemetry:
        Additional key-value pairs returned by ``get_resource_telemetry()``.
    """

    BACKEND_ID = "fake"
    MODEL_ID = "fake-model-v0"

    def __init__(
        self,
        *,
        available: bool = True,
        fixed_text: str = "fake output",
        stream_chunks: list[str] | None = None,
        simulate_timeout: bool = False,
        simulate_resource_error: bool = False,
        extra_telemetry: dict | None = None,
    ) -> None:
        self._available = available
        self._fixed_text = fixed_text
        self._stream_chunks: list[str] = (
            stream_chunks if stream_chunks is not None else ["chunk1", " chunk2"]
        )
        self._simulate_timeout = simulate_timeout
        self._simulate_resource_error = simulate_resource_error
        self._extra_telemetry: dict = extra_telemetry or {}

    # ------------------------------------------------------------------
    # InferenceBackend implementation
    # ------------------------------------------------------------------

    def get_info(self) -> BackendInfo:
        return BackendInfo(
            backend_id=self.BACKEND_ID,
            display_name="Fake Backend (test only)",
            is_local=True,
            capabilities=frozenset(
                {BackendCapability.STREAMING, BackendCapability.CANCELLATION}
            ),
            model_id=self.MODEL_ID,
            engine_version="0.0.0",
            requires_api_key=False,
        )

    def is_available(self) -> bool:
        return self._available

    def generate(
        self,
        request: GenerationRequest,
        cancel_token: threading.Event | None = None,
    ) -> GenerationResult:
        if not self._available:
            raise BackendNotAvailableError("FakeBackend is configured as unavailable.")

        # Cancellation check — consistent with contract: caller sets token
        if cancel_token is not None and cancel_token.is_set():
            raise BackendCancelledError("Generation cancelled before it started.")

        if self._simulate_timeout:
            raise BackendTimeoutError(
                f"Deadline of {request.timeout_seconds}s elapsed (simulated)."
            )

        if self._simulate_resource_error:
            raise BackendResourceError("Out of memory (simulated).")

        return GenerationResult(
            text=self._fixed_text,
            model_id=self.MODEL_ID,
            backend_id=self.BACKEND_ID,
            finish_reason="stop",
            elapsed_ms=1.0,
        )

    def generate_stream(
        self,
        request: GenerationRequest,
        cancel_token: threading.Event | None = None,
    ) -> Iterator[str]:
        if not self._available:
            raise BackendNotAvailableError("FakeBackend is configured as unavailable.")

        for chunk in self._stream_chunks:
            # Check cancellation before each chunk (contract requirement)
            if cancel_token is not None and cancel_token.is_set():
                raise BackendCancelledError("Streaming cancelled mid-stream.")
            yield chunk

    def get_resource_telemetry(self) -> dict:
        return {"backend": self.BACKEND_ID, **self._extra_telemetry}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_backend() -> FakeBackend:
    """Default available FakeBackend with no error simulation."""
    return FakeBackend()


@pytest.fixture
def minimal_request() -> GenerationRequest:
    """A minimal valid GenerationRequest."""
    return GenerationRequest(prompt="Hello, world!")


# ──────────────────────────────────────────────────────────────────────────────
# BACKEND CAPABILITY
# ──────────────────────────────────────────────────────────────────────────────


class TestBackendCapability:
    def test_all_expected_members_present(self) -> None:
        """The enum must declare all capabilities defined in the design."""
        expected = {
            "STREAMING",
            "STRUCTURED_OUTPUT",
            "CANCELLATION",
            "TOOL_CALLING",
            "MULTIMODAL",
        }
        actual = {member.name for member in BackendCapability}
        assert expected == actual

    def test_enum_values_are_strings(self) -> None:
        for cap in BackendCapability:
            assert isinstance(cap.value, str)

    def test_enum_values_are_lowercase(self) -> None:
        """Values should be lowercase for stable serialisation."""
        for cap in BackendCapability:
            assert cap.value == cap.value.lower(), f"{cap.name} value is not lowercase"


# ──────────────────────────────────────────────────────────────────────────────
# BACKEND INFO
# ──────────────────────────────────────────────────────────────────────────────


class TestBackendInfo:
    def test_minimal_construction(self) -> None:
        info = BackendInfo(
            backend_id="test",
            display_name="Test",
            is_local=True,
            capabilities=frozenset(),
        )
        assert info.backend_id == "test"
        assert info.display_name == "Test"
        assert info.is_local is True
        assert info.capabilities == frozenset()
        assert info.model_id is None
        assert info.engine_version is None
        assert info.requires_api_key is False

    def test_full_construction(self) -> None:
        info = BackendInfo(
            backend_id="llama_cpp",
            display_name="llama.cpp (local)",
            is_local=True,
            capabilities=frozenset(
                {BackendCapability.STREAMING, BackendCapability.CANCELLATION}
            ),
            model_id="/models/qwen.gguf",
            engine_version="b3300",
            requires_api_key=False,
        )
        assert BackendCapability.STREAMING in info.capabilities
        assert info.model_id == "/models/qwen.gguf"

    def test_frozen_dataclass_is_immutable(self) -> None:
        info = BackendInfo(
            backend_id="test",
            display_name="Test",
            is_local=True,
            capabilities=frozenset(),
        )
        with pytest.raises((AttributeError, TypeError)):
            info.backend_id = "mutated"  # type: ignore[misc]

    def test_remote_backend_declaration(self) -> None:
        """A remote backend marks is_local=False and requires_api_key=True."""
        info = BackendInfo(
            backend_id="gemini",
            display_name="Gemini (cloud)",
            is_local=False,
            capabilities=frozenset(
                {BackendCapability.STREAMING, BackendCapability.STRUCTURED_OUTPUT}
            ),
            requires_api_key=True,
        )
        assert info.is_local is False
        assert info.requires_api_key is True

    def test_capabilities_is_frozenset(self) -> None:
        info = BackendInfo(
            backend_id="test",
            display_name="Test",
            is_local=True,
            capabilities=frozenset({BackendCapability.STREAMING}),
        )
        assert isinstance(info.capabilities, frozenset)

    def test_is_local_is_not_a_security_guarantee(self) -> None:
        """is_local is a declaration, not a runtime enforcement mechanism.
        This test documents that the field is a bool and nothing more.
        """
        local_info = BackendInfo(
            backend_id="local",
            display_name="Local",
            is_local=True,
            capabilities=frozenset(),
        )
        assert isinstance(local_info.is_local, bool)


# ──────────────────────────────────────────────────────────────────────────────
# GENERATION REQUEST
# ──────────────────────────────────────────────────────────────────────────────


class TestGenerationRequest:
    def test_prompt_only_construction(self) -> None:
        req = GenerationRequest(prompt="test prompt")
        assert req.prompt == "test prompt"
        assert req.system_instruction == ""
        assert req.history == []
        assert req.max_tokens == 1024
        assert req.temperature == 0.2
        assert req.stop_sequences == []
        assert req.structured_schema is None
        assert req.timeout_seconds == 30.0

    def test_history_is_not_shared_between_instances(self) -> None:
        """Default mutable field must produce independent lists."""
        req_a = GenerationRequest(prompt="a")
        req_b = GenerationRequest(prompt="b")
        req_a.history.append({"role": "user", "content": "x"})
        assert req_b.history == [], "History lists must not be shared across instances"

    def test_stop_sequences_not_shared(self) -> None:
        req_a = GenerationRequest(prompt="a")
        req_b = GenerationRequest(prompt="b")
        req_a.stop_sequences.append("</s>")
        assert req_b.stop_sequences == []

    def test_structured_schema_accepts_pydantic_class(self) -> None:
        """structured_schema holds a type reference, not an instance."""
        from pydantic import BaseModel

        class MySchema(BaseModel):
            answer: str

        req = GenerationRequest(prompt="q", structured_schema=MySchema)
        assert req.structured_schema is MySchema

    def test_temperature_can_be_zero(self) -> None:
        req = GenerationRequest(prompt="x", temperature=0.0)
        assert req.temperature == 0.0

    def test_timeout_seconds_can_be_fractional(self) -> None:
        req = GenerationRequest(prompt="x", timeout_seconds=0.5)
        assert req.timeout_seconds == 0.5

    def test_history_role_convention(self) -> None:
        """History entries use 'user' and 'assistant' roles."""
        req = GenerationRequest(
            prompt="follow up",
            history=[
                {"role": "user", "content": "first message"},
                {"role": "assistant", "content": "first response"},
            ],
        )
        assert req.history[0]["role"] == "user"
        assert req.history[1]["role"] == "assistant"


# ──────────────────────────────────────────────────────────────────────────────
# GENERATION RESULT
# ──────────────────────────────────────────────────────────────────────────────


class TestGenerationResult:
    def test_minimal_construction(self) -> None:
        result = GenerationResult(text="hello", model_id="m", backend_id="b")
        assert result.text == "hello"
        assert result.model_id == "m"
        assert result.backend_id == "b"
        assert result.finish_reason == "stop"
        assert result.tokens_prompt is None
        assert result.tokens_completion is None
        assert result.elapsed_ms == 0.0

    def test_all_finish_reasons_are_valid_strings(self) -> None:
        """finish_reason is a plain string; test the four documented values."""
        for reason in ("stop", "length", "cancelled", "error"):
            result = GenerationResult(
                text="", model_id="m", backend_id="b", finish_reason=reason
            )
            assert result.finish_reason == reason

    def test_token_counts_can_be_set(self) -> None:
        result = GenerationResult(
            text="hi",
            model_id="m",
            backend_id="b",
            tokens_prompt=10,
            tokens_completion=5,
            elapsed_ms=123.45,
        )
        assert result.tokens_prompt == 10
        assert result.tokens_completion == 5
        assert result.elapsed_ms == pytest.approx(123.45)

    def test_empty_text_is_allowed(self) -> None:
        """A model may produce no output (e.g. max_tokens=0)."""
        result = GenerationResult(text="", model_id="m", backend_id="b")
        assert result.text == ""


# ──────────────────────────────────────────────────────────────────────────────
# EXCEPTION HIERARCHY
# ──────────────────────────────────────────────────────────────────────────────


class TestExceptionHierarchy:
    def test_all_exceptions_inherit_from_base(self) -> None:
        subclasses = [
            BackendNotAvailableError,
            BackendTimeoutError,
            BackendCancelledError,
            BackendResourceError,
            BackendStructuredOutputError,
        ]
        for exc_cls in subclasses:
            assert issubclass(exc_cls, InferenceBackendError), (
                f"{exc_cls.__name__} must inherit from InferenceBackendError"
            )

    def test_base_inherits_from_exception(self) -> None:
        assert issubclass(InferenceBackendError, Exception)

    def test_exceptions_are_distinct_types(self) -> None:
        """Each subclass is a separate type; they are not aliases of each other."""
        classes = [
            BackendNotAvailableError,
            BackendTimeoutError,
            BackendCancelledError,
            BackendResourceError,
            BackendStructuredOutputError,
        ]
        assert len(set(classes)) == len(classes)

    def test_backend_timeout_error_can_be_raised_and_caught(self) -> None:
        with pytest.raises(BackendTimeoutError):
            raise BackendTimeoutError("deadline elapsed")

    def test_backend_cancelled_error_can_be_raised_and_caught(self) -> None:
        with pytest.raises(BackendCancelledError):
            raise BackendCancelledError("cancelled")

    def test_backend_not_available_caught_as_base(self) -> None:
        with pytest.raises(InferenceBackendError):
            raise BackendNotAvailableError("not available")

    def test_timeout_does_not_imply_worker_terminated(self) -> None:
        """Document contract: BackendTimeoutError expresses a deadline; it makes
        no guarantee that the underlying worker has been stopped.
        """
        err = BackendTimeoutError("30s deadline elapsed")
        # The exception carries a message; it has no 'worker_terminated' attribute.
        assert "worker_terminated" not in dir(err)

    def test_cancelled_partial_output_convention(self) -> None:
        """Document contract: chunks yielded before BackendCancelledError are partial."""
        err = BackendCancelledError("Streaming cancelled mid-stream.")
        assert isinstance(err, InferenceBackendError)


# ──────────────────────────────────────────────────────────────────────────────
# FAKE BACKEND — protocol conformance
# ──────────────────────────────────────────────────────────────────────────────


class TestFakeBackendInfo:
    def test_get_info_returns_backend_info(self, fake_backend: FakeBackend) -> None:
        info = fake_backend.get_info()
        assert isinstance(info, BackendInfo)

    def test_get_info_does_not_require_availability(self) -> None:
        backend = FakeBackend(available=False)
        info = backend.get_info()
        assert info.backend_id == FakeBackend.BACKEND_ID

    def test_backend_id_is_stable(self, fake_backend: FakeBackend) -> None:
        assert fake_backend.get_info().backend_id == "fake"

    def test_is_local_declared_true(self, fake_backend: FakeBackend) -> None:
        assert fake_backend.get_info().is_local is True

    def test_streaming_capability_declared(self, fake_backend: FakeBackend) -> None:
        assert BackendCapability.STREAMING in fake_backend.get_info().capabilities

    def test_cancellation_capability_declared(self, fake_backend: FakeBackend) -> None:
        assert BackendCapability.CANCELLATION in fake_backend.get_info().capabilities

    def test_structured_output_not_declared(self, fake_backend: FakeBackend) -> None:
        """FakeBackend does not claim native structured-output enforcement."""
        assert (
            BackendCapability.STRUCTURED_OUTPUT
            not in fake_backend.get_info().capabilities
        )


class TestFakeBackendAvailability:
    def test_available_when_configured_true(self, fake_backend: FakeBackend) -> None:
        assert fake_backend.is_available() is True

    def test_not_available_when_configured_false(self) -> None:
        backend = FakeBackend(available=False)
        assert backend.is_available() is False

    def test_is_available_does_not_raise(self) -> None:
        backend = FakeBackend(available=False)
        # Must return a bool, never raise
        result = backend.is_available()
        assert isinstance(result, bool)


class TestFakeBackendGenerate:
    def test_generate_returns_generation_result(
        self, fake_backend: FakeBackend, minimal_request: GenerationRequest
    ) -> None:
        result = fake_backend.generate(minimal_request)
        assert isinstance(result, GenerationResult)

    def test_generate_result_backend_id_matches(
        self, fake_backend: FakeBackend, minimal_request: GenerationRequest
    ) -> None:
        result = fake_backend.generate(minimal_request)
        assert result.backend_id == fake_backend.get_info().backend_id

    def test_generate_result_model_id_matches(
        self, fake_backend: FakeBackend, minimal_request: GenerationRequest
    ) -> None:
        result = fake_backend.generate(minimal_request)
        assert result.model_id == fake_backend.get_info().model_id

    def test_generate_returns_configured_text(
        self, minimal_request: GenerationRequest
    ) -> None:
        backend = FakeBackend(fixed_text="expected output")
        result = backend.generate(minimal_request)
        assert result.text == "expected output"

    def test_generate_raises_when_unavailable(
        self, minimal_request: GenerationRequest
    ) -> None:
        backend = FakeBackend(available=False)
        with pytest.raises(BackendNotAvailableError):
            backend.generate(minimal_request)

    def test_generate_raises_timeout_error_when_simulated(
        self, minimal_request: GenerationRequest
    ) -> None:
        backend = FakeBackend(simulate_timeout=True)
        with pytest.raises(BackendTimeoutError):
            backend.generate(minimal_request)

    def test_generate_timeout_error_does_not_guarantee_worker_terminated(
        self, minimal_request: GenerationRequest
    ) -> None:
        """Per contract: BackendTimeoutError expresses deadline expiry only."""
        backend = FakeBackend(simulate_timeout=True)
        try:
            backend.generate(minimal_request)
        except BackendTimeoutError as exc:
            # The exception must not carry a 'worker_terminated' attribute
            assert not hasattr(exc, "worker_terminated")

    def test_generate_raises_resource_error_when_simulated(
        self, minimal_request: GenerationRequest
    ) -> None:
        backend = FakeBackend(simulate_resource_error=True)
        with pytest.raises(BackendResourceError):
            backend.generate(minimal_request)

    def test_generate_raises_cancelled_when_token_pre_set(
        self, fake_backend: FakeBackend, minimal_request: GenerationRequest
    ) -> None:
        """If the cancel token is already set before generate is called,
        BackendCancelledError must be raised.  This tests the single-shot
        cancellation path.
        """
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(BackendCancelledError):
            fake_backend.generate(minimal_request, cancel_token=cancel)

    def test_generate_without_cancel_token_succeeds(
        self, fake_backend: FakeBackend, minimal_request: GenerationRequest
    ) -> None:
        """cancel_token=None must not cause errors."""
        result = fake_backend.generate(minimal_request, cancel_token=None)
        assert isinstance(result, GenerationResult)

    def test_generate_does_not_mutate_request(self, fake_backend: FakeBackend) -> None:
        """The backend must not modify the caller's request object."""
        req = GenerationRequest(prompt="immutable?", max_tokens=50)
        fake_backend.generate(req)
        assert req.prompt == "immutable?"
        assert req.max_tokens == 50


class TestFakeBackendStream:
    def test_stream_yields_configured_chunks(
        self, fake_backend: FakeBackend, minimal_request: GenerationRequest
    ) -> None:
        chunks = list(fake_backend.generate_stream(minimal_request))
        assert chunks == ["chunk1", " chunk2"]

    def test_stream_concatenated_equals_full_text(
        self, minimal_request: GenerationRequest
    ) -> None:
        backend = FakeBackend(stream_chunks=["Hello", ", ", "world!"])
        result = "".join(backend.generate_stream(minimal_request))
        assert result == "Hello, world!"

    def test_stream_raises_when_unavailable(
        self, minimal_request: GenerationRequest
    ) -> None:
        backend = FakeBackend(available=False)
        with pytest.raises(BackendNotAvailableError):
            list(backend.generate_stream(minimal_request))

    def test_stream_raises_cancelled_when_token_set_mid_stream(
        self, minimal_request: GenerationRequest
    ) -> None:
        """The cancel token is set after the first chunk.  The generator must
        raise BackendCancelledError before yielding the second chunk.
        """
        cancel = threading.Event()
        backend = FakeBackend(stream_chunks=["first", "second", "third"])

        collected: list[str] = []
        with pytest.raises(BackendCancelledError):
            for chunk in backend.generate_stream(minimal_request, cancel_token=cancel):
                collected.append(chunk)
                cancel.set()  # Set after consuming first chunk

        # Only the first chunk should have been yielded before cancellation
        assert collected == ["first"]

    def test_stream_partial_output_before_cancellation_is_incomplete(
        self, minimal_request: GenerationRequest
    ) -> None:
        """Chunks yielded before BackendCancelledError are partial.
        This test documents the contract; callers must not treat partial
        output as a complete generation.
        """
        cancel = threading.Event()
        backend = FakeBackend(stream_chunks=["partial", " content"])

        partial: list[str] = []
        with pytest.raises(BackendCancelledError):
            for chunk in backend.generate_stream(minimal_request, cancel_token=cancel):
                partial.append(chunk)
                cancel.set()

        assert "".join(partial) == "partial"  # Only the first chunk

    def test_stream_without_cancel_token_yields_all_chunks(
        self, fake_backend: FakeBackend, minimal_request: GenerationRequest
    ) -> None:
        chunks = list(fake_backend.generate_stream(minimal_request, cancel_token=None))
        assert len(chunks) == 2

    def test_stream_pre_set_cancel_token_raises_immediately(
        self, fake_backend: FakeBackend, minimal_request: GenerationRequest
    ) -> None:
        cancel = threading.Event()
        cancel.set()  # Already set before streaming starts

        collected: list[str] = []
        with pytest.raises(BackendCancelledError):
            collected.extend(
                fake_backend.generate_stream(minimal_request, cancel_token=cancel)
            )

        assert collected == [], "No chunks should be yielded when token is pre-set"

    def test_stream_single_chunk_backend(
        self, minimal_request: GenerationRequest
    ) -> None:
        backend = FakeBackend(stream_chunks=["only chunk"])
        chunks = list(backend.generate_stream(minimal_request))
        assert chunks == ["only chunk"]


class TestFakeBackendTelemetry:
    def test_telemetry_returns_dict(self, fake_backend: FakeBackend) -> None:
        telemetry = fake_backend.get_resource_telemetry()
        assert isinstance(telemetry, dict)

    def test_telemetry_does_not_raise(self) -> None:
        backend = FakeBackend(available=False)
        telemetry = backend.get_resource_telemetry()
        assert isinstance(telemetry, dict)

    def test_telemetry_includes_backend_id(self, fake_backend: FakeBackend) -> None:
        telemetry = fake_backend.get_resource_telemetry()
        assert "backend" in telemetry
        assert telemetry["backend"] == FakeBackend.BACKEND_ID

    def test_telemetry_includes_extra_fields(self) -> None:
        backend = FakeBackend(extra_telemetry={"vram_mb": 4096, "tokens_per_sec": 12.5})
        telemetry = backend.get_resource_telemetry()
        assert telemetry["vram_mb"] == 4096
        assert telemetry["tokens_per_sec"] == pytest.approx(12.5)

    def test_default_telemetry_from_base_returns_empty_dict(self) -> None:
        """The base-class default implementation must return {} without raising."""

        # Instantiate a minimal concrete backend that uses the default telemetry
        class MinimalBackend(InferenceBackend):
            def get_info(self) -> BackendInfo:
                return BackendInfo("min", "Minimal", True, frozenset())

            def is_available(self) -> bool:
                return True

            def generate(self, request, cancel_token=None) -> GenerationResult:
                return GenerationResult("ok", "m", "min")

            def generate_stream(self, request, cancel_token=None):
                yield "ok"

        backend = MinimalBackend()
        assert backend.get_resource_telemetry() == {}


# ──────────────────────────────────────────────────────────────────────────────
# PROTOCOL SURFACE — ensure InferenceBackend cannot be instantiated directly
# ──────────────────────────────────────────────────────────────────────────────


class TestInferenceBackendIsAbstract:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            InferenceBackend()  # type: ignore[abstract]

    def test_partial_implementation_raises_on_instantiation(self) -> None:
        """A class that omits generate must not be instantiable."""

        class Partial(InferenceBackend):
            def get_info(self):
                return BackendInfo("p", "P", True, frozenset())

            def is_available(self):
                return True

            def generate(self, request, cancel_token=None):
                return GenerationResult("", "m", "p")

            # generate_stream deliberately omitted

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


# ──────────────────────────────────────────────────────────────────────────────
# SQLGUARD NON-INVOLVEMENT
# ──────────────────────────────────────────────────────────────────────────────


class TestSQLGuardNonInvolvement:
    """These tests document that the backend contract does NOT enforce SQLGuard.

    SQLGuard enforcement belongs exclusively to ``execute_safe_query``.
    The backend contract handles generation; execution safety is a separate
    concern in the call stack above the backend.
    """

    def test_backend_result_containing_sql_is_not_validated_here(
        self, fake_backend: FakeBackend
    ) -> None:
        """A result containing SQL is returned as-is; no guard is applied."""
        backend = FakeBackend(fixed_text="SELECT * FROM users; DROP TABLE users;")
        req = GenerationRequest(prompt="generate sql")
        result = backend.generate(req)
        # The contract returns raw text — the caller is responsible for
        # passing any SQL through SQLGuard before execution.
        assert "DROP TABLE" in result.text

    def test_backend_imports_do_not_include_sql_guard(self) -> None:
        """Verify sql_guard is not imported by the backends package."""
        import schemasense.runtime.backends as backends_pkg
        import schemasense.runtime.backends.base as base_mod

        assert not hasattr(backends_pkg, "SQLGuard")
        assert not hasattr(base_mod, "SQLGuard")

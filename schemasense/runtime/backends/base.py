"""SchemaSense Inference Backend Contract.

This module defines the backend-neutral interface that all inference adapters
must implement, together with the request/result data types and the typed
exception hierarchy.

Design guarantees
-----------------
* No imports from ``google.genai``, ``llama_cpp``, or any inference engine.
* No model loading or network I/O is performed in this module.
* ``SQLGuard`` is **not** enforced here. Every SQL statement that will be
  executed against a real database must still pass through
  ``schemasense.sql_guard.execute_safe_query``.  The contract does not modify
  or bypass that requirement.

Timeout semantics
-----------------
A ``BackendTimeoutError`` means the **caller's wall-clock deadline has
expired** and the backend was instructed to stop producing output.  It does
*not* guarantee that the underlying worker thread or model-generation process
has been terminated by the time the exception is raised — backends that cannot
cancel in-flight work are required to document that limitation in their class
docstring.  Callers must not assume the worker is dead after catching this
exception.

Cancellation semantics
----------------------
* **Single-shot** (``generate``): pass a :class:`threading.Event` as
  ``cancel_token``.  The backend checks the token before starting and at
  internal checkpoints.  It raises :class:`BackendCancelledError` when
  detected.  The caller is responsible for setting the event; the backend
  never sets it.
* **Streaming** (``generate_stream``): the same token is checked before each
  yielded chunk.  The generator raises :class:`BackendCancelledError` and
  terminates.  Partial output already yielded must be treated as incomplete
  by the caller.

Structured output
-----------------
``GenerationRequest.structured_schema`` carries a ``type`` reference (e.g. a
Pydantic model class) that the backend is *asked* to enforce.  Two cases are
distinguished:

* **Native enforcement**: the backend uses grammar-constrained decoding,
  JSON-schema mode, or a provider-side ``response_schema`` parameter so that
  the raw model output is guaranteed to be valid JSON conforming to the schema
  before text is returned.  Backends that do this MUST set
  ``BackendCapability.STRUCTURED_OUTPUT`` in their ``BackendInfo.capabilities``
  and MUST raise :class:`BackendStructuredOutputError` when the schema cannot
  be honoured.
* **Prompt-based suggestion**: the backend embeds schema instructions in the
  system prompt but does *not* guarantee the output is schema-conformant.
  Backends that do only this MUST NOT claim ``STRUCTURED_OUTPUT`` capability.

Local versus remote
-------------------
``BackendInfo.is_local`` is a **declaration of intent**: the backend author
asserts that no data leaves the local process during inference.  It is not a
runtime security guarantee.  Callers and operators must establish trust through
additional means (network firewall rules, process isolation, audit logging)
appropriate to their deployment context.  A ``is_local=True`` backend that
experiences an import-time network call due to a dependency should be treated
as a defect, not a policy enforcement failure of this contract.

Cloud fallback policy
---------------------
Fallback from a failing backend to a cloud backend is the responsibility of
the *caller* (e.g. ``SchemaSenseAIAgent``), not of individual backend
implementations.  A backend MUST NOT silently forward a request to a different
backend or network endpoint.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


# ──────────────────────────────────────────────────────────────────────────────
# CAPABILITY FLAGS
# ──────────────────────────────────────────────────────────────────────────────


class BackendCapability(Enum):
    """Feature flags that a backend may declare in :attr:`BackendInfo.capabilities`.

    A backend MUST NOT claim a capability it cannot reliably provide.
    """

    STREAMING = "streaming"
    """Backend supports incremental token streaming via ``generate_stream``."""

    STRUCTURED_OUTPUT = "structured_output"
    """Backend enforces a JSON schema at the model level (grammar / schema-mode),
    not merely via prompt instructions.  Only set this flag if conformance is
    mechanically enforced before output is returned to the caller.
    """

    CANCELLATION = "cancellation"
    """Backend checks the ``cancel_token`` event at meaningful intervals and
    raises :class:`BackendCancelledError` promptly.  Backends that cannot
    cancel in-flight inference must not set this flag.
    """

    TOOL_CALLING = "tool_calling"
    """Backend supports native function/tool-calling protocols."""

    MULTIMODAL = "multimodal"
    """Backend accepts non-text input (images, audio, etc.)."""


# ──────────────────────────────────────────────────────────────────────────────
# BACKEND METADATA
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BackendInfo:
    """Static, read-only description of a backend's identity and capabilities.

    Instances MUST be constructable without performing I/O (no model loading,
    no network calls).  :meth:`InferenceBackend.get_info` may be called at any
    time, including before :meth:`InferenceBackend.is_available` returns
    ``True``.
    """

    backend_id: str
    """Short, stable identifier, e.g. ``"llama_cpp"`` or ``"gemini"``.
    Used for logging and telemetry.  Must not change between versions.
    """

    display_name: str
    """Human-readable name shown in the UI, e.g. ``"llama.cpp (local)"``."""

    is_local: bool
    """``True`` if the author asserts that this backend performs all inference
    in-process with no outbound network calls.  See module docstring for the
    scope of this declaration.
    """

    capabilities: frozenset[BackendCapability]
    """Declared capabilities.  Use ``BackendCapability`` members."""

    model_id: str | None = None
    """Resolved model path or API model name, if known at construction time."""

    engine_version: str | None = None
    """Inference engine version string, e.g. ``"llama.cpp b3300"``."""

    requires_api_key: bool = False
    """``True`` if the backend needs an externally-supplied API key."""


# ──────────────────────────────────────────────────────────────────────────────
# REQUEST / RESULT TYPES
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class GenerationRequest:
    """Backend-neutral specification of a single generation turn.

    All fields have defaults so callers need only provide ``prompt``.
    Backends must honour ``timeout_seconds``; they must not silently ignore it.
    """

    prompt: str
    """The user-facing prompt for this turn."""

    system_instruction: str = ""
    """Static system/role instruction.  May be empty."""

    history: list[dict[str, str]] = field(default_factory=list)
    """Prior conversation turns as a list of ``{"role": str, "content": str}``
    dicts.  Roles are ``"user"`` and ``"assistant"``.
    """

    max_tokens: int = 1024
    """Maximum number of new tokens to generate."""

    temperature: float = 0.2
    """Sampling temperature.  0.0 = deterministic greedy decoding."""

    stop_sequences: list[str] = field(default_factory=list)
    """Token sequences that, when generated, terminate output immediately."""

    structured_schema: type | None = None
    """Optional Pydantic model class.  When provided, the caller requests that
    the backend enforce conformance.  See module docstring for the distinction
    between native enforcement and prompt-based suggestion.  ``None`` means
    plain-text output.
    """

    timeout_seconds: float = 30.0
    """Wall-clock deadline for the entire generation call.  Backends must
    raise :class:`BackendTimeoutError` when this deadline elapses.  See module
    docstring for what "timeout" guarantees and does not guarantee.
    """


@dataclass
class GenerationResult:
    """Backend-neutral result of a completed generation call.

    All fields except ``text``, ``model_id``, and ``backend_id`` are optional
    because not every backend can provide token counts or timing.
    """

    text: str
    """The complete generated text.  May be empty if the model produced no
    tokens before the finish condition (e.g. ``max_tokens=0``).
    """

    model_id: str
    """Identifies which model produced this output (path or API model name)."""

    backend_id: str
    """Matches :attr:`BackendInfo.backend_id` of the producing backend."""

    finish_reason: str = "stop"
    """Why generation ended: ``"stop"`` | ``"length"`` | ``"cancelled"`` |
    ``"error"``.  Backends should set this to ``"cancelled"`` when
    :class:`BackendCancelledError` would have been raised but the backend
    chose to return partial output instead.
    """

    tokens_prompt: int | None = None
    """Number of tokens in the prompt, if the backend can report this."""

    tokens_completion: int | None = None
    """Number of new tokens generated, if the backend can report this."""

    elapsed_ms: float = 0.0
    """Wall-clock time spent in the ``generate`` call, in milliseconds."""


# ──────────────────────────────────────────────────────────────────────────────
# EXCEPTION HIERARCHY
# ──────────────────────────────────────────────────────────────────────────────


class InferenceBackendError(Exception):
    """Root exception for all inference backend failures.

    Callers may catch this base class to handle any backend-level failure
    uniformly, or catch specific subclasses for targeted handling.
    """


class BackendNotAvailableError(InferenceBackendError):
    """The backend cannot accept requests.

    Raised when the inference engine is not installed, a required model file
    is missing or unreadable, or the backend fails its self-check.
    :meth:`InferenceBackend.is_available` returns ``False`` in all these
    cases; this exception is raised by ``generate`` / ``generate_stream``
    when called on an unavailable backend.
    """


class BackendTimeoutError(InferenceBackendError):
    """The caller's wall-clock deadline expired before generation completed.

    See module docstring for the precise semantics: raising this exception
    means the deadline elapsed and the backend was instructed to stop.  It
    does **not** guarantee that the underlying worker thread or model process
    has been terminated.
    """


class BackendCancelledError(InferenceBackendError):
    """Generation was stopped by an explicit caller-supplied cancellation signal.

    Raised when the ``cancel_token`` :class:`threading.Event` passed to
    ``generate`` or ``generate_stream`` is set before or during generation.
    For streaming generation, any chunks already yielded before the
    cancellation are partial and MUST be treated as incomplete by the caller.
    """


class BackendResourceError(InferenceBackendError):
    """A non-recoverable resource condition occurred during generation.

    Covers GPU/CPU out-of-memory, disk full, and similar conditions that
    prevent completion but are distinct from configuration errors.  Where
    possible, backends should include the :class:`SizingReport` advisory from
    ``ModelCompatibilityCalculator`` in the exception message.
    """


class BackendStructuredOutputError(InferenceBackendError):
    """The backend cannot enforce the requested structured-output schema.

    Only raised by backends that declare
    :attr:`BackendCapability.STRUCTURED_OUTPUT` and then encounter an
    irrecoverable schema-enforcement failure (e.g. grammar compilation error,
    schema too complex for the engine).  Backends that do *not* claim
    ``STRUCTURED_OUTPUT`` must not raise this exception.
    """


# ──────────────────────────────────────────────────────────────────────────────
# ABSTRACT BACKEND INTERFACE
# ──────────────────────────────────────────────────────────────────────────────


class InferenceBackend(ABC):
    """Abstract base class for all SchemaSense inference backends.

    Implementors must provide concrete ``get_info``, ``is_available``,
    ``generate``, and ``generate_stream``.  The default ``get_resource_telemetry``
    returns an empty dict; backends with richer instrumentation may override it.

    Thread safety
    ~~~~~~~~~~~~~
    Concurrent calls to ``generate`` or ``generate_stream`` from multiple
    threads are **not** guaranteed safe unless the concrete backend documents
    otherwise.  Callers are responsible for serialising access when needed.

    No model loading in this class
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This abstract class performs no I/O, loads no weights, and makes no
    network calls.  Concrete implementations may perform lazy initialisation
    inside ``generate`` / ``generate_stream``, but should document the
    latency implications.
    """

    # ------------------------------------------------------------------
    # Identity & availability
    # ------------------------------------------------------------------

    @abstractmethod
    def get_info(self) -> BackendInfo:
        """Return static backend identity and capability declaration.

        MUST complete without I/O.  MUST NOT raise.  May be called at any
        time, including before :meth:`is_available` returns ``True``.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the backend is ready to accept generation requests.

        MUST complete in less than 100 ms.  MUST NOT raise — return ``False``
        for any failure condition.  Implementations should cache any expensive
        availability check to meet the latency requirement.
        """

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
        cancel_token: threading.Event | None = None,
    ) -> GenerationResult:
        """Blocking single-shot generation.

        The call blocks until one of these conditions is met:

        * Generation finishes normally (``finish_reason="stop"`` or
          ``"length"``).
        * ``request.timeout_seconds`` elapses → raises
          :class:`BackendTimeoutError`.
        * ``cancel_token.is_set()`` is observed → raises
          :class:`BackendCancelledError`.
        * A resource condition occurs → raises :class:`BackendResourceError`.
        * The backend is not ready → raises :class:`BackendNotAvailableError`.

        Parameters
        ----------
        request:
            Fully-specified generation request.
        cancel_token:
            Optional :class:`threading.Event`.  The backend monitors this at
            internal checkpoints.  The caller is responsible for setting it;
            the backend must never set it.

        Returns
        -------
        GenerationResult
            Complete generated text and associated metadata.
        """

    @abstractmethod
    def generate_stream(
        self,
        request: GenerationRequest,
        cancel_token: threading.Event | None = None,
    ) -> Iterator[str]:
        """Streaming generation — yields text chunks as they are produced.

        Each yielded value is a non-empty string fragment.  The generator
        terminates normally when generation ends, or raises one of the
        following exceptions:

        * :class:`BackendTimeoutError` if ``request.timeout_seconds`` elapses.
        * :class:`BackendCancelledError` if ``cancel_token.is_set()`` is
          detected before a chunk.  Output yielded before this exception is
          partial and MUST be treated as incomplete by the caller.
        * :class:`BackendResourceError` on resource exhaustion.
        * :class:`BackendNotAvailableError` if the backend is not ready.

        Backends that do not support streaming MUST NOT declare
        :attr:`BackendCapability.STREAMING` and should raise
        :class:`BackendNotAvailableError` with an explanatory message.

        Parameters
        ----------
        request:
            Fully-specified generation request.
        cancel_token:
            Optional :class:`threading.Event`.  Checked before each yielded
            chunk.  The caller is responsible for setting it.
        """

    # ------------------------------------------------------------------
    # Telemetry (optional override)
    # ------------------------------------------------------------------

    def get_resource_telemetry(self) -> dict[str, Any]:
        """Return current backend resource usage as a plain dict.

        Keys and value types are backend-specific.  The default implementation
        returns an empty dict.  Backends with engine-level instrumentation
        (e.g. VRAM usage, token throughput) may override this method.

        MUST NOT raise.  MUST complete without blocking.
        """
        return {}

"""SchemaSense Local AI Runtime — llama.cpp inference backend.

Architecture and Guarantees
----------------------------
* Implements :class:`InferenceBackend` using the optional ``llama-cpp-python`` package.
* Optional dependency: If ``llama-cpp-python`` is not installed, importing this module
  never raises.  :meth:`is_available` safely returns ``False``.
* Local provenance: ``BackendInfo.is_local`` is ``True``; ``BackendInfo.requires_api_key``
  is ``False``.  No network requests are ever made.  No cloud fallback is performed.
* Lazy allocation: The model is **never** loaded at import time or backend instantiation.
  Weights are only mapped to memory upon the first invocation of :meth:`generate` or
  :meth:`generate_stream`.
* Model safety: Models are never downloaded automatically.  The backend accepts only
  explicit local ``.gguf`` file paths and verifies their existence before loading.
* SQLGuard safety: This backend does **not** validate or execute SQL.  All SQL produced
  must route through ``schemasense.sql_guard.execute_safe_query`` before execution.

Timeout & Cancellation Semantics
--------------------------------
* Streaming (:meth:`generate_stream`): Cooperative chunk-level cancellation is enforced
  by checking ``cancel_token`` before each yielded chunk.  Wall-clock timeouts raise
  :class:`BackendTimeoutError`.
* Single-shot (:meth:`generate`): ``cancel_token`` is checked before starting execution.
  Once in-flight in native C kernels, thread execution cannot be preemptively terminated
  from Python.  A caller-side deadline or :class:`BackendTimeoutError` indicates that
  the wall-clock allowance expired; any late output from abandoned execution is discarded.
* Structured Output: Supports native grammar-constrained decoding via
  ``llama_cpp.LlamaGrammar.from_json_schema``.  If grammar compilation fails,
  :class:`BackendStructuredOutputError` is raised.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from schemasense.runtime.backends.base import (
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

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger("schemasense.runtime.backends.llama_cpp")

# Optional dependency: safely detect llama-cpp-python
try:
    import llama_cpp

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    llama_cpp = None  # type: ignore[assignment]
    LLAMA_CPP_AVAILABLE = False


def _validate_model_path(path: str | Path | None) -> Path:
    """Validate that the given model path points to an existing local GGUF file.

    Raises:
        BackendNotAvailableError: If path is None, non-existent, not a file,
            or has an unsupported extension.
    """
    if path is None:
        raise BackendNotAvailableError(
            "LlamaCppBackend requires an explicit local model_path. None was provided."
        )

    p = Path(path).resolve()
    if not p.is_file():
        raise BackendNotAvailableError(
            f"Model file does not exist or is not a regular file: '{p}'"
        )

    if p.suffix.lower() != ".gguf":
        raise BackendNotAvailableError(
            f"Unsupported model container format '{p.suffix}'. Only .gguf files are supported."
        )

    return p


class LlamaCppBackend(InferenceBackend):
    """Local inference backend powered by llama-cpp-python and GGUF model files.

    Lazy-loads the model into memory only when inference is requested, supports
    both CPU and GPU offloading, and enforces bounded streaming and structured output.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        chat_format: str | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.chat_format = chat_format

        # Internal model handle — lazily loaded
        self._llm: Any = None
        self._load_lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────
    # Identity & Availability
    # ──────────────────────────────────────────────────────────────────

    def get_info(self) -> BackendInfo:
        """Return static metadata and capability declarations.

        Completes without I/O and never loads weights.
        """
        engine_ver = (
            getattr(llama_cpp, "__version__", "unknown")
            if LLAMA_CPP_AVAILABLE and llama_cpp is not None
            else None
        )
        model_id_str = str(self.model_path.name) if self.model_path else None

        return BackendInfo(
            backend_id="llama_cpp",
            display_name="llama.cpp (local)",
            is_local=True,
            capabilities=frozenset(
                {
                    BackendCapability.STREAMING,
                    BackendCapability.STRUCTURED_OUTPUT,
                }
            ),
            model_id=model_id_str,
            engine_version=f"llama-cpp-python {engine_ver}" if engine_ver else None,
            requires_api_key=False,
        )

    def is_available(self) -> bool:
        """Return True if llama-cpp-python is installed and the model file exists.

        Executes in under 100ms without loading weights into RAM or VRAM.
        """
        if not LLAMA_CPP_AVAILABLE or llama_cpp is None:
            return False

        if self.model_path is None:
            return False

        try:
            p = self.model_path.resolve()
            return p.is_file() and p.suffix.lower() == ".gguf"
        except Exception:  # noqa: BLE001
            return False

    # ──────────────────────────────────────────────────────────────────
    # Model Loading (Lazy)
    # ──────────────────────────────────────────────────────────────────

    def _get_or_load_llm(self) -> Any:
        """Load the GGUF model into memory if not already loaded. Thread-safe."""
        if self._llm is not None:
            return self._llm

        with self._load_lock:
            if self._llm is not None:
                return self._llm

            if not LLAMA_CPP_AVAILABLE or llama_cpp is None:
                raise BackendNotAvailableError(
                    "The llama-cpp-python package is not installed. To run local models, "
                    "install it via: pip install -r requirements-local.txt"
                )

            valid_path = _validate_model_path(self.model_path)

            try:
                logger.info(
                    "Loading local GGUF model '%s' (n_ctx=%d, n_gpu_layers=%d)",
                    valid_path.name,
                    self.n_ctx,
                    self.n_gpu_layers,
                )
                init_kwargs: dict[str, Any] = {
                    "model_path": str(valid_path),
                    "n_ctx": self.n_ctx,
                    "n_gpu_layers": self.n_gpu_layers,
                    "verbose": self.verbose,
                }
                if self.n_threads is not None:
                    init_kwargs["n_threads"] = self.n_threads
                if self.chat_format is not None:
                    init_kwargs["chat_format"] = self.chat_format

                self._llm = llama_cpp.Llama(**init_kwargs)
                return self._llm

            except Exception as e:
                err_msg = str(e).lower()
                if (
                    "out of memory" in err_msg
                    or "cuda error" in err_msg
                    or "cannot allocate" in err_msg
                ):
                    raise BackendResourceError(
                        f"Resource exhaustion (RAM/VRAM) loading '{valid_path.name}': {e}"
                    ) from e
                raise BackendNotAvailableError(
                    f"Failed to load local model '{valid_path.name}': {e}"
                ) from e

    # ──────────────────────────────────────────────────────────────────
    # Grammar Compilation (Structured Output)
    # ──────────────────────────────────────────────────────────────────

    def _compile_grammar(self, schema: type | None) -> Any | None:
        """Compile a Pydantic schema class into a LlamaGrammar for native enforcement."""
        if schema is None:
            return None

        if not LLAMA_CPP_AVAILABLE or llama_cpp is None:
            raise BackendNotAvailableError(
                "llama-cpp-python is required to enforce structured output schemas."
            )

        if not hasattr(llama_cpp, "LlamaGrammar"):
            raise BackendStructuredOutputError(
                "Installed llama-cpp-python version lacks LlamaGrammar support."
            )

        try:
            if hasattr(schema, "model_json_schema"):
                schema_json = json.dumps(schema.model_json_schema())
            elif hasattr(schema, "schema"):
                schema_json = json.dumps(schema.schema())
            else:
                raise ValueError(
                    f"Requested structured_schema '{schema}' lacks JSON schema serialization."
                )

            return llama_cpp.LlamaGrammar.from_json_schema(schema_json)
        except Exception as e:
            raise BackendStructuredOutputError(
                f"Failed to compile native JSON schema grammar for {schema}: {e}"
            ) from e

    # ──────────────────────────────────────────────────────────────────
    # Generation & Streaming
    # ──────────────────────────────────────────────────────────────────

    def _prepare_prompt_and_messages(
        self, request: GenerationRequest
    ) -> tuple[str, list[dict[str, str]] | None]:
        """Prepare raw prompt string and optional message structure."""
        if not request.system_instruction and not request.history:
            return request.prompt, None

        messages: list[dict[str, str]] = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})
        for msg in request.history:
            messages.append(
                {
                    "role": msg.get("role", "user"),
                    "content": str(msg.get("content", "")),
                }
            )
        messages.append({"role": "user", "content": request.prompt})

        # Fallback raw prompt if chat API is not used
        flat_parts: list[str] = []
        if request.system_instruction:
            flat_parts.append(f"System: {request.system_instruction}\n")
        for msg in request.history:
            flat_parts.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}\n")
        flat_parts.append(f"User: {request.prompt}\nAssistant:")
        raw_prompt = "\n".join(flat_parts)

        return raw_prompt, messages

    def generate(
        self,
        request: GenerationRequest,
        cancel_token: threading.Event | None = None,
    ) -> GenerationResult:
        """Blocking single-shot generation via llama.cpp."""
        if not self.is_available():
            raise BackendNotAvailableError(
                "LlamaCppBackend is not available: either llama-cpp-python is not installed, "
                "or the configured model_path is missing or invalid."
            )

        if cancel_token is not None and cancel_token.is_set():
            raise BackendCancelledError(
                "Generation cancelled by caller before inference started."
            )

        grammar = self._compile_grammar(request.structured_schema)
        llm = self._get_or_load_llm()

        raw_prompt, messages = self._prepare_prompt_and_messages(request)
        timeout_sec = request.timeout_seconds if request.timeout_seconds > 0 else 30.0

        start_time = time.monotonic()
        try:
            # Use chat completion if messages are present and supported
            if messages is not None and hasattr(llm, "create_chat_completion"):
                kwargs: dict[str, Any] = {
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "stream": False,
                }
                if request.stop_sequences:
                    kwargs["stop"] = request.stop_sequences
                if grammar is not None:
                    kwargs["grammar"] = grammar

                response = llm.create_chat_completion(**kwargs)
                choices = response.get("choices", [])
                text = (
                    choices[0].get("message", {}).get("content", "") if choices else ""
                )
                finish_reason = (
                    choices[0].get("finish_reason", "stop") if choices else "stop"
                )
                usage = response.get("usage", {})

            else:
                kwargs = {
                    "prompt": raw_prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "stream": False,
                }
                if request.stop_sequences:
                    kwargs["stop"] = request.stop_sequences
                if grammar is not None:
                    kwargs["grammar"] = grammar

                response = llm.create_completion(**kwargs)
                choices = response.get("choices", [])
                text = choices[0].get("text", "") if choices else ""
                finish_reason = (
                    choices[0].get("finish_reason", "stop") if choices else "stop"
                )
                usage = response.get("usage", {})

            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            # Verify deadline was not exceeded during blocking execution
            if (time.monotonic() - start_time) > timeout_sec:
                raise BackendTimeoutError(
                    f"Local inference exceeded deadline of {timeout_sec:.1f}s. "
                    "Note: caller deadline does not guarantee underlying C-runtime stopped."
                )

            return GenerationResult(
                text=text or "",
                model_id=str(self.model_path.name) if self.model_path else "llama_cpp",
                backend_id="llama_cpp",
                finish_reason=str(finish_reason or "stop"),
                tokens_prompt=usage.get("prompt_tokens"),
                tokens_completion=usage.get("completion_tokens"),
                elapsed_ms=elapsed_ms,
            )

        except (
            BackendTimeoutError,
            BackendCancelledError,
            BackendStructuredOutputError,
        ):
            raise
        except Exception as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "cuda error" in err_msg:
                raise BackendResourceError(
                    f"Resource exhaustion during local inference: {e}"
                ) from e
            raise InferenceBackendError(f"llama.cpp generation failed: {e}") from e

    def generate_stream(
        self,
        request: GenerationRequest,
        cancel_token: threading.Event | None = None,
    ) -> Iterator[str]:
        """Streaming generation — yields text tokens incrementally."""
        if not self.is_available():
            raise BackendNotAvailableError(
                "LlamaCppBackend is not available: either llama-cpp-python is not installed, "
                "or the configured model_path is missing or invalid."
            )

        if cancel_token is not None and cancel_token.is_set():
            raise BackendCancelledError(
                "Streaming generation cancelled by caller before stream initiation."
            )

        grammar = self._compile_grammar(request.structured_schema)
        llm = self._get_or_load_llm()

        raw_prompt, messages = self._prepare_prompt_and_messages(request)
        timeout_sec = request.timeout_seconds if request.timeout_seconds > 0 else 30.0
        start_time = time.monotonic()

        try:
            if messages is not None and hasattr(llm, "create_chat_completion"):
                kwargs: dict[str, Any] = {
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "stream": True,
                }
                if request.stop_sequences:
                    kwargs["stop"] = request.stop_sequences
                if grammar is not None:
                    kwargs["grammar"] = grammar

                stream_iter = llm.create_chat_completion(**kwargs)

                for chunk in stream_iter:
                    if cancel_token is not None and cancel_token.is_set():
                        raise BackendCancelledError(
                            "Streaming cancelled by caller token during generation. "
                            "Partial output yielded before cancellation is incomplete."
                        )
                    if (time.monotonic() - start_time) > timeout_sec:
                        raise BackendTimeoutError(
                            f"Streaming exceeded wall-clock deadline of {timeout_sec:.1f}s. "
                            "Note: caller deadline does not guarantee remote server has stopped."
                        )

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token

            else:
                kwargs = {
                    "prompt": raw_prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "stream": True,
                }
                if request.stop_sequences:
                    kwargs["stop"] = request.stop_sequences
                if grammar is not None:
                    kwargs["grammar"] = grammar

                stream_iter = llm.create_completion(**kwargs)

                for chunk in stream_iter:
                    if cancel_token is not None and cancel_token.is_set():
                        raise BackendCancelledError(
                            "Streaming cancelled by caller token during generation. "
                            "Partial output yielded before cancellation is incomplete."
                        )
                    if (time.monotonic() - start_time) > timeout_sec:
                        raise BackendTimeoutError(
                            f"Streaming exceeded wall-clock deadline of {timeout_sec:.1f}s. "
                            "Note: caller deadline does not guarantee underlying C-runtime stopped."
                        )

                    choices = chunk.get("choices", [])
                    if choices:
                        token = choices[0].get("text", "")
                        if token:
                            yield token

        except (
            BackendCancelledError,
            BackendTimeoutError,
            BackendStructuredOutputError,
        ):
            raise
        except Exception as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "cuda error" in err_msg:
                raise BackendResourceError(
                    f"Resource exhaustion during streaming: {e}"
                ) from e
            raise InferenceBackendError(f"llama.cpp streaming failed: {e}") from e

    # ──────────────────────────────────────────────────────────────────
    # Resource Lifecycle & Explicit Cleanup
    # ──────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Explicitly release native llama.cpp model context and memory.

        Thread-safe. Once closed, the model handle is freed and will be lazily
        re-instantiated if generation is requested again.
        """
        with self._load_lock:
            if self._llm is not None:
                try:
                    if hasattr(self._llm, "close"):
                        self._llm.close()
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Error releasing native llama.cpp model handle: %s", e
                    )
                finally:
                    self._llm = None

    # ──────────────────────────────────────────────────────────────────
    # Privacy-Preserving Telemetry
    # ──────────────────────────────────────────────────────────────────

    def get_resource_telemetry(self) -> dict[str, Any]:
        """Return non-sensitive local runtime telemetry.

        Never exposes prompts, file paths, database schemas, or credentials.
        """
        return {
            "backend_id": "llama_cpp",
            "is_local": True,
            "engine_installed": LLAMA_CPP_AVAILABLE,
            "model_loaded": self._llm is not None,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
        }

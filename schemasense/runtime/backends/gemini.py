"""SchemaSense Gemini Backend — google-genai implementation of InferenceBackend.

Architecture and Guarantees
----------------------------
* Implements :class:`InferenceBackend` for Google Gemini models via ``google-genai``.
* All Google GenAI SDK imports (``google.genai``, ``types``) are strictly isolated
  within this module.  Callers like ``SchemaSenseAIAgent`` or the application layer
  do not import Google GenAI directly.
* Remote provenance is explicit: ``BackendInfo.is_local`` is ``False``;
  ``BackendInfo.requires_api_key`` is ``True``.  There is no silent local-to-cloud
  or cloud-to-local fallback.
* SQLGuard is **not** enforced here.  Generated text containing SQL must still pass
  through ``schemasense.sql_guard.execute_safe_query`` before execution.

Timeout & Cancellation Semantics
--------------------------------
1. SDK-level transport deadline:
   When supported by the installed SDK, timeouts are passed to
   ``google.genai.types.HttpOptions(timeout=int(timeout_seconds * 1000))``,
   instructing the HTTP transport client to terminate the socket connection.
2. Caller-side deadline limitation:
   A caller-side deadline or :class:`BackendTimeoutError` indicates that the
   wall-clock allowance has elapsed.  It is **not** proof that the remote
   Google server or a running background thread has terminated.  Python threads
   cannot be forcibly killed.  Any late results produced by a timed-out or
   abandoned request are discarded and never treated as successful.
3. Cancellation limitation:
   The ``cancel_token`` :class:`threading.Event` is checked before starting
   single-shot generation and before yielding each streaming chunk.  However,
   the synchronous ``google.genai`` SDK does not support preemptive mid-flight
   cancellation of an in-flight network request once sent to the wire.
   For cooperative cancellation, use :meth:`generate_stream`.

Credential Hygiene
------------------
* API keys are never logged, formatted into exception messages, or exposed in
  telemetry.  Only key indices or counts are logged.
* Placeholder / demo keys (e.g. ``"DEMO_KEY_STANDBY"``) are recognized and
  never treated as live credentials.  If no valid non-placeholder keys are
  configured, :meth:`is_available` returns ``False`` and generation attempts
  raise :class:`BackendNotAvailableError`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from schemasense.runtime.backends.base import (
    BackendCancelledError,
    BackendCapability,
    BackendInfo,
    BackendNotAvailableError,
    BackendResourceError,
    BackendTimeoutError,
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
    InferenceBackendError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# Google SDK imports strictly isolated to this module
try:
    from google import genai
    from google.genai import types

    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    GOOGLE_GENAI_AVAILABLE = False

logger = logging.getLogger("schemasense.runtime.backends.gemini")

# Known placeholder key patterns that indicate missing live credentials
KNOWN_PLACEHOLDER_KEYS = frozenset(
    {
        "DEMO_KEY_STANDBY",
        "NO_KEY",
        "CHANGE_ME",
        "YOUR_API_KEY",
        "YOUR_GEMINI_API_KEY",
        "YOUR_GEMINI_API_KEY_HERE",
        "YOUR-GEMINI-API-KEY-HERE",
        "PLACEHOLDER",
        "DUMMY",
        "DEMO",
    }
)


def is_placeholder_key(key: str | None) -> bool:
    """Return True if the key is empty, None, or a known placeholder token.

    Never logs or leaks the key content.
    """
    if not key or not isinstance(key, str):
        return True
    cleaned = key.strip()
    if not cleaned:
        return True
    upper = cleaned.upper()
    if upper in KNOWN_PLACEHOLDER_KEYS:
        return True
    return upper.startswith(("DEMO_KEY", "PLACEHOLDER_", "YOUR_"))


def load_api_keys() -> list[str]:
    """Loads Gemini API keys from environment variables or Streamlit secrets.

    Returns ``["DEMO_KEY_STANDBY"]`` as fallback if no valid key is configured.
    """
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return [env_key]

    try:
        import streamlit as st

        keys = st.secrets.get("GEMINI_KEYS", None)
        if keys and isinstance(keys, list):
            valid_keys = [k for k in keys if k and isinstance(k, str) and k.strip()]
            if valid_keys:
                return valid_keys
    except Exception:  # noqa: BLE001, S110
        pass

    try:
        import streamlit as st

        single = st.secrets.get("GEMINI_API_KEY", "")
        if single and isinstance(single, str) and single.strip():
            return [single.strip()]
    except Exception:  # noqa: BLE001, S110
        pass

    return ["DEMO_KEY_STANDBY"]


class GeminiBackend(InferenceBackend):
    """Google Gemini inference adapter implementing the InferenceBackend contract.

    Manages credential rotation across a pool of API keys, configures
    structured output and streaming, and enforces bounded execution timeouts.
    """

    def __init__(
        self,
        api_keys: Sequence[str] | None = None,
        model_id: str = "gemini-2.5-flash",
        timeout_seconds: float = 20.0,
    ) -> None:
        raw_keys = list(api_keys) if api_keys is not None else load_api_keys()
        self.api_keys: list[str] = raw_keys
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self._active_key_index = 0

    # ──────────────────────────────────────────────────────────────────
    # Credential management & rotation
    # ──────────────────────────────────────────────────────────────────

    @property
    def active_key_index(self) -> int:
        """Zero-based index of the currently active API key in the pool."""
        return self._active_key_index

    @active_key_index.setter
    def active_key_index(self, index: int) -> None:
        if self.api_keys:
            self._active_key_index = index % len(self.api_keys)
        else:
            self._active_key_index = 0

    @property
    def active_key(self) -> str:
        """Returns the currently active key string.

        Callers must NEVER log or print this value.
        """
        if not self.api_keys:
            return "NO_KEY"
        return self.api_keys[self.active_key_index % len(self.api_keys)]

    def rotate_key(self) -> str:
        """Advances to the next key in the pool and returns it.

        Never logs the key material; logs only the new key index.
        """
        if not self.api_keys:
            return "NO_KEY"
        self._active_key_index = (self._active_key_index + 1) % len(self.api_keys)
        logger.info("Rotated to API key index %d", self._active_key_index)
        return self.active_key

    # ──────────────────────────────────────────────────────────────────
    # InferenceBackend interface
    # ──────────────────────────────────────────────────────────────────

    def get_info(self) -> BackendInfo:
        """Return static metadata and declared capabilities.

        Completes without I/O and never raises.
        """
        return BackendInfo(
            backend_id="gemini",
            display_name="Google Gemini (cloud)",
            is_local=False,
            capabilities=frozenset(
                {
                    BackendCapability.STREAMING,
                    BackendCapability.STRUCTURED_OUTPUT,
                }
            ),
            model_id=self.model_id,
            engine_version="google-genai",
            requires_api_key=True,
        )

    def is_available(self) -> bool:
        """Returns True if the SDK is installed and at least one non-placeholder key exists.

        Completes in under 100ms without network I/O.  Never raises.
        """
        if not GOOGLE_GENAI_AVAILABLE:
            return False
        if not self.api_keys:
            return False
        # Must have at least one credential that is not a placeholder/demo key
        return any(not is_placeholder_key(k) for k in self.api_keys)

    def _create_client(self, api_key: str, timeout_ms: int | None = None) -> Any:
        """Constructs a google-genai Client with optional transport timeout."""
        if not GOOGLE_GENAI_AVAILABLE or genai is None:
            raise BackendNotAvailableError(
                "The google-genai SDK is not installed. Install via: pip install google-genai"
            )

        http_options = None
        if (
            timeout_ms is not None
            and types is not None
            and hasattr(types, "HttpOptions")
        ):
            try:
                http_options = types.HttpOptions(timeout=timeout_ms)
            except Exception:  # noqa: BLE001
                http_options = None

        if http_options is not None:
            try:
                return genai.Client(api_key=api_key, http_options=http_options)
            except TypeError:
                return genai.Client(api_key=api_key)

        return genai.Client(api_key=api_key)

    def _build_sdk_contents(self, request: GenerationRequest) -> list[Any] | str:
        """Formats GenerationRequest prompt and history into google-genai contents."""
        if not request.history:
            return request.prompt

        contents: list[Any] = []
        for msg in request.history:
            role = "user" if msg.get("role") == "user" else "model"
            content_text = msg.get("content", "")
            if content_text:
                if (
                    types is not None
                    and hasattr(types, "Content")
                    and hasattr(types, "Part")
                ):
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=content_text)],
                        )
                    )
                else:
                    contents.append({"role": role, "parts": [{"text": content_text}]})

        if types is not None and hasattr(types, "Content") and hasattr(types, "Part"):
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=request.prompt)],
                )
            )
        else:
            contents.append({"role": "user", "parts": [{"text": request.prompt}]})

        return contents

    def _build_config(self, request: GenerationRequest, timeout_ms: int) -> Any:
        """Constructs GenerateContentConfig from GenerationRequest."""
        if types is None or not hasattr(types, "GenerateContentConfig"):
            return None

        config_kwargs: dict[str, Any] = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
        }
        if request.system_instruction:
            config_kwargs["system_instruction"] = request.system_instruction
        if request.stop_sequences:
            config_kwargs["stop_sequences"] = request.stop_sequences
        if request.structured_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = request.structured_schema

        if hasattr(types, "HttpOptions"):
            try:
                config_kwargs["http_options"] = types.HttpOptions(timeout=timeout_ms)
            except Exception:  # noqa: BLE001, S110
                pass

        try:
            return types.GenerateContentConfig(**config_kwargs)
        except Exception:  # noqa: BLE001
            # Fallback if specific kwargs are rejected by mock or schema version
            return types.GenerateContentConfig(temperature=request.temperature)

    def generate(
        self,
        request: GenerationRequest,
        cancel_token: threading.Event | None = None,
    ) -> GenerationResult:
        """Blocking single-shot generation via google-genai SDK.

        Raises:
            BackendNotAvailableError: If SDK is absent or no valid keys configured.
            BackendCancelledError: If cancel_token is set prior to start.
            BackendTimeoutError: If request deadline expires.
            BackendResourceError: If quota/rate limits are exhausted on all keys.
            InferenceBackendError: On unrecoverable provider failures.
        """
        if not self.is_available():
            raise BackendNotAvailableError(
                "Gemini backend is not available: Google GenAI SDK is not installed or "
                "no valid Gemini API keys are configured (only placeholder credentials found)."
            )

        if cancel_token is not None and cancel_token.is_set():
            raise BackendCancelledError(
                "Generation cancelled by caller before request started."
            )

        timeout_sec = (
            request.timeout_seconds
            if request.timeout_seconds > 0
            else self.timeout_seconds
        )
        timeout_ms = int(timeout_sec * 1000)

        attempts = len(self.api_keys)
        last_error: Exception | None = None

        for _ in range(attempts):
            if cancel_token is not None and cancel_token.is_set():
                raise BackendCancelledError(
                    "Generation cancelled by caller before initiating turn."
                )

            # Skip placeholder keys in rotation if others are available
            if is_placeholder_key(self.active_key):
                self.rotate_key()
                continue

            client = self._create_client(self.active_key, timeout_ms=timeout_ms)
            config = self._build_config(request, timeout_ms=timeout_ms)
            contents = self._build_sdk_contents(request)

            start_time = time.monotonic()
            try:
                response = client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=config,
                )

                elapsed_ms = (time.monotonic() - start_time) * 1000.0
                text = response.text if (response and response.text) else ""

                tokens_prompt = getattr(
                    getattr(response, "usage_metadata", None),
                    "prompt_token_count",
                    None,
                )
                tokens_completion = getattr(
                    getattr(response, "usage_metadata", None),
                    "candidates_token_count",
                    None,
                )

                return GenerationResult(
                    text=text,
                    model_id=self.model_id,
                    backend_id="gemini",
                    finish_reason="stop",
                    tokens_prompt=tokens_prompt,
                    tokens_completion=tokens_completion,
                    elapsed_ms=elapsed_ms,
                )

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                    last_error = BackendResourceError(
                        f"Gemini quota/rate limit exhausted on key index {self.active_key_index}: {e}"
                    )
                    self.rotate_key()
                elif "timeout" in err_str or "deadline" in err_str:
                    last_error = BackendTimeoutError(
                        f"Gemini API request timed out after {timeout_sec:.1f}s on key index {self.active_key_index}: {e}. "
                        "Note: SDK/caller deadline does not guarantee termination of remote worker."
                    )
                    self.rotate_key()
                else:
                    raise InferenceBackendError(
                        f"Gemini API provider error: {e}"
                    ) from e

        if isinstance(last_error, BackendTimeoutError):
            raise last_error
        if isinstance(last_error, BackendResourceError):
            raise last_error

        raise BackendResourceError(
            f"Exhausted all available API keys without receiving a response: {last_error}"
        )

    def generate_stream(
        self,
        request: GenerationRequest,
        cancel_token: threading.Event | None = None,
    ) -> Iterator[str]:
        """Streaming generation — yields text chunks incrementally.

        Monitors ``cancel_token`` and wall-clock deadline between chunks.
        """
        if not self.is_available():
            raise BackendNotAvailableError(
                "Gemini backend is not available: Google GenAI SDK is not installed or "
                "no valid Gemini API keys are configured (only placeholder credentials found)."
            )

        if cancel_token is not None and cancel_token.is_set():
            raise BackendCancelledError(
                "Streaming generation cancelled by caller before starting."
            )

        timeout_sec = (
            request.timeout_seconds
            if request.timeout_seconds > 0
            else self.timeout_seconds
        )
        timeout_ms = int(timeout_sec * 1000)

        attempts = len(self.api_keys)
        last_error: Exception | None = None
        start_time = time.monotonic()

        for _ in range(attempts):
            if cancel_token is not None and cancel_token.is_set():
                raise BackendCancelledError(
                    "Streaming generation cancelled by caller before stream initiation."
                )

            if is_placeholder_key(self.active_key):
                self.rotate_key()
                continue

            client = self._create_client(self.active_key, timeout_ms=timeout_ms)
            config = self._build_config(request, timeout_ms=timeout_ms)
            contents = self._build_sdk_contents(request)

            try:
                stream = client.models.generate_content_stream(
                    model=self.model_id,
                    contents=contents,
                    config=config,
                )

                for chunk in stream:
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

                    if chunk and hasattr(chunk, "text") and chunk.text:
                        yield chunk.text
                    elif isinstance(chunk, str) and chunk:
                        yield chunk
                return

            except (BackendCancelledError, BackendTimeoutError):
                raise
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                    last_error = BackendResourceError(
                        f"Gemini streaming quota exhausted on key index {self.active_key_index}: {e}"
                    )
                    self.rotate_key()
                elif "timeout" in err_str or "deadline" in err_str:
                    last_error = BackendTimeoutError(
                        f"Gemini streaming timed out after {timeout_sec:.1f}s on key index {self.active_key_index}: {e}. "
                        "Note: caller deadline does not guarantee remote server has stopped."
                    )
                    self.rotate_key()
                else:
                    raise InferenceBackendError(
                        f"Gemini streaming provider error: {e}"
                    ) from e

        if isinstance(last_error, BackendTimeoutError):
            raise last_error
        if isinstance(last_error, BackendResourceError):
            raise last_error

        raise BackendResourceError(
            f"Exhausted all available API keys during streaming: {last_error}"
        )

    def get_resource_telemetry(self) -> dict[str, Any]:
        """Return non-sensitive telemetry without exposing key credentials."""
        return {
            "backend_id": "gemini",
            "model_id": self.model_id,
            "is_local": False,
            "active_key_index": self.active_key_index,
            "total_keys": len(self.api_keys),
            "sdk_available": GOOGLE_GENAI_AVAILABLE,
        }

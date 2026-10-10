"""SchemaSense Inference Backends — public package interface.

Exposes the backend-neutral contract types, exceptions, and concrete adapters.
"""

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
from schemasense.runtime.backends.gemini import GeminiBackend
from schemasense.runtime.backends.llama_cpp import LlamaCppBackend

__all__ = [
    "BackendCancelledError",
    "BackendCapability",
    "BackendInfo",
    "BackendNotAvailableError",
    "BackendResourceError",
    "BackendStructuredOutputError",
    "BackendTimeoutError",
    "GeminiBackend",
    "GenerationRequest",
    "GenerationResult",
    "InferenceBackend",
    "InferenceBackendError",
    "LlamaCppBackend",
]

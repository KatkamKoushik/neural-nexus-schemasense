"""SchemaSense Local AI Runtime — Core Types, Data Contracts, and Enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class BackendAccelerator(Enum):
    """Underlying accelerator capability discovered on the host system."""

    NONE = "none"
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    ROCM = "rocm"
    DXGI = "dxgi"
    UNKNOWN = "unknown"


class ModelFormat(Enum):
    """Supported binary model container formats."""

    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    UNKNOWN = "unknown"


class ModelArchitecture(Enum):
    """Normalized transformer model family."""

    LLAMA = "llama"
    QWEN2 = "qwen2"
    MISTRAL = "mistral"
    GEMMA2 = "gemma2"
    DEEPSEEK = "deepseek"
    UNKNOWN = "unknown"


class CompatibilityStatus(Enum):
    """Verdict on model compatibility against host resources."""

    CONFIDENT_FIT = "confident_fit"
    MARGINAL_FIT = "marginal_fit"
    LIKELY_OOM = "likely_oom"
    INSUFFICIENT_METADATA = "insufficient_metadata"
    UNSUPPORTED_ARCH = "unsupported_arch"


class ConfidenceLevel(Enum):
    """Degree of confidence behind compatibility sizing estimates."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass(frozen=True)
class GPUInfo:
    """Discovered GPU/accelerator details."""

    name: str
    accelerator: BackendAccelerator
    total_vram_mb: int | None = None
    available_vram_mb: int | None = None
    is_unified_memory: bool = False
    details: str = ""


@dataclass(frozen=True)
class HardwareProfile:
    """Host system compute, memory, and accelerator capacity."""

    total_ram_mb: int
    available_ram_mb: int
    total_swap_mb: int | None = None
    available_swap_mb: int | None = None
    cpu_cores_physical: int | None = None
    cpu_cores_logical: int = 1
    cpu_arch: str = "unknown"
    is_wsl: bool = False
    wsl_warning: str | None = None
    gpus: list[GPUInfo] = field(default_factory=list)
    storage_total_gb: float | None = None
    storage_free_gb: float | None = None


@dataclass
class ModelHyperparameters:
    """Extracted transformer architecture parameters."""

    architecture: ModelArchitecture = ModelArchitecture.UNKNOWN
    raw_architecture_name: str = ""
    layer_count: int | None = None  # L (block_count)
    embedding_length: int | None = None  # d_model
    feed_forward_length: int | None = None  # d_ffn
    head_count: int | None = None  # n_heads
    head_count_kv: int | None = None  # n_kv_heads (GQA)
    key_length: int | None = None  # explicit key dimension if decoupled
    value_length: int | None = None  # explicit value dimension if decoupled
    context_length: int | None = None  # max trained context window
    sliding_window: int | None = None  # SWA window size if active
    sliding_window_layers: int | None = None  # verified layer count using SWA if hybrid
    sliding_window_pattern: str | None = (
        None  # verified pattern: 'all', 'alternating', etc.
    )
    expert_count: int | None = None  # Total MoE experts
    expert_used_count: int | None = None  # Active routed experts per token


@dataclass
class ModelManifest:
    """Ground-truth manifest representing an inspected local model file."""

    model_id: str
    file_path: Path
    file_size_bytes: int
    format: ModelFormat
    hyperparameters: ModelHyperparameters | None = None
    tensor_count: int = 0
    quantization_type: str = "UNKNOWN"
    sha256_checksum: str | None = None
    checksum_verified: bool = False
    license_str: str | None = None
    untrusted_chat_template: str | None = None
    provenance_notes: list[str] = field(default_factory=list)
    filename: str = ""
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.filename and self.file_path:
            self.filename = self.file_path.name
        if not self.display_name and self.file_path:
            self.display_name = self.file_path.stem


@dataclass
class SizingReport:
    """Evidence-based model sizing and compatibility report."""

    model_id: str
    target_context_tokens: int
    status: CompatibilityStatus
    confidence: ConfidenceLevel
    verified_weights_mb: int
    estimated_kv_cache_mb: int | None = None
    estimated_activation_mb: int | None = None
    total_estimated_ram_mb: int | None = None
    advisory_gpu_layers: int | None = None
    advisory_vram_usage_mb: int | None = None
    evidence_notes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

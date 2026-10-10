"""SchemaSense Local AI Runtime — Bounded Binary GGUF v2/v3 Header Parser."""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Any, ClassVar

from schemasense.runtime.types import (
    ModelArchitecture,
    ModelFormat,
    ModelHyperparameters,
    ModelManifest,
)

GGUF_MAGIC = 0x46554747  # b"GGUF" in little-endian uint32
MAX_HEADER_SCAN_BYTES = 16 * 1024 * 1024  # 16 MB maximum header read boundary
MAX_METADATA_KV_COUNT = 1000  # Prevent infinite/loop DoS
MAX_STRING_LENGTH_BYTES = 65536  # 64 KB maximum string length
MAX_ARRAY_LENGTH_ELEMENTS = 65536  # Maximum array elements


class GGUFParseError(ValueError):
    """Raised when GGUF file structure is invalid, truncated, or exceeds safety limits."""


class GGUFParser:
    """Zero-dependency, hardened binary reader for GGUF v2/v3 model metadata."""

    VALUE_TYPES: ClassVar[dict[int, str]] = {
        0: "UINT8",
        1: "INT8",
        2: "UINT16",
        3: "INT16",
        4: "UINT32",
        5: "INT32",
        6: "FLOAT32",
        7: "BOOL",
        8: "STRING",
        9: "ARRAY",
        10: "UINT64",
        11: "INT64",
        12: "FLOAT64",
    }

    FILE_TYPES_QUANT: ClassVar[dict[int, str]] = {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        7: "Q8_0",
        8: "Q5_0",
        9: "Q5_1",
        10: "Q2_K",
        11: "Q3_K_S",
        12: "Q3_K_M",
        13: "Q3_K_L",
        14: "Q4_K_S",
        15: "Q4_K_M",
        16: "Q5_K_S",
        17: "Q5_K_M",
        18: "Q6_K",
    }

    @classmethod
    def read_manifest(cls, file_path: Path | str) -> ModelManifest:
        """Parses model manifest and hyperparameters from GGUF file header within strict bounds."""
        p = Path(file_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Model file not found: {p}")

        file_size = p.stat().st_size
        if file_size < 24:
            raise GGUFParseError(
                f"File too small to be a valid GGUF container: {file_size} bytes"
            )

        # Read only up to MAX_HEADER_SCAN_BYTES into an in-memory buffer
        with open(p, "rb") as f:
            header_chunk = f.read(min(file_size, MAX_HEADER_SCAN_BYTES))

        stream = io.BytesIO(header_chunk)
        magic, version, tensor_count, kv_count = cls._unpack("<IIQQ", stream, 24)

        if magic != GGUF_MAGIC:
            raise GGUFParseError(
                f"Invalid GGUF magic bytes (got 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X})"
            )

        if version not in (2, 3):
            raise GGUFParseError(
                f"Unsupported GGUF version: {version} (supports v2 and v3)"
            )

        if kv_count > MAX_METADATA_KV_COUNT:
            raise GGUFParseError(
                f"Metadata key-value count ({kv_count}) exceeds security boundary ({MAX_METADATA_KV_COUNT})"
            )

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = cls._read_string(stream)
            val_type_int = cls._unpack("<I", stream, 4)[0]
            if val_type_int not in cls.VALUE_TYPES:
                raise GGUFParseError(
                    f"Unknown GGUF value type: {val_type_int} for key '{key}'"
                )
            val = cls._read_value(stream, val_type_int)
            metadata[key] = val

        hyperparams = cls._extract_hyperparameters(metadata)
        quant = cls._determine_quantization(metadata, p.name)

        return ModelManifest(
            model_id=p.stem,
            file_path=p,
            file_size_bytes=file_size,
            format=ModelFormat.GGUF,
            hyperparameters=hyperparams,
            tensor_count=tensor_count,
            quantization_type=quant,
            sha256_checksum=None,
            checksum_verified=False,
            license_str=str(metadata.get("general.license", "")) or None,
            untrusted_chat_template=str(metadata.get("tokenizer.chat_template", ""))
            or None,
            provenance_notes=[
                f"GGUF v{version} container",
                f"Raw Architecture: {metadata.get('general.architecture', 'unknown')}",
            ],
        )

    @classmethod
    def _read_string(cls, stream: io.BytesIO) -> str:
        """Reads length-prefixed UTF-8 string with length bounds."""
        length = cls._unpack("<Q", stream, 8)[0]
        if length > MAX_STRING_LENGTH_BYTES:
            raise GGUFParseError(
                f"String length ({length} bytes) exceeds safety limit ({MAX_STRING_LENGTH_BYTES})"
            )
        raw_bytes = stream.read(length)
        if len(raw_bytes) < length:
            raise GGUFParseError("Unexpected EOF while reading GGUF string")
        return raw_bytes.decode("utf-8", errors="replace")

    @classmethod
    def _read_value(cls, stream: io.BytesIO, val_type: int) -> Any:
        """Decodes standard GGUF scalar or array values."""
        if val_type == 0:  # UINT8
            return cls._unpack("<B", stream, 1)[0]
        elif val_type == 1:  # INT8
            return cls._unpack("<b", stream, 1)[0]
        elif val_type == 2:  # UINT16
            return cls._unpack("<H", stream, 2)[0]
        elif val_type == 3:  # INT16
            return cls._unpack("<h", stream, 2)[0]
        elif val_type == 4:  # UINT32
            return cls._unpack("<I", stream, 4)[0]
        elif val_type == 5:  # INT32
            return cls._unpack("<i", stream, 4)[0]
        elif val_type == 6:  # FLOAT32
            return cls._unpack("<f", stream, 4)[0]
        elif val_type == 7:  # BOOL
            return cls._unpack("<?", stream, 1)[0]
        elif val_type == 8:  # STRING
            return cls._read_string(stream)
        elif val_type == 9:  # ARRAY
            item_type, arr_len = cls._unpack("<IQ", stream, 12)
            if arr_len > MAX_ARRAY_LENGTH_ELEMENTS:
                raise GGUFParseError(
                    f"Array length ({arr_len}) exceeds safety limit ({MAX_ARRAY_LENGTH_ELEMENTS})"
                )
            items = []
            for _ in range(arr_len):
                items.append(cls._read_value(stream, item_type))
            return items
        elif val_type == 10:  # UINT64
            return cls._unpack("<Q", stream, 8)[0]
        elif val_type == 11:  # INT64
            return cls._unpack("<q", stream, 8)[0]
        elif val_type == 12:  # FLOAT64
            return cls._unpack("<d", stream, 8)[0]
        raise GGUFParseError(f"Unsupported value type id: {val_type}")

    @classmethod
    def _unpack(
        cls, fmt: str, stream: io.BytesIO, expected_len: int
    ) -> tuple[Any, ...]:
        buf = stream.read(expected_len)
        if len(buf) < expected_len:
            raise GGUFParseError(
                f"Unexpected EOF in GGUF stream (needed {expected_len} bytes, got {len(buf)})"
            )
        return struct.unpack(fmt, buf)

    @classmethod
    def _extract_hyperparameters(cls, metadata: dict[str, Any]) -> ModelHyperparameters:
        """Extracts architecture and dimensional hyperparameters from normalized GGUF keys."""
        raw_arch = str(metadata.get("general.architecture", "")).lower()
        arch_enum = cls._normalize_architecture(raw_arch)

        # Prefix for architecture keys, e.g. "llama.block_count" or "qwen2.block_count"
        prefix = f"{raw_arch}." if raw_arch else ""

        def get_int(*keys: str) -> int | None:
            for k in keys:
                v = metadata.get(f"{prefix}{k}", metadata.get(k))
                if isinstance(v, int):
                    return v
            return None

        layers = get_int("block_count", "layer_count")
        embed_dim = get_int("embedding_length")
        ffn_dim = get_int("feed_forward_length")
        heads = get_int("attention.head_count")
        heads_kv = get_int("attention.head_count_kv")
        key_len = get_int("attention.key_length")
        val_len = get_int("attention.value_length")
        context_len = get_int("context_length")
        sliding_window = get_int("attention.sliding_window", "sliding_window")
        swa_layers = get_int(
            "attention.sliding_window_layers",
            "sliding_window_layers",
            "max_window_layers",
        )
        swa_pattern = None
        for k in ("attention.sliding_window_pattern", "sliding_window_pattern"):
            pat_val = metadata.get(f"{prefix}{k}", metadata.get(k))
            if isinstance(pat_val, str) and pat_val.strip():
                swa_pattern = pat_val.strip().lower()
                break

        experts = get_int("expert_count")
        experts_used = get_int("expert_used_count")

        return ModelHyperparameters(
            architecture=arch_enum,
            raw_architecture_name=raw_arch,
            layer_count=layers,
            embedding_length=embed_dim,
            feed_forward_length=ffn_dim,
            head_count=heads,
            head_count_kv=heads_kv,
            key_length=key_len,
            value_length=val_len,
            context_length=context_len,
            sliding_window=sliding_window,
            sliding_window_layers=swa_layers,
            sliding_window_pattern=swa_pattern,
            expert_count=experts,
            expert_used_count=experts_used,
        )

    @classmethod
    def _normalize_architecture(cls, raw: str) -> ModelArchitecture:
        if "llama" in raw:
            return ModelArchitecture.LLAMA
        if "qwen" in raw:
            return ModelArchitecture.QWEN2
        if "mistral" in raw:
            return ModelArchitecture.MISTRAL
        if "gemma" in raw:
            return ModelArchitecture.GEMMA2
        if "deepseek" in raw:
            return ModelArchitecture.DEEPSEEK
        return ModelArchitecture.UNKNOWN

    @classmethod
    def _determine_quantization(cls, metadata: dict[str, Any], filename: str) -> str:
        ft = metadata.get("general.file_type")
        if isinstance(ft, int) and ft in cls.FILE_TYPES_QUANT:
            return cls.FILE_TYPES_QUANT[ft]

        # Fallback to inspecting common filename quantization suffixes (e.g. Q4_K_M)
        upper = filename.upper()
        for q in (
            "Q4_K_M",
            "Q4_K_S",
            "Q5_K_M",
            "Q5_K_S",
            "Q8_0",
            "Q4_0",
            "Q5_0",
            "F16",
            "BF16",
        ):
            if q in upper:
                return q
        return "UNKNOWN"

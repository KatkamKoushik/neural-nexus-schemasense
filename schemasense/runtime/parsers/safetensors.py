"""SchemaSense Local AI Runtime — SafeTensors Container and Config Parser."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from schemasense.runtime.types import (
    ModelArchitecture,
    ModelFormat,
    ModelHyperparameters,
    ModelManifest,
)

MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024  # 16 MB boundary


class SafeTensorsParseError(ValueError):
    """Raised when SafeTensors header is malformed, truncated, or exceeds safety bounds."""


class SafeTensorsParser:
    """Reads SafeTensors JSON header and merges co-located architecture config.json if present."""

    @classmethod
    def read_manifest(cls, file_path: Path | str) -> ModelManifest:
        p = Path(file_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"SafeTensors file not found: {p}")

        file_size = p.stat().st_size
        if file_size < 8:
            raise SafeTensorsParseError(
                f"File too small for SafeTensors container: {file_size} bytes"
            )

        with open(p, "rb") as f:
            raw_len = f.read(8)
            header_len = struct.unpack("<Q", raw_len)[0]

            if header_len > MAX_SAFETENSORS_HEADER_BYTES:
                raise SafeTensorsParseError(
                    f"SafeTensors header length ({header_len} bytes) exceeds safety limit ({MAX_SAFETENSORS_HEADER_BYTES})"
                )

            if header_len > file_size - 8:
                raise SafeTensorsParseError(
                    "SafeTensors header length exceeds actual file size"
                )

            raw_header = f.read(header_len)
            if len(raw_header) < header_len:
                raise SafeTensorsParseError(
                    "Unexpected EOF while reading SafeTensors header JSON"
                )

        try:
            header_data = json.loads(raw_header.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise SafeTensorsParseError(
                f"Malformed SafeTensors JSON header: {e}"
            ) from e

        metadata_dict = header_data.pop("__metadata__", {})
        tensor_count = len(header_data)

        # Check for co-located config.json
        hyperparams, config_found = cls._load_companion_config(p.parent)

        provenance = [
            "SafeTensors container",
            f"Tensor count: {tensor_count}",
        ]
        if config_found:
            provenance.append(
                "Architecture parameters merged from companion config.json"
            )
        else:
            provenance.append(
                "No companion config.json found; architectural parameters unknown"
            )

        # Determine dominant dtype
        dominant_dtype = cls._determine_dominant_dtype(header_data)

        return ModelManifest(
            model_id=p.stem,
            file_path=p,
            file_size_bytes=file_size,
            format=ModelFormat.SAFETENSORS,
            hyperparameters=hyperparams,
            tensor_count=tensor_count,
            quantization_type=dominant_dtype,
            sha256_checksum=None,
            checksum_verified=False,
            license_str=metadata_dict.get("license"),
            untrusted_chat_template=metadata_dict.get("chat_template"),
            provenance_notes=provenance,
        )

    @classmethod
    def _determine_dominant_dtype(cls, tensors_dict: dict[str, dict]) -> str:
        for t_info in tensors_dict.values():
            if isinstance(t_info, dict) and "dtype" in t_info:
                return str(t_info["dtype"])
        return "F16"

    @classmethod
    def _load_companion_config(
        cls, parent_dir: Path
    ) -> tuple[ModelHyperparameters | None, bool]:
        config_path = parent_dir / "config.json"
        if not config_path.is_file():
            return None, False

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            raw_arch = str(cfg.get("model_type", "")).lower()
            arch_enum = ModelArchitecture.UNKNOWN
            if "llama" in raw_arch:
                arch_enum = ModelArchitecture.LLAMA
            elif "qwen" in raw_arch:
                arch_enum = ModelArchitecture.QWEN2
            elif "mistral" in raw_arch:
                arch_enum = ModelArchitecture.MISTRAL
            elif "gemma" in raw_arch:
                arch_enum = ModelArchitecture.GEMMA2
            elif "deepseek" in raw_arch:
                arch_enum = ModelArchitecture.DEEPSEEK

            layers = cfg.get("num_hidden_layers", cfg.get("n_layer"))
            embed_dim = cfg.get("hidden_size", cfg.get("n_embd"))
            ffn_dim = cfg.get("intermediate_size")
            heads = cfg.get("num_attention_heads", cfg.get("n_head"))
            heads_kv = cfg.get("num_key_value_heads", heads)
            context_len = cfg.get("max_position_embeddings", cfg.get("n_positions"))
            sliding_window = cfg.get("sliding_window")
            swa_layers = cfg.get("max_window_layers", cfg.get("sliding_window_layers"))
            if not isinstance(swa_layers, int):
                swa_layers = None

            swa_pattern_raw = cfg.get("sliding_window_pattern")
            swa_pattern = (
                str(swa_pattern_raw).lower().strip()
                if isinstance(swa_pattern_raw, str) and str(swa_pattern_raw).strip()
                else None
            )

            return (
                ModelHyperparameters(
                    architecture=arch_enum,
                    raw_architecture_name=raw_arch,
                    layer_count=layers,
                    embedding_length=embed_dim,
                    feed_forward_length=ffn_dim,
                    head_count=heads,
                    head_count_kv=heads_kv,
                    context_length=context_len,
                    sliding_window=sliding_window,
                    sliding_window_layers=swa_layers,
                    sliding_window_pattern=swa_pattern,
                ),
                True,
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None, False

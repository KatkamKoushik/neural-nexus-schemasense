"""Unit & Adversarial Tests for GGUF and SafeTensors Binary Parsers."""

import io
import json
import struct
from pathlib import Path

import pytest

from schemasense.runtime.parsers.gguf import (
    GGUF_MAGIC,
    GGUFParseError,
    GGUFParser,
)
from schemasense.runtime.parsers.safetensors import (
    SafeTensorsParseError,
    SafeTensorsParser,
)
from schemasense.runtime.types import ModelArchitecture, ModelFormat


def build_synthetic_gguf_v3(
    architecture: str = "qwen2",
    block_count: int = 28,
    embedding_length: int = 3584,
    head_count: int = 28,
    head_count_kv: int = 4,
    context_length: int = 32768,
    sliding_window: int | None = None,
    license_str: str = "Apache-2.0",
    chat_template: str = "{% for m in messages %}{{ m.content }}{% endfor %}",
    file_type: int = 15,  # Q4_K_M
) -> bytes:
    """Helper to assemble a valid minimal binary GGUF v3 header."""
    buf = io.BytesIO()

    # 1. Header: Magic (4B), Version (4B), Tensor count (8B), Metadata KV count (8B)
    kv_items: list[tuple[str, int, bytes]] = []

    def add_str(key: str, val: str):
        val_bytes = val.encode("utf-8")
        payload = struct.pack("<Q", len(val_bytes)) + val_bytes
        kv_items.append((key, 8, payload))  # Type 8: STRING

    def add_u32(key: str, val: int):
        kv_items.append((key, 4, struct.pack("<I", val)))  # Type 4: UINT32

    add_str("general.architecture", architecture)
    add_str("general.license", license_str)
    add_u32("general.file_type", file_type)
    add_str("tokenizer.chat_template", chat_template)
    add_u32(f"{architecture}.block_count", block_count)
    add_u32(f"{architecture}.embedding_length", embedding_length)
    add_u32(f"{architecture}.attention.head_count", head_count)
    add_u32(f"{architecture}.attention.head_count_kv", head_count_kv)
    add_u32(f"{architecture}.context_length", context_length)

    if sliding_window is not None:
        add_u32(f"{architecture}.attention.sliding_window", sliding_window)

    buf.write(struct.pack("<IIQQ", GGUF_MAGIC, 3, 100, len(kv_items)))

    # 2. Key-Value pairs
    for k, val_type, payload in kv_items:
        k_bytes = k.encode("utf-8")
        buf.write(struct.pack("<Q", len(k_bytes)))
        buf.write(k_bytes)
        buf.write(struct.pack("<I", val_type))
        buf.write(payload)

    # Pad with some mock weight bytes
    buf.write(b"\x00" * 256)
    return buf.getvalue()


def test_gguf_synthetic_valid_parsing(tmp_path: Path):
    """Verifies that a valid GGUF v3 file extracts exact architectural parameters."""
    gguf_bytes = build_synthetic_gguf_v3(
        architecture="qwen2",
        block_count=28,
        embedding_length=3584,
        head_count=28,
        head_count_kv=4,
        context_length=32768,
        sliding_window=4096,
        license_str="Apache-2.0",
    )
    model_path = tmp_path / "qwen2_test.gguf"
    model_path.write_bytes(gguf_bytes)

    manifest = GGUFParser.read_manifest(model_path)

    assert manifest.format == ModelFormat.GGUF
    assert manifest.tensor_count == 100
    assert manifest.quantization_type == "Q4_K_M"
    assert manifest.license_str == "Apache-2.0"
    assert "tokenizer.chat_template" not in manifest.provenance_notes
    assert manifest.untrusted_chat_template is not None
    assert "{% for m in messages %}" in manifest.untrusted_chat_template

    hp = manifest.hyperparameters
    assert hp is not None
    assert hp.architecture == ModelArchitecture.QWEN2
    assert hp.layer_count == 28
    assert hp.embedding_length == 3584
    assert hp.head_count == 28
    assert hp.head_count_kv == 4
    assert hp.context_length == 32768
    assert hp.sliding_window == 4096


def test_gguf_adversarial_invalid_magic(tmp_path: Path):
    """Rejects files with invalid magic bytes."""
    bad_file = tmp_path / "bad_magic.gguf"
    bad_file.write_bytes(b"NOTGGUF_HEADER_DATA_1234567890")
    with pytest.raises(GGUFParseError, match="Invalid GGUF magic bytes"):
        GGUFParser.read_manifest(bad_file)


def test_gguf_adversarial_file_too_small(tmp_path: Path):
    """Rejects files smaller than minimum header size."""
    small_file = tmp_path / "too_small.gguf"
    small_file.write_bytes(b"GGUF\x03\x00\x00")
    with pytest.raises(GGUFParseError, match="too small"):
        GGUFParser.read_manifest(small_file)


def test_gguf_adversarial_unsupported_version(tmp_path: Path):
    """Rejects unsupported major GGUF versions."""
    bad_ver = struct.pack("<IIQQ", GGUF_MAGIC, 99, 10, 0)
    p = tmp_path / "bad_version.gguf"
    p.write_bytes(bad_ver)
    with pytest.raises(GGUFParseError, match="Unsupported GGUF version: 99"):
        GGUFParser.read_manifest(p)


def test_gguf_adversarial_kv_count_bomb(tmp_path: Path):
    """Rejects headers advertising excessively large KV pair counts."""
    bomb = struct.pack("<IIQQ", GGUF_MAGIC, 3, 10, 50000)
    p = tmp_path / "kv_bomb.gguf"
    p.write_bytes(bomb)
    with pytest.raises(GGUFParseError, match="exceeds security boundary"):
        GGUFParser.read_manifest(p)


def test_gguf_adversarial_string_length_bomb(tmp_path: Path):
    """Rejects strings claiming gigabytes of length to prevent memory exhaustion."""
    buf = io.BytesIO()
    buf.write(struct.pack("<IIQQ", GGUF_MAGIC, 3, 10, 1))
    # String length = 10,000,000 bytes (>64KB limit)
    buf.write(struct.pack("<Q", 10_000_000))
    p = tmp_path / "string_bomb.gguf"
    p.write_bytes(buf.getvalue())
    with pytest.raises(GGUFParseError, match="String length .* exceeds safety limit"):
        GGUFParser.read_manifest(p)


def test_gguf_adversarial_array_length_bomb(tmp_path: Path):
    """Rejects array allocations exceeding element limits."""
    buf = io.BytesIO()
    buf.write(struct.pack("<IIQQ", GGUF_MAGIC, 3, 10, 1))
    key_b = b"test.array"
    buf.write(struct.pack("<Q", len(key_b)) + key_b)
    buf.write(struct.pack("<I", 9))  # Type 9 = ARRAY
    # Array of type 4 (UINT32) with 2,000,000 elements (>65536 limit)
    buf.write(struct.pack("<IQ", 4, 2_000_000))
    p = tmp_path / "array_bomb.gguf"
    p.write_bytes(buf.getvalue())
    with pytest.raises(GGUFParseError, match="Array length .* exceeds safety limit"):
        GGUFParser.read_manifest(p)


def test_gguf_adversarial_truncated_file(tmp_path: Path):
    """Rejects prematurely truncated files safely with typed error."""
    buf = io.BytesIO()
    buf.write(struct.pack("<IIQQ", GGUF_MAGIC, 3, 10, 5))
    key_b = b"general.architecture"
    buf.write(struct.pack("<Q", len(key_b)) + key_b)
    # File ends abruptly before value type or payload
    p = tmp_path / "truncated.gguf"
    p.write_bytes(buf.getvalue())
    with pytest.raises(GGUFParseError, match="Unexpected EOF"):
        GGUFParser.read_manifest(p)


def test_safetensors_synthetic_with_config(tmp_path: Path):
    """Verifies that SafeTensors container merges parameters from companion config.json."""
    tensors = {
        "model.layers.0.weight": {
            "dtype": "BF16",
            "shape": [4096, 4096],
            "data_offsets": [0, 33554432],
        },
        "__metadata__": {"license": "MIT"},
    }
    json_bytes = json.dumps(tensors).encode("utf-8")
    st_bytes = struct.pack("<Q", len(json_bytes)) + json_bytes + (b"\x00" * 1024)

    st_file = tmp_path / "model.safetensors"
    st_file.write_bytes(st_bytes)

    # Write companion config.json
    config_data = {
        "model_type": "llama",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
    }
    (tmp_path / "config.json").write_text(json.dumps(config_data), encoding="utf-8")

    manifest = SafeTensorsParser.read_manifest(st_file)

    assert manifest.format == ModelFormat.SAFETENSORS
    assert manifest.tensor_count == 1
    assert manifest.quantization_type == "BF16"
    assert manifest.license_str == "MIT"

    hp = manifest.hyperparameters
    assert hp is not None
    assert hp.architecture == ModelArchitecture.LLAMA
    assert hp.layer_count == 32
    assert hp.embedding_length == 4096
    assert hp.head_count == 32
    assert hp.head_count_kv == 8
    assert hp.context_length == 8192


def test_safetensors_without_companion_config(tmp_path: Path):
    """Verifies that standalone SafeTensors without config.json marks hyperparameters as None."""
    tensors = {
        "weight.0": {"dtype": "F16", "shape": [10, 10], "data_offsets": [0, 200]},
    }
    json_bytes = json.dumps(tensors).encode("utf-8")
    st_bytes = struct.pack("<Q", len(json_bytes)) + json_bytes

    st_file = tmp_path / "isolated.safetensors"
    st_file.write_bytes(st_bytes)

    manifest = SafeTensorsParser.read_manifest(st_file)
    assert manifest.format == ModelFormat.SAFETENSORS
    assert manifest.hyperparameters is None
    assert any("No companion config.json" in note for note in manifest.provenance_notes)


def test_safetensors_adversarial_oversized_header(tmp_path: Path):
    """Rejects SafeTensors files with header length exceeding 16 MB."""
    # Header length claimed = 32 MB
    p = tmp_path / "giant_header.safetensors"
    p.write_bytes(struct.pack("<Q", 32 * 1024 * 1024) + b"{}")
    with pytest.raises(SafeTensorsParseError, match="exceeds safety limit"):
        SafeTensorsParser.read_manifest(p)


def test_safetensors_adversarial_corrupted_json(tmp_path: Path):
    """Rejects malformed JSON payloads gracefully."""
    bad_json = b"{ not valid json ... "
    p = tmp_path / "bad_json.safetensors"
    p.write_bytes(struct.pack("<Q", len(bad_json)) + bad_json)
    with pytest.raises(SafeTensorsParseError, match="Malformed SafeTensors JSON"):
        SafeTensorsParser.read_manifest(p)


def test_safetensors_file_smaller_than_8_bytes(tmp_path: Path):
    """Rejects SafeTensors files smaller than 8 bytes."""
    p = tmp_path / "tiny.safetensors"
    p.write_bytes(b"1234")
    with pytest.raises(SafeTensorsParseError, match="File too small"):
        SafeTensorsParser.read_manifest(p)


def test_safetensors_header_larger_than_file_size(tmp_path: Path):
    """Rejects SafeTensors files where claimed header length exceeds file size."""
    p = tmp_path / "exceeds_size.safetensors"
    # Claim 1000 bytes header, but file only has 8 + 10 bytes
    p.write_bytes(struct.pack("<Q", 1000) + b"0123456789")
    with pytest.raises(
        SafeTensorsParseError,
        match="SafeTensors header length exceeds actual file size",
    ):
        SafeTensorsParser.read_manifest(p)


def test_safetensors_nonexistent_file():
    """Verifies FileNotFoundError on missing SafeTensors file."""
    with pytest.raises(FileNotFoundError):
        SafeTensorsParser.read_manifest("nonexistent/path/model.safetensors")


def test_gguf_all_scalar_and_array_types(tmp_path: Path):
    """Verifies that GGUF parser handles all scalar data types and array types."""
    buf = io.BytesIO()
    buf.write(struct.pack("<I", GGUF_MAGIC))
    buf.write(struct.pack("<I", 3))  # Version 3
    buf.write(struct.pack("<Q", 0))  # 0 tensors

    # Pack 11 different KV metadata pairs covering all value types
    kv_items = [
        ("general.architecture", 8, struct.pack("<Q", 5) + b"llama"),
        ("test.u8", 0, struct.pack("<B", 255)),
        ("test.i8", 1, struct.pack("<b", -128)),
        ("test.u16", 2, struct.pack("<H", 65535)),
        ("test.i16", 3, struct.pack("<h", -32768)),
        ("test.i32", 5, struct.pack("<i", -1000)),
        ("test.f32", 6, struct.pack("<f", 3.14)),
        ("test.bool", 7, struct.pack("<?", True)),
        ("test.u64", 10, struct.pack("<Q", 18446744073709551615)),
        ("test.i64", 11, struct.pack("<q", -9223372036854775808)),
        ("test.f64", 12, struct.pack("<d", 2.718281828)),
    ]

    buf.write(struct.pack("<Q", len(kv_items)))
    for key, val_type, payload in kv_items:
        k_bytes = key.encode("utf-8")
        buf.write(struct.pack("<Q", len(k_bytes)))
        buf.write(k_bytes)
        buf.write(struct.pack("<I", val_type))
        buf.write(payload)

    p = tmp_path / "all_types.gguf"
    p.write_bytes(buf.getvalue())

    manifest = GGUFParser.read_manifest(p)
    assert manifest.format == ModelFormat.GGUF
    assert manifest.model_id == "all_types"


def test_gguf_unsupported_value_type(tmp_path: Path):
    """Verifies that an unsupported GGUF value type ID triggers GGUFParseError."""
    buf = io.BytesIO()
    buf.write(struct.pack("<I", GGUF_MAGIC))
    buf.write(struct.pack("<I", 3))
    buf.write(struct.pack("<Q", 0))
    buf.write(struct.pack("<Q", 1))

    key = b"bad.type"
    buf.write(struct.pack("<Q", len(key)))
    buf.write(key)
    buf.write(struct.pack("<I", 99))  # Unsupported type 99

    p = tmp_path / "bad_type.gguf"
    p.write_bytes(buf.getvalue())

    with pytest.raises(GGUFParseError, match="Unknown GGUF value type: 99"):
        GGUFParser.read_manifest(p)


def test_gguf_architecture_normalization():
    """Verifies architecture string normalization across known open models."""
    assert GGUFParser._normalize_architecture("llama") == ModelArchitecture.LLAMA
    assert GGUFParser._normalize_architecture("qwen2") == ModelArchitecture.QWEN2
    assert GGUFParser._normalize_architecture("mistral") == ModelArchitecture.MISTRAL
    assert GGUFParser._normalize_architecture("gemma2") == ModelArchitecture.GEMMA2
    assert GGUFParser._normalize_architecture("deepseek2") == ModelArchitecture.DEEPSEEK
    assert (
        GGUFParser._normalize_architecture("custom_transformer")
        == ModelArchitecture.UNKNOWN
    )


def test_gguf_quantization_fallback():
    """Verifies quantization determination from filename suffix when general.file_type is missing."""
    assert GGUFParser._determine_quantization({}, "model-q4_k_m.gguf") == "Q4_K_M"
    assert GGUFParser._determine_quantization({}, "model-q8_0.gguf") == "Q8_0"
    assert GGUFParser._determine_quantization({}, "model-f16.gguf") == "F16"
    assert GGUFParser._determine_quantization({}, "unannotated_model.gguf") == "UNKNOWN"

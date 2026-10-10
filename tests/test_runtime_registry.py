"""Unit Tests for Model Discovery, Registry, and Safe Checksumming."""

import hashlib
import json
import struct
from pathlib import Path

from schemasense.runtime.registry import ModelRegistry
from schemasense.runtime.types import ModelFormat


def build_minimal_gguf(path: Path) -> Path:
    """Helper creating a minimal valid GGUF file."""
    data = bytearray()
    data.extend(b"GGUF")
    data.extend(struct.pack("<I", 3))  # version 3
    data.extend(struct.pack("<Q", 0))  # 0 tensors
    data.extend(struct.pack("<Q", 1))  # 1 metadata KV

    # KV: general.architecture = "llama"
    key = b"general.architecture"
    data.extend(struct.pack("<Q", len(key)))
    data.extend(key)
    data.extend(struct.pack("<I", 8))  # GGUF_TYPE_STRING
    val = b"llama"
    data.extend(struct.pack("<Q", len(val)))
    data.extend(val)

    path.write_bytes(bytes(data))
    return path


def build_minimal_safetensors(path: Path) -> Path:
    """Helper creating a minimal valid SafeTensors file."""
    header = {
        "__metadata__": {"format": "pt"},
        "weight": {"dtype": "F16", "shape": [10, 10], "data_offsets": [0, 200]},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    data = bytearray()
    data.extend(struct.pack("<Q", len(header_bytes)))
    data.extend(header_bytes)
    data.extend(b"\x00" * 200)

    path.write_bytes(bytes(data))
    return path


def test_registry_discover_valid_models(tmp_path: Path):
    """Verifies scanning finds GGUF and SafeTensors models."""
    build_minimal_gguf(tmp_path / "model1.gguf")
    build_minimal_safetensors(tmp_path / "model2.safetensors")

    # Add an unrelated text file
    (tmp_path / "notes.txt").write_text("just notes")

    registry = ModelRegistry(model_dirs=[tmp_path])
    manifests = registry.discover()

    assert len(manifests) == 2
    formats = {m.format for m in manifests}
    assert ModelFormat.GGUF in formats
    assert ModelFormat.SAFETENSORS in formats

    # Verify checksums are unverified by default during discovery (mandatory safeguard)
    for m in manifests:
        assert m.checksum_verified is False
        assert m.sha256_checksum is None


def test_registry_unsafe_format_rejection(tmp_path: Path):
    """Verifies that PyTorch pickle (.pt, .bin, .pkl) files trigger security rejection."""
    build_minimal_gguf(tmp_path / "safe.gguf")

    unsafe_pt = tmp_path / "weights.pt"
    unsafe_pt.write_bytes(b"PK\x03\x04fake_pytorch_zip")

    unsafe_pkl = tmp_path / "weights.pkl"
    unsafe_pkl.write_bytes(b"\x80\x04\x95fake_pickle")

    registry = ModelRegistry(model_dirs=[tmp_path])
    manifests = registry.discover()

    # Only safe model should be registered in discovery
    assert len(manifests) == 1
    assert manifests[0].format == ModelFormat.GGUF

    # Verify security alerts were logged for unsafe formats
    alerts = registry.get_security_alerts()
    assert len(alerts) >= 2
    alert_text = " ".join(alerts)
    assert "weights.pt" in alert_text
    assert "weights.pkl" in alert_text
    assert "Arbitrary code execution risk" in alert_text


def test_registry_compute_checksum_on_demand(tmp_path: Path):
    """Verifies on-demand chunked SHA-256 checksum calculation."""
    gguf_path = build_minimal_gguf(tmp_path / "tiny.gguf")
    expected_hash = hashlib.sha256(gguf_path.read_bytes()).hexdigest()

    registry = ModelRegistry(model_dirs=[tmp_path])
    manifests = registry.discover()
    manifest = manifests[0]
    assert manifest.checksum_verified is False

    computed_hash = registry.compute_checksum(manifest)
    assert computed_hash == expected_hash
    assert manifest.sha256_checksum == expected_hash
    assert manifest.checksum_verified is True


def test_registry_compute_checksum_cancellation(tmp_path: Path):
    """Verifies that the cancellation predicate safely aborts hashing."""
    build_minimal_gguf(tmp_path / "cancel_test.gguf")

    registry = ModelRegistry(model_dirs=[tmp_path])
    manifests = registry.discover()
    manifest = manifests[0]

    # Immediately cancel via predicate
    result = registry.compute_checksum(manifest, is_cancelled=lambda: True)
    assert result is None
    assert manifest.checksum_verified is False


def test_registry_verify_checksum_match_and_mismatch(tmp_path: Path):
    """Verifies checksum matching against an expected hash."""
    gguf_path = build_minimal_gguf(tmp_path / "verify.gguf")
    correct_hash = hashlib.sha256(gguf_path.read_bytes()).hexdigest()

    registry = ModelRegistry(model_dirs=[tmp_path])
    manifests = registry.discover()
    manifest = manifests[0]

    # 1. Matching hash
    is_valid = registry.verify_checksum(manifest, expected_sha256=correct_hash)
    assert is_valid is True
    assert manifest.checksum_verified is True

    # 2. Mismatching hash
    wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    is_valid_wrong = registry.verify_checksum(manifest, expected_sha256=wrong_hash)
    assert is_valid_wrong is False
    assert manifest.checksum_verified is False


def test_registry_symlink_escape_rejection(tmp_path: Path):
    """Verifies that symlinks pointing outside the scan directory are rejected with a security alert."""
    from unittest.mock import MagicMock, patch

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    target_file = secret_dir / "escaped.gguf"
    build_minimal_gguf(target_file)

    symlink_file = models_dir / "link.gguf"
    try:
        symlink_file.symlink_to(target_file)
        has_real_symlink = True
    except OSError:
        has_real_symlink = False

    registry = ModelRegistry(model_dirs=[models_dir])
    if has_real_symlink:
        manifests = registry.discover()
        assert len(manifests) == 0
        alerts = registry.get_security_alerts()
        assert any("pointing outside target directory" in a for a in alerts)
    else:
        mock_item = MagicMock()
        mock_item.is_file.return_value = True
        mock_item.is_symlink.return_value = True
        mock_item.resolve.return_value = Path("/outside/target/file.gguf")
        mock_item.name = "escaped_link.gguf"
        with patch.object(Path, "rglob", return_value=[mock_item]):
            manifests = registry.discover()
            assert len(manifests) == 0
            alerts = registry.get_security_alerts()
            assert any("pointing outside target directory" in a for a in alerts)


def test_registry_same_stem_different_subdirectories_no_collision(tmp_path: Path):
    """Verifies that models with the same stem in different subdirectories have unique IDs and do not overwrite each other."""
    sub_q4 = tmp_path / "q4"
    sub_q4.mkdir()
    file_q4 = build_minimal_gguf(sub_q4 / "model.gguf")

    sub_q8 = tmp_path / "q8"
    sub_q8.mkdir()
    file_q8 = build_minimal_gguf(sub_q8 / "model.gguf")

    registry = ModelRegistry(model_dirs=[tmp_path])
    manifests = registry.discover()

    assert len(manifests) == 2
    assert len(registry.list_models()) == 2

    id_set = {m.model_id for m in manifests}
    assert len(id_set) == 2
    assert "q4/model" in id_set
    assert "q8/model" in id_set

    # Verify original filename and display_name are preserved
    for m in manifests:
        assert m.display_name == "model"
        assert m.filename == "model.gguf"

    # Both models can be retrieved independently without collision
    m_q4 = registry.get_model("q4/model")
    m_q8 = registry.get_model("q8/model")
    assert m_q4 is not None and m_q8 is not None
    assert m_q4.file_path == file_q4
    assert m_q8.file_path == file_q8


def test_registry_repeated_scans_produce_stable_ids(tmp_path: Path):
    """Verifies that scanning the same directory repeatedly produces identical, deterministic model IDs."""
    sub = tmp_path / "nested" / "v1"
    sub.mkdir(parents=True)
    build_minimal_gguf(sub / "model.gguf")

    registry1 = ModelRegistry(model_dirs=[tmp_path])
    manifests1 = registry1.discover()
    id1 = manifests1[0].model_id

    registry2 = ModelRegistry(model_dirs=[tmp_path])
    manifests2 = registry2.discover()
    id2 = manifests2[0].model_id

    assert id1 == id2 == "nested/v1/model"


def test_registry_build_model_id_path_traversal_and_absolute_path_safety(
    tmp_path: Path,
):
    """Verifies build_model_id strips traversal characters and does not expose absolute filesystem paths."""
    scan_root = tmp_path / "models"
    scan_root.mkdir()
    target_file = scan_root / "subdir" / "target.gguf"

    model_id = ModelRegistry.build_model_id(target_file, scan_root=scan_root)
    assert model_id == "subdir/target"
    # Never includes absolute system drive or root
    assert str(tmp_path) not in model_id
    assert ":" not in model_id  # No Windows drive letters
    assert ".." not in model_id  # No traversal

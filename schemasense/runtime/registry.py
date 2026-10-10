"""SchemaSense Local AI Runtime — Model Discovery, Registry, and Checksumming."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

from schemasense.runtime.parsers.gguf import GGUFParseError, GGUFParser
from schemasense.runtime.parsers.safetensors import (
    SafeTensorsParseError,
    SafeTensorsParser,
)
from schemasense.runtime.types import ModelManifest

logger = logging.getLogger("schemasense.runtime.registry")

UNSAFE_EXTENSIONS = {".pt", ".bin", ".pkl", ".pickle", ".pth"}
HASH_CHUNK_BYTES = 64 * 1024  # 64 KB streaming buffer for checksumming


class ModelRegistry:
    """Discovers local models on disk, builds manifests, and provides explicit chunked hashing."""

    def __init__(self, model_dirs: list[Path | str] | None = None) -> None:
        self.model_dirs = [Path(d) for d in model_dirs] if model_dirs else []
        self._models: dict[str, ModelManifest] = {}
        self._security_alerts: list[str] = []

    def get_security_alerts(self) -> list[str]:
        """Returns security warnings generated during discovery (e.g. unsafe pickle formats)."""
        return list(self._security_alerts)

    def discover(self, compute_checksums: bool = False) -> list[ModelManifest]:
        """Scans all configured model directories."""
        all_manifests: list[ModelManifest] = []
        for d in self.model_dirs:
            all_manifests.extend(
                self.discover_directory(d, compute_checksums=compute_checksums)
            )
        return all_manifests

    def discover_directory(
        self, directory: Path | str, compute_checksums: bool = False
    ) -> list[ModelManifest]:
        """Scans a local directory for GGUF and SafeTensors model containers.

        Does NOT automatically hash multi-gigabyte files by default to preserve responsiveness.
        """
        p = Path(directory).resolve()
        if not p.is_dir():
            logger.warning("Models directory does not exist: %s", p)
            return []

        discovered: list[ModelManifest] = []

        for item in p.rglob("*"):
            if not item.is_file():
                continue

            # Security defense: reject symlinks escaping outside the target directory
            if item.is_symlink():
                try:
                    resolved_target = item.resolve()
                    if not resolved_target.is_relative_to(p):
                        alert = f"SECURITY: Skipping symlink '{item.name}' pointing outside target directory ({resolved_target})."
                        logger.warning(alert)
                        self._security_alerts.append(alert)
                        continue
                except (ValueError, OSError):
                    continue

            suffix = item.suffix.lower()

            # Security defense: reject dangerous executable pickle files
            if suffix in UNSAFE_EXTENSIONS:
                alert = (
                    f"SECURITY: Skipping unsafe model file '{item.name}' with executable pickle format ({suffix}). "
                    "Arbitrary code execution risk."
                )
                logger.warning(alert)
                self._security_alerts.append(alert)
                continue

            manifest: ModelManifest | None = None

            if suffix == ".gguf":
                try:
                    manifest = GGUFParser.read_manifest(item)
                except (GGUFParseError, ValueError, OSError) as e:
                    logger.warning(
                        "Failed to parse GGUF header for '%s': %s", item.name, e
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Unexpected error parsing GGUF header for '%s': %s",
                        item.name,
                        e,
                    )

            elif suffix == ".safetensors":
                try:
                    manifest = SafeTensorsParser.read_manifest(item)
                except (SafeTensorsParseError, ValueError, OSError) as e:
                    logger.warning(
                        "Failed to parse SafeTensors header for '%s': %s", item.name, e
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Unexpected error parsing SafeTensors header for '%s': %s",
                        item.name,
                        e,
                    )

            if manifest is not None:
                # Assign collision-resistant relative ID while preserving display name and filename
                manifest.model_id = self.build_model_id(item, scan_root=p)
                manifest.display_name = item.stem
                manifest.filename = item.name

                if compute_checksums:
                    try:
                        self.compute_checksum(manifest)
                    except (OSError, ValueError) as e:
                        logger.warning(
                            "Checksum computation failed for '%s': %s", item.name, e
                        )

                self._models[manifest.model_id] = manifest
                discovered.append(manifest)

        return discovered

    @classmethod
    def build_model_id(
        cls, file_path: Path | str, scan_root: Path | str | None = None
    ) -> str:
        """Derives a deterministic, collision-resistant model ID relative to the scan root.

        Prevents directory traversal, preserves stability across repeated scans,
        and avoids exposing unnecessary absolute-path information.
        """
        p = Path(file_path).resolve()
        if scan_root is not None:
            root = Path(scan_root).resolve()
            try:
                rel = p.relative_to(root)
                parts = list(rel.parts)
                if parts:
                    # Strip suffix from leaf filename while preserving subdirectory namespace
                    parts[-1] = Path(parts[-1]).stem
                    # Sanitize traversal characters
                    clean_parts = [
                        part.replace("..", "")
                        .replace("/", "")
                        .replace("\\", "")
                        .strip()
                        for part in parts
                        if part
                    ]
                    if clean_parts:
                        return "/".join(clean_parts)
            except ValueError:
                pass
        return p.stem

    def get_model(self, model_id: str) -> ModelManifest | None:
        """Retrieves a registered model manifest by id."""
        return self._models.get(model_id)

    def list_models(self) -> list[ModelManifest]:
        """Returns all currently registered models."""
        return list(self._models.values())

    @classmethod
    def compute_checksum(
        cls, manifest: ModelManifest, is_cancelled: Callable[[], bool] | None = None
    ) -> str | None:
        """Computes SHA-256 checksum in small chunked reads without loading files into memory.

        Supports cancellation via an optional predicate. Returns None if cancelled.
        """
        p = manifest.file_path
        if not p.is_file():
            raise FileNotFoundError(f"File not found for checksum: {p}")

        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                if is_cancelled and is_cancelled():
                    logger.info("Checksumming cancelled for %s", manifest.model_id)
                    return None

                chunk = f.read(HASH_CHUNK_BYTES)
                if not chunk:
                    break
                hasher.update(chunk)

        checksum = hasher.hexdigest()
        manifest.sha256_checksum = checksum
        manifest.checksum_verified = True
        return checksum

    @classmethod
    def verify_checksum(
        cls,
        manifest: ModelManifest,
        expected_sha256: str,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        """Verifies model file against an expected SHA-256 hash."""
        computed = cls.compute_checksum(manifest, is_cancelled=is_cancelled)
        if computed is None:
            manifest.checksum_verified = False
            return False
        is_match = computed.lower() == expected_sha256.lower()
        manifest.checksum_verified = is_match
        return is_match

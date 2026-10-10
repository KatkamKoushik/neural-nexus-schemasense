"""SchemaSense Local AI Runtime Management Package."""

from schemasense.runtime.compatibility import ModelCompatibilityCalculator
from schemasense.runtime.hardware import HardwareInspector
from schemasense.runtime.parsers.gguf import GGUFParseError, GGUFParser
from schemasense.runtime.parsers.safetensors import (
    SafeTensorsParseError,
    SafeTensorsParser,
)
from schemasense.runtime.registry import ModelRegistry
from schemasense.runtime.types import (
    BackendAccelerator,
    CompatibilityStatus,
    ConfidenceLevel,
    GPUInfo,
    HardwareProfile,
    ModelArchitecture,
    ModelFormat,
    ModelHyperparameters,
    ModelManifest,
    SizingReport,
)

__all__ = [
    "BackendAccelerator",
    "CompatibilityStatus",
    "ConfidenceLevel",
    "GGUFParseError",
    "GGUFParser",
    "GPUInfo",
    "HardwareInspector",
    "HardwareProfile",
    "ModelArchitecture",
    "ModelCompatibilityCalculator",
    "ModelFormat",
    "ModelHyperparameters",
    "ModelManifest",
    "ModelRegistry",
    "SafeTensorsParseError",
    "SafeTensorsParser",
    "SizingReport",
]

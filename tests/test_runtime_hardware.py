"""Unit Tests for Cross-Platform Hardware Inspection."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from schemasense.runtime.hardware import HardwareInspector
from schemasense.runtime.types import BackendAccelerator


def test_hardware_inspector_live_host_smoke():
    """Verifies that HardwareInspector runs cleanly on the current host without crashing."""
    profile = HardwareInspector.get_profile()
    assert profile.total_ram_mb > 0
    assert profile.available_ram_mb > 0
    assert profile.cpu_cores_logical >= 1
    assert profile.cpu_arch != ""


def test_linux_meminfo_parser(tmp_path: Path):
    """Verifies /proc/meminfo parsing with mock Linux memory stats."""
    meminfo_content = (
        "MemTotal:       32840960 kB\n"
        "MemFree:         4194304 kB\n"
        "MemAvailable:   24576000 kB\n"
        "SwapTotal:       8388608 kB\n"
        "SwapFree:        6291456 kB\n"
    )
    fake_meminfo = tmp_path / "meminfo"
    fake_meminfo.write_text(meminfo_content, encoding="utf-8")

    with (
        patch("sys.platform", "linux"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", MagicMock(return_value=io_wrapper(meminfo_content))),
    ):
        tot, av, sw_tot, sw_av = HardwareInspector._inspect_memory()
        # 32840960 kB // 1024 = 32071 MB
        assert abs(tot - 32071) <= 1
        # 24576000 kB // 1024 = 24000 MB
        assert abs(av - 24000) <= 1
        assert abs(sw_tot - 8192) <= 1
        assert abs(sw_av - 6144) <= 1


def io_wrapper(text: str):
    import io

    return io.StringIO(text)


def test_wsl_detection_and_warning(tmp_path: Path):
    """Verifies WSL detection and warning trigger when storage is on /mnt/ Windows mount."""
    with (
        patch("sys.platform", "linux"),
        patch("os.path.exists", return_value=True),
        patch(
            "builtins.open",
            MagicMock(
                side_effect=lambda *a, **kw: io_wrapper(
                    "Linux version 5.15.153.1-microsoft-standard-WSL2"
                )
            ),
        ),
    ):
        # 1. Path in native linux home
        is_wsl, warn1 = HardwareInspector._inspect_wsl(Path("/home/user/models"))
        assert is_wsl is True
        assert warn1 is None

        # 2. Path on Windows mounted drive /mnt/c
        is_wsl, warn2 = HardwareInspector._inspect_wsl(Path("/mnt/c/Users/user/models"))
        assert is_wsl is True
        assert warn2 is not None
        assert "9p mount" in warn2


def test_nvidia_smi_success():
    """Verifies parsing of successful nvidia-smi CSV output."""
    mock_csv = "NVIDIA GeForce RTX 4090, 24576, 18432\n"
    with patch("subprocess.check_output", return_value=mock_csv):
        gpus = HardwareInspector._inspect_gpus()
        assert len(gpus) == 1
        gpu = gpus[0]
        assert gpu.accelerator == BackendAccelerator.CUDA
        assert "RTX 4090" in gpu.name
        assert gpu.total_vram_mb == 24576
        assert gpu.available_vram_mb == 18432
        assert gpu.is_unified_memory is False


def test_nvidia_smi_timeout_or_missing():
    """Verifies that missing or hanging nvidia-smi fails gracefully without raising."""
    with patch(
        "subprocess.check_output",
        side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=1.5),
    ):
        # Should catch timeout and return empty list or fallback
        gpus = HardwareInspector._inspect_gpus()
        assert isinstance(gpus, list)

    with patch("subprocess.check_output", side_effect=FileNotFoundError):
        gpus = HardwareInspector._inspect_gpus()
        assert isinstance(gpus, list)


def test_macos_apple_silicon_detection():
    """Verifies macOS Apple Silicon unified memory reporting."""
    with (
        patch("sys.platform", "darwin"),
        patch("platform.machine", return_value="arm64"),
        patch(
            "schemasense.runtime.hardware.HardwareInspector._inspect_memory",
            return_value=(32768, 24000, None, None),
        ),
    ):
        gpus = HardwareInspector._inspect_gpus()
        assert len(gpus) == 1
        gpu = gpus[0]
        assert gpu.accelerator == BackendAccelerator.METAL
        assert gpu.is_unified_memory is True
        assert gpu.total_vram_mb is None  # Accurately documented as shared
        assert "Unified memory" in gpu.details

"""SchemaSense Local AI Runtime — Cross-Platform Hardware Inspection."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from schemasense.runtime.types import BackendAccelerator, GPUInfo, HardwareProfile

logger = logging.getLogger("schemasense.runtime.hardware")

PROBE_TIMEOUT_SECONDS = 1.5


class HardwareInspector:
    """Discovers host resources and hardware capabilities across Windows, Linux, and macOS."""

    @classmethod
    def get_profile(cls, target_dir: Path | str | None = None) -> HardwareProfile:
        """Collects complete hardware profile with failure-tolerant probes."""
        total_ram, avail_ram, total_swap, avail_swap = cls._inspect_memory()
        cpu_phys, cpu_log, arch = cls._inspect_cpu()
        is_wsl, wsl_warn = cls._inspect_wsl(target_dir)
        gpus = cls._inspect_gpus()
        disk_total, disk_free = cls._inspect_storage(target_dir)

        return HardwareProfile(
            total_ram_mb=total_ram,
            available_ram_mb=avail_ram,
            total_swap_mb=total_swap,
            available_swap_mb=avail_swap,
            cpu_cores_physical=cpu_phys,
            cpu_cores_logical=cpu_log,
            cpu_arch=arch,
            is_wsl=is_wsl,
            wsl_warning=wsl_warn,
            gpus=gpus,
            storage_total_gb=disk_total,
            storage_free_gb=disk_free,
        )

    @classmethod
    def _inspect_memory(cls) -> tuple[int, int, int | None, int | None]:
        """Probes host physical memory and swap/pagefile in megabytes."""
        # 1. Windows native probe via GlobalMemoryStatusEx
        if sys.platform == "win32":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    tot_ram = int(stat.ullTotalPhys // (1024 * 1024))
                    av_ram = int(stat.ullAvailPhys // (1024 * 1024))
                    tot_swap = int(stat.ullTotalPageFile // (1024 * 1024))
                    av_swap = int(stat.ullAvailPageFile // (1024 * 1024))
                    return tot_ram, av_ram, tot_swap, av_swap
            except (AttributeError, OSError, ValueError) as e:
                logger.debug("Windows native memory probe failed: %s; falling back", e)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Unexpected error in Windows memory probe: %s; falling back", e
                )

        # 2. Linux native probe via /proc/meminfo
        if sys.platform.startswith("linux") and os.path.exists("/proc/meminfo"):
            try:
                mem_data: dict[str, int] = {}
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            val = parts[1].strip().split()[0]
                            mem_data[parts[0].strip()] = int(val) // 1024  # kB to MB

                tot_ram = mem_data.get("MemTotal", 0)
                av_ram = mem_data.get("MemAvailable", mem_data.get("MemFree", 0))
                tot_swap = mem_data.get("SwapTotal")
                av_swap = mem_data.get("SwapFree")
                if tot_ram > 0:
                    return tot_ram, av_ram, tot_swap, av_swap
            except (OSError, ValueError) as e:
                logger.debug("Linux /proc/meminfo probe failed: %s; falling back", e)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Unexpected error in Linux /proc/meminfo probe: %s; falling back", e
                )

        # 3. macOS native probe via sysctl
        if sys.platform == "darwin":
            try:
                tot_bytes = int(
                    subprocess.check_output(
                        ["sysctl", "-n", "hw.memsize"],
                        stderr=subprocess.DEVNULL,
                        timeout=PROBE_TIMEOUT_SECONDS,
                    )
                )
                tot_ram = tot_bytes // (1024 * 1024)
                # psutil or fallback conservative estimation for macOS available memory
                try:
                    import psutil

                    av_ram = int(psutil.virtual_memory().available // (1024 * 1024))
                except ImportError:
                    av_ram = int(tot_ram * 0.50)  # Conservative fallback
                return tot_ram, av_ram, None, None
            except (subprocess.SubprocessError, OSError, ValueError) as e:
                logger.debug("macOS memory probe failed: %s; falling back", e)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Unexpected error in macOS memory probe: %s; falling back", e
                )

        # 4. Universal psutil fallback if installed
        try:
            import psutil

            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            return (
                int(vm.total // (1024 * 1024)),
                int(vm.available // (1024 * 1024)),
                int(sm.total // (1024 * 1024)),
                int(sm.free // (1024 * 1024)),
            )
        except Exception:  # noqa: BLE001
            # Minimal emergency return if all probes fail
            return 8192, 4096, None, None

    @classmethod
    def _inspect_cpu(cls) -> tuple[int | None, int, str]:
        """Inspects CPU physical and logical cores and architecture."""
        log_cores = os.cpu_count() or 1
        phys_cores = None
        try:
            import psutil

            phys_cores = psutil.cpu_count(logical=False)
        except (ImportError, AttributeError, OSError):
            pass

        arch = platform.machine().lower() or "unknown"
        return phys_cores, log_cores, arch

    @classmethod
    def _inspect_wsl(cls, target_dir: Path | str | None) -> tuple[bool, str | None]:
        """Detects Windows Subsystem for Linux and verifies storage mount performance."""
        if not sys.platform.startswith("linux"):
            return False, None

        is_wsl = False
        try:
            if os.path.exists("/proc/version"):
                with open("/proc/version", "r", encoding="utf-8") as f:
                    ver = f.read().lower()
                    if "microsoft" in ver or "wsl" in ver:
                        is_wsl = True
        except OSError:
            pass

        wsl_warning = None
        if is_wsl and target_dir:
            posix_path = str(target_dir).replace("\\", "/")
            if posix_path.startswith("/mnt/"):
                wsl_warning = (
                    f"Directory '{posix_path}' is on a Windows 9p mount (/mnt/). "
                    "Disk throughput will be severely degraded; use native Linux filesystem (~/) instead."
                )

        return is_wsl, wsl_warning

    @classmethod
    def _inspect_gpus(cls) -> list[GPUInfo]:
        """Attempts safe, bounded GPU discovery without crashing or blocking."""
        gpus: list[GPUInfo] = []

        # Probe 1: Apple Silicon Metal (macOS ARM64)
        if sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64"):
            _total_ram_mb, avail_ram_mb, _, _ = cls._inspect_memory()
            gpus.append(
                GPUInfo(
                    name="Apple Silicon Integrated GPU",
                    accelerator=BackendAccelerator.METAL,
                    total_vram_mb=None,  # Reported as None because unified memory dynamically shares RAM
                    available_vram_mb=None,
                    is_unified_memory=True,
                    details=f"Unified memory architecture; shares up to {avail_ram_mb}MB available host RAM.",
                )
            )
            return gpus

        # Probe 2: NVIDIA via nvidia-smi
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
            ).strip()
            if out:
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        name = parts[0]
                        total = int(parts[1]) if parts[1].isdigit() else None
                        free = int(parts[2]) if parts[2].isdigit() else None
                        gpus.append(
                            GPUInfo(
                                name=name,
                                accelerator=BackendAccelerator.CUDA,
                                total_vram_mb=total,
                                available_vram_mb=free,
                                is_unified_memory=False,
                                details="NVIDIA GPU detected via nvidia-smi",
                            )
                        )
                if gpus:
                    return gpus
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.debug("nvidia-smi query unavailable: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.debug("Unexpected error querying nvidia-smi: %s", e)

        # Probe 3: Windows DXGI query for discrete adapter (CTypes, non-invasive)
        if sys.platform == "win32":
            try:
                # If DXGI is available, query display adapters
                dxgi_gpu = cls._probe_windows_dxgi()
                if dxgi_gpu:
                    gpus.append(dxgi_gpu)
                    return gpus
            except (subprocess.SubprocessError, OSError) as e:
                logger.debug("Windows DXGI probe failed: %s", e)
            except Exception as e:  # noqa: BLE001
                logger.debug("Unexpected error in Windows DXGI probe: %s", e)

        return gpus

    @classmethod
    def _probe_windows_dxgi(cls) -> GPUInfo | None:
        """Best-effort Windows DXGI adapter query via PowerShell (bounded timeout)."""
        try:
            cmd = (
                "Get-CimInstance Win32_VideoController | "
                "Select-Object -First 1 Name, AdapterRAM | "
                "ConvertTo-Json"
            )
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", cmd],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
            ).strip()
            if out:
                import json

                data = json.loads(out)
                name = data.get("Name", "Generic Display Adapter")
                raw_bytes = data.get("AdapterRAM", 0)
                vram_mb = int(raw_bytes // (1024 * 1024)) if raw_bytes else None
                return GPUInfo(
                    name=name,
                    accelerator=BackendAccelerator.DXGI,
                    total_vram_mb=vram_mb if vram_mb and vram_mb > 0 else None,
                    available_vram_mb=None,  # Windows does not expose free VRAM via WMI
                    is_unified_memory=False,
                    details="Queried via Windows CIM video controller interface",
                )
        except (subprocess.SubprocessError, OSError, ValueError):
            return None
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def _inspect_storage(
        cls, target_dir: Path | str | None
    ) -> tuple[float | None, float | None]:
        """Inspects disk capacity in gigabytes."""
        try:
            p = Path(target_dir).resolve() if target_dir else Path.cwd()
            usage = shutil.disk_usage(p)
            total_gb = round(usage.total / (1024**3), 2)
            free_gb = round(usage.free / (1024**3), 2)
            return total_gb, free_gb
        except OSError:
            return None, None
        except Exception:  # noqa: BLE001
            return None, None

"""NexusRAG Hardware Auto-Detector and Device Selector.

Detects CPU, RAM, Apple Silicon (MPS / Metal), and NVIDIA CUDA VRAM to recommend
appropriate model quantizations and runtime devices.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HardwareProfile:
    os_name: str
    architecture: str
    total_ram_gb: float
    available_ram_gb: float
    is_apple_silicon: bool
    apple_chip_name: Optional[str]
    has_cuda: bool
    cuda_device_name: Optional[str]
    cuda_vram_gb: float
    recommended_device: str  # "mps", "cuda", or "cpu"
    recommended_profile: str  # "balanced", "deep_reasoning", "speed_demon"

    def summary(self) -> Dict[str, Any]:
        return {
            "os": f"{self.os_name} ({self.architecture})",
            "total_ram_gb": round(self.total_ram_gb, 1),
            "available_ram_gb": round(self.available_ram_gb, 1),
            "apple_silicon": self.is_apple_silicon,
            "apple_chip": self.apple_chip_name,
            "cuda_available": self.has_cuda,
            "cuda_device": self.cuda_device_name,
            "cuda_vram_gb": round(self.cuda_vram_gb, 1),
            "recommended_device": self.recommended_device,
            "recommended_profile": self.recommended_profile,
        }


def detect_apple_silicon() -> tuple[bool, Optional[str]]:
    """Detect if running on Apple Silicon and return chip model."""
    if platform.system() != "Darwin":
        return False, None

    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if "Apple" in brand or platform.machine() == "arm64":
            return True, brand if "Apple" in brand else "Apple Silicon (ARM64)"
    except Exception:
        if platform.machine() == "arm64":
            return True, "Apple Silicon (ARM64)"
    return False, None


def detect_ram() -> tuple[float, float]:
    """Detect total and approximately available RAM in GB."""
    total_gb = 8.0
    available_gb = 4.0

    sys_name = platform.system()
    if sys_name == "Darwin":
        try:
            mem_bytes = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
            total_gb = mem_bytes / (1024**3)
            # macOS memory pressure heuristic: ~50-70% total is usable for ML
            available_gb = total_gb * 0.65
        except Exception:
            pass
    elif sys_name == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    info[key] = int(val)
            if "MemTotal" in info:
                total_gb = info["MemTotal"] / (1024**2)
            if "MemAvailable" in info:
                available_gb = info["MemAvailable"] / (1024**2)
            else:
                available_gb = total_gb * 0.5
        except Exception:
            pass
    elif sys_name == "Windows":
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
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_gb = stat.ullTotalPhys / (1024**3)
            available_gb = stat.ullAvailPhys / (1024**3)
        except Exception:
            pass

    return total_gb, available_gb


def detect_cuda() -> tuple[bool, Optional[str], float]:
    """Detect NVIDIA CUDA GPU, name, and total VRAM in GB."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        if lines:
            parts = lines[0].split(",")
            name = parts[0].strip()
            vram_mb = float(parts[1].strip())
            return True, name, vram_mb / 1024.0
    except Exception:
        pass

    # Check if torch is installed and sees cuda
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            return True, name, vram_bytes / (1024**3)
    except Exception:
        pass

    return False, None, 0.0


def get_hardware_profile() -> HardwareProfile:
    """Analyze current machine and return hardware profile with recommendations."""
    os_name = platform.system()
    arch = platform.machine()
    total_ram, avail_ram = detect_ram()
    is_apple, apple_chip = detect_apple_silicon()
    has_cuda, cuda_name, cuda_vram = detect_cuda()

    # Determine recommended compute device
    if has_cuda:
        rec_device = "cuda"
    elif is_apple:
        rec_device = "mps"
    else:
        rec_device = "cpu"

    # Determine recommended profile
    effective_vram = cuda_vram if has_cuda else (total_ram if is_apple else avail_ram)

    if effective_vram >= 24.0:
        rec_profile = "deep_reasoning"
    elif effective_vram >= 12.0:
        rec_profile = "balanced"
    else:
        rec_profile = "speed_demon"

    return HardwareProfile(
        os_name=os_name,
        architecture=arch,
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        is_apple_silicon=is_apple,
        apple_chip_name=apple_chip,
        has_cuda=has_cuda,
        cuda_device_name=cuda_name,
        cuda_vram_gb=cuda_vram,
        recommended_device=rec_device,
        recommended_profile=rec_profile,
    )


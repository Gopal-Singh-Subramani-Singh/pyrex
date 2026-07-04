from __future__ import annotations
import subprocess
import json
import time
import psutil
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


def get_mps_memory_mb() -> Optional[float]:
    """Current MPS (GPU) allocated memory in MB."""
    try:
        import torch
        if torch.backends.mps.is_available():
            bytes_ = torch.mps.current_allocated_memory()
            return bytes_ / (1024 ** 2)
    except Exception:
        pass
    return None


def get_cpu_memory_mb() -> float:
    """Current process RSS memory in MB."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 ** 2)


def get_cpu_percent() -> float:
    """System-wide CPU utilisation %."""
    return psutil.cpu_percent(interval=0.1)


def get_power_watts() -> Optional[float]:
    """
    Read CPU+GPU power draw via macOS powermetrics.
    Requires sudo on most systems.
    Returns None if powermetrics is unavailable or fails.
    """
    try:
        result = subprocess.run(
            [
                "sudo", "-n", "powermetrics",
                "--samplers", "cpu_power,gpu_power",
                "-n", "1",
                "-i", "100",
                "--format", "plist",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        import plistlib
        data = plistlib.loads(result.stdout.encode())
        cpu_power = data.get("processor", {}).get("package_watts", 0)
        gpu_power = data.get("gpu", {}).get("gpu_energy", {}).get("total", 0) / 1000.0
        return float(cpu_power + gpu_power)
    except Exception:
        return None


def snapshot() -> dict:
    """Capture a full telemetry snapshot."""
    return {
        "mps_memory_mb": get_mps_memory_mb(),
        "cpu_memory_mb": get_cpu_memory_mb(),
        "power_watts": get_power_watts(),
        "cpu_percent": get_cpu_percent(),
    }

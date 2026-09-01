"""Small, dependency-light hardware observations for benchmark reports."""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any


def _first_cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name") and ":" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_total_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _nvidia_observation() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "model": None, "memory_total_mb": None, "driver": None}
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"available": False, "model": None, "memory_total_mb": None, "driver": None, "error": completed.stderr.strip()}
    first = completed.stdout.strip().splitlines()[0].split(",")
    return {
        "available": True,
        "model": first[0].strip() if first else None,
        "memory_total_mb": int(first[1].strip()) if len(first) > 1 and first[1].strip().isdigit() else None,
        "driver": first[2].strip() if len(first) > 2 else None,
    }


def collect_hardware_info(delegate_requested: str) -> dict[str, Any]:
    """Collect observations without inferring that an installed GPU was used."""

    gpu = _nvidia_observation()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _first_cpu_model(),
        "ram_total_bytes": _ram_total_bytes(),
        "gpu": gpu,
        "mediapipe_execution_device": delegate_requested.upper(),
        "execution_device_note": "The recorded device is the delegate requested in the task options; GPU presence alone is not treated as GPU execution.",
    }

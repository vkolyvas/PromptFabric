"""Hardware detection utilities for auto-selecting LLM provider and models."""

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareInfo:
    """Detected hardware information."""

    has_nvidia_gpu: bool = False
    has_apple_silicon: bool = False
    has_amd_gpu: bool = False
    total_ram_gb: float = 0.0
    os_type: str = ""
    cpu_cores: int = 0


def detect_hardware() -> HardwareInfo:
    """Detect available hardware capabilities."""
    info = HardwareInfo()
    info.os_type = platform.system().lower()
    info.cpu_cores = os.cpu_count() or 4

    # Detect RAM
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info.total_ram_gb = kb / (1024 * 1024)
                        break
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info.total_ram_gb = int(result.stdout.strip()) / (1024**3)
        elif platform.system() == "Windows":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", c_ulong),
                    ("dwMemoryLoad", c_ulong),
                    ("dwTotalPhys", c_ulong),
                    ("dwAvailPhys", c_ulong),
                    ("dwTotalPageFile", c_ulong),
                    ("dwAvailPageFile", c_ulong),
                    ("dwTotalVirtual", c_ulong),
                    ("dwAvailVirtual", c_ulong),
                ]

            stat = MEMORYSTATUS()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatus(ctypes.byref(stat))
            info.total_ram_gb = stat.dwTotalPhys / (1024**3)
    except Exception:
        pass

    # Detect NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            info.has_nvidia_gpu = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Detect Apple Silicon
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if "Apple" in result.stdout:
                info.has_apple_silicon = True
        except Exception:
            pass

    # Detect AMD GPU (Linux)
    if platform.system() == "Linux" and not info.has_nvidia_gpu:
        try:
            result = subprocess.run(
                ["lspci"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "AMD" in result.stdout or "Radeon" in result.stdout:
                info.has_amd_gpu = True
        except Exception:
            pass

    return info


def get_recommended_provider(hardware: HardwareInfo) -> str:
    """Recommend LLM provider based on hardware."""
    if hardware.has_nvidia_gpu:
        return "lm_studio"  # Best for NVIDIA GPUs
    elif hardware.has_apple_silicon:
        return "ollama"  # Ollama works well on Apple Silicon
    elif hardware.has_amd_gpu:
        return "ollama"  # Ollama has better AMD support
    else:
        return "ollama"  # CPU-only default


def get_recommended_models(
    provider: str, hardware: HardwareInfo
) -> dict:
    """Recommend models based on hardware and provider."""
    ram = hardware.total_ram_gb

    # NVIDIA GPU with LM Studio (VRAM focused)
    if hardware.has_nvidia_gpu and provider == "lm_studio":
        if ram >= 32:
            return {
                "generator": "deepseek-coder-r1-14b",
                "refiner": "gemma-3-4b-it",
                "validator": "phi-4-mini",
            }
        elif ram >= 24:
            return {
                "generator": "deepseek-coder-r1-7b",
                "refiner": "gemma-2b-it",
                "validator": "phi-3-mini",
            }
        elif ram >= 16:
            return {
                "generator": "qwen2.5-coder-7b",
                "refiner": "gemma-2b-it",
                "validator": "phi-3-mini",
            }
        else:
            return {
                "generator": "llama3.1:8b",
                "refiner": "gemma:2b",
                "validator": "phi3:3.8b",
            }

    # Apple Silicon with Ollama (Metal GPU)
    elif hardware.has_apple_silicon and provider == "ollama":
        if ram >= 24:
            return {
                "generator": "llama3.2:3b",
                "refiner": "gemma:2b",
                "validator": "phi3:3.8b",
            }
        elif ram >= 16:
            return {
                "generator": "llama3.2:3b",
                "refiner": "gemma:2b",
                "validator": "phi3:3.8b",
            }
        else:
            return {
                "generator": "llama3.2:1b",
                "refiner": "gemma:1b",
                "validator": "phi3:3.8b",
            }

    # AMD GPU with Ollama (ROCm)
    elif hardware.has_amd_gpu and provider == "ollama":
        if ram >= 24:
            return {
                "generator": "llama3.2:3b",
                "refiner": "gemma:2b",
                "validator": "phi3:3.8b",
            }
        elif ram >= 16:
            return {
                "generator": "llama3.2:3b",
                "refiner": "gemma:2b",
                "validator": "phi3:3.8b",
            }
        else:
            return {
                "generator": "llama3.2:1b",
                "refiner": "gemma:1b",
                "validator": "phi3:3.8b",
            }

    # CPU Only
    else:
        if ram >= 16:
            return {
                "generator": "llama3.2:1b",
                "refiner": "gemma:1b",
                "validator": "phi3:3.8b",
            }
        elif ram >= 8:
            return {
                "generator": "llama3.2:1b",
                "refiner": "tinyllama",
                "validator": "phi3:3.8b",
            }
        else:
            return {
                "generator": "llama3.2:1b",
                "refiner": "tinyllama",
                "validator": "tinyllama",
            }


def print_hardware_info():
    """Print detected hardware and recommendations."""
    hardware = detect_hardware()

    print("\n" + "=" * 50)
    print("HARDWARE DETECTION")
    print("=" * 50)
    print(f"OS: {hardware.os_type}")
    print(f"CPU Cores: {hardware.cpu_cores}")
    print(f"Total RAM: {hardware.total_ram_gb:.1f} GB")
    print(f"NVIDIA GPU: {'Yes' if hardware.has_nvidia_gpu else 'No'}")
    print(f"Apple Silicon: {'Yes' if hardware.has_apple_silicon else 'No'}")
    print(f"AMD GPU: {'Yes' if hardware.has_amd_gpu else 'No'}")

    provider = get_recommended_provider(hardware)
    models = get_recommended_models(provider, hardware)

    print("\n" + "-" * 50)
    print("RECOMMENDATIONS")
    print("-" * 50)
    print(f"Provider: {provider.upper()}")

    if hardware.has_nvidia_gpu:
        print("Note: LM Studio is recommended for NVIDIA GPUs")
    elif hardware.has_apple_silicon:
        print("Note: Ollama is recommended for Apple Silicon")
    elif hardware.has_amd_gpu:
        print("Note: Ollama is recommended for AMD GPUs")
    else:
        print("Note: Ollama is best for CPU-only environments")

    print(f"\nModels (for {hardware.total_ram_gb:.0f}GB RAM):")
    print(f"  Generator: {models['generator']}")
    print(f"  Refiner:   {models['refiner']}")
    print(f"  Validator: {models['validator']}")

    print("\n" + "=" * 50)
    return hardware, provider, models


if __name__ == "__main__":
    h, p, m = print_hardware_info()

    print("\nTo configure PromptFabric, set these environment variables:")
    if p == "lm_studio":
        print(f"  export LLM_PROVIDER=lm_studio")
        print(f"  export GENERATOR_MODEL={m['generator']}")
        print(f"  export REFINER_MODEL={m['refiner']}")
        print(f"  export VALIDATOR_MODEL={m['validator']}")
    else:
        print(f"  export LLM_PROVIDER=ollama")
        print(f"  export OLLAMA_MODEL={m['generator']}")
        print(f"  export REFINER_MODEL={m['refiner']}")
        print(f"  export VALIDATOR_MODEL={m['validator']}")

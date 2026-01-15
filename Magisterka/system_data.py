import platform
import psutil
import shutil
import subprocess
import json

def get_cpu_info():
    return {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "max_frequency_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else None,
        "current_frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
    }

def get_memory_info():
    mem = psutil.virtual_memory()
    return {
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2)
    }

def get_disk_info():
    total, used, free = shutil.disk_usage("/")
    return {
        "total_disk_gb": round(total / (1024**3), 2),
        "free_disk_gb": round(free / (1024**3), 2)
    }

def get_gpu_info():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            encoding="utf-8"
        )
        gpus = []
        for line in output.strip().split("\n"):
            name, memory = line.split(",")
            gpus.append({
                "name": name.strip(),
                "memory_mb": memory.strip()
            })
        return gpus
    except Exception:
        return "No NVIDIA GPU detected or nvidia-smi not available"

def get_system_info():
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.architecture()[0],
        "python_version": platform.python_version()
    }

hardware_info = {
    "system": get_system_info(),
    "cpu": get_cpu_info(),
    "memory": get_memory_info(),
    "disk": get_disk_info(),
    "gpu": get_gpu_info()
}

print(json.dumps(hardware_info, indent=4))

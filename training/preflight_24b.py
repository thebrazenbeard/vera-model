"""Fail-closed Lappy training-capacity preflight for full 24B donor LoRA.

Tests whether the specific safe QLoRA recipe may run without evicting unrelated
workloads. A HOLD is NOT a claim that no conceivable offload strategy exists.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


DONOR_REPOSITORY = ("DavidAU/"
    "Dolphin-Mistral-GLM-4.7-Flash-24B-Venice-Edition-Thinking-Uncensored")
DONOR_REVISION = "f658f1724210316e17463d06b752ac243e5c85b0"
# Conservative minimum for the planned 4-bit LoRA recipe. Not a physical law.
MIN_GPU_TOTAL_MIB = 24576
MIN_GPU_FREE_MIB = 20480
MIN_HOST_FREE_GIB = 24
DONOR_MIN_UNQUANTIZED_GIB = 43.9


def classify(*, gpu_total_mib: int, gpu_free_mib: int,
             host_total_gib: float, host_free_gib: float,
             donor_checkpoint_complete: bool) -> dict:
    reasons = []
    if gpu_total_mib < MIN_GPU_TOTAL_MIB:
        reasons.append("planned 24B QLoRA requires a larger GPU than is installed")
    if gpu_free_mib < MIN_GPU_FREE_MIB:
        reasons.append("insufficient currently free VRAM for the bounded recipe")
    if host_free_gib < MIN_HOST_FREE_GIB:
        reasons.append("insufficient unoccupied host RAM for this recipe")
    if not donor_checkpoint_complete:
        reasons.append("exact full donor safetensors are not available locally")
    return {
        "schema": "VERA_DOLPHIN_24B_TRAINING_PREFLIGHT_V1",
        "target_donor": DONOR_REPOSITORY,
        "target_revision": DONOR_REVISION,
        "status": "READY_FOR_RECIPE_ONLY" if not reasons else "HOLD_NO_OPTIMIZER_RUN",
        "reasons": reasons,
        "observations": {
            "gpu_total_mib": gpu_total_mib,
            "gpu_free_mib": gpu_free_mib,
            "host_total_gib": round(host_total_gib, 2),
            "host_free_gib": round(host_free_gib, 2),
            "donor_checkpoint_complete": donor_checkpoint_complete,
        },
        "conservative_recipe_thresholds": {
            "gpu_total_mib": MIN_GPU_TOTAL_MIB,
            "gpu_free_mib": MIN_GPU_FREE_MIB,
            "host_free_gib": MIN_HOST_FREE_GIB,
        },
        "minimum_raw_bf16_weight_size_gib_approx": DONOR_MIN_UNQUANTIZED_GIB,
        "quantized_gguf_is_not_a_pytorch_training_checkpoint": True,
        "training_executed_verified": False,
        "optimizer_steps_verified": 0,
        "adapter_artifact_verified": False,
        "qualification": "UNVERIFIED",
    }


def windows_ram() -> tuple[float, float]:
    script = ('Get-CimInstance Win32_OperatingSystem | '
              'Select-Object TotalVisibleMemorySize,FreePhysicalMemory | '
              'ConvertTo-Json -Compress')
    call = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script],
                          text=True, capture_output=True, timeout=25, check=True)
    data = json.loads(call.stdout)
    return (int(data["TotalVisibleMemorySize"]) / 1048576,
            int(data["FreePhysicalMemory"]) / 1048576)


def parse_linux_meminfo(source: str) -> tuple[float, float]:
    values = {}
    for line in source.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if len(parts) == 2 and parts[1] == "kB" and parts[0].isdigit():
            values[key] = int(parts[0])
    if "MemTotal" not in values or "MemAvailable" not in values:
        raise ValueError("Linux memory report lacks total/available")
    total = values["MemTotal"] / 1048576
    available = values["MemAvailable"] / 1048576
    if total <= 0 or not 0 <= available <= total:
        raise ValueError("invalid Linux memory capacity")
    return total, available


def host_ram() -> tuple[float, float]:
    if sys.platform == "win32":
        return windows_ram()
    if sys.platform == "linux":
        return parse_linux_meminfo(Path("/proc/meminfo").read_text(encoding="ascii"))
    raise RuntimeError("unsupported host-memory platform")


def gpu_memory() -> tuple[int, int]:
    call = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=15, check=True)
    lines = [line.strip() for line in call.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("expected exactly one GPU for this Lappy recipe")
    pieces = [int(x.strip()) for x in lines[0].split(",")]
    if len(pieces) != 2 or not (0 <= pieces[1] <= pieces[0]):
        raise ValueError("invalid GPU memory readback")
    return pieces[0], pieces[1]


def complete_checkpoint(root: Path) -> bool:
    # The pinned donor was published in 10 safetensors shards. A directory of
    # GGUFs, partial files or empty placeholders is NOT a training checkpoint.
    shards = sorted(root.glob("model-*-of-00010.safetensors"))
    names = {p.name for p in shards}
    expected = {f"model-{idx:05d}-of-00010.safetensors"
                for idx in range(1, 11)}
    return names == expected and all(p.is_file() and p.stat().st_size > 1000000
                                     for p in shards) and (root / "config.json").is_file()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise SystemExit("HOLD: output must be a new file in an existing directory")
    try:
        host_total, host_free = host_ram()
        gpu_total, gpu_free = gpu_memory()
        candidate = classify(gpu_total_mib=gpu_total, gpu_free_mib=gpu_free,
                             host_total_gib=host_total,
                             host_free_gib=host_free,
                             donor_checkpoint_complete=complete_checkpoint(
                                 args.base_checkpoint))
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        candidate = {"schema": "VERA_DOLPHIN_24B_TRAINING_PREFLIGHT_V1",
                     "status": "HOLD_MEASUREMENT_UNKNOWN", "reason": str(exc),
                     "training_executed_verified": False}
    args.output.write_text(json.dumps(candidate, indent=2, sort_keys=True)
                           + "\n", encoding="utf8")
    print(json.dumps(candidate, indent=2, sort_keys=True))
    return 0 if candidate["status"] == "READY_FOR_RECIPE_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

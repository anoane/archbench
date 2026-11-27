# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from pathlib import Path
import re
import shutil
import subprocess
from .runner import atomic_json, utc_now

ROOT = Path(__file__).resolve().parents[2]


def command_info(name, args=("--version",)):
    path = shutil.which(name)
    if not path: return {"available": False}
    try:
        result = subprocess.run([path, *args], capture_output=True, text=True, timeout=15)
        return {"available": True, "path": path, "returncode": result.returncode,
                "stdout": result.stdout[:16000], "stderr": result.stderr[:4000]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "path": path, "error": str(error)}


def doctor():
    result = {"schema_version": 1, "checked_at": utc_now(), "required_toolkit": "13.5 or newer",
              "tools": {name: command_info(name) for name in ("nvcc", "ptxas", "cuobjdump", "cmake", "ninja", "g++", "ncu", "compute-sanitizer")}}
    result["nvidia_smi"] = command_info("nvidia-smi", ("-q",))
    result["topology"] = command_info("nvidia-smi", ("topo", "-m"))
    result["compiler_targets"] = command_info("nvcc", ("--list-gpu-code",))
    nvcc = result["tools"]["nvcc"]
    match = re.search(r"release\s+(\d+)\.(\d+)", nvcc.get("stdout", ""))
    result["toolkit_version_ok"] = bool(match and tuple(map(int, match.groups())) >= (13, 5))
    result["missing_targets"] = ["sm_" + arch for arch in ("80", "89", "120a") if "sm_" + arch not in result["compiler_targets"].get("stdout", "")]
    return result


def build(args):
    cmake = shutil.which("cmake")
    if not cmake: raise ValueError("cmake >= 3.24 is required; install it on the build machine")
    output = Path(args.build_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    architectures = args.arch.split(",")
    if any(arch not in ("80", "89", "120a") for arch in architectures): raise ValueError("--arch must use 80,89,120a")
    command = [cmake, "-S", str(ROOT), "-B", str(output), "-DCMAKE_BUILD_TYPE=Release",
               "-DARCHBENCH_CUDA_ARCHS=" + ";".join(architectures),
               "-DARCHBENCH_ENABLE_CUDA=" + ("OFF" if args.cpu_only else "ON")]
    if args.nvcc: command += ["-DCMAKE_CUDA_COMPILER=" + str(Path(args.nvcc).resolve())]
    if shutil.which("ninja"): command += ["-G", "Ninja"]
    if args.allow_older_cuda: command += ["-DARCHBENCH_ALLOW_OLDER_CUDA=ON"]
    metadata = {"started_at": utc_now(), "configure_argv": command, "doctor": doctor(), "status": "configuring"}
    atomic_json(output / "build_provenance.json", metadata)
    with (output / "configure.log").open("w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        metadata["status"] = "configure_failed"; atomic_json(output / "build_provenance.json", metadata)
        raise ValueError(f"CMake configuration failed; inspect {output / 'configure.log'}")
    command = [cmake, "--build", str(output), "--parallel", str(args.jobs)]
    with (output / "build.log").open("w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    metadata.update({"build_argv": command, "finished_at": utc_now(), "returncode": code,
                     "status": "compiled" if code == 0 else "compile_failed", "gpu_execution_verified": False})
    atomic_json(output / "build_provenance.json", metadata)
    if code: raise ValueError(f"CUDA compilation failed; inspect {output / 'build.log'}; do not treat this as unsupported hardware")
    return metadata

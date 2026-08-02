#!/usr/bin/env python3
"""Capture an auditable platform fingerprint and evaluate its readiness."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import yaml


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_ROOT = REPOSITORY_ROOT / "platform"
SCHEMA_VERSION = "lunar-platform-fingerprint/v1"
PROFILES = ("train_amd64_rtx4080_super", "deploy_agx_orin_r36")


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a probe in a stable locale without raising for missing tools."""
    environment = os.environ.copy()
    environment.update({"LC_ALL": "C", "LANG": "C"})
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except OSError as error:
        return subprocess.CompletedProcess(command, 127, stdout="", stderr=str(error))


def parse_os_release(contents: str) -> dict[str, str]:
    """Parse the identity fields used for Ubuntu baseline comparison."""
    values: dict[str, str] = {}
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", maxsplit=1)
        try:
            parts = shlex.split(raw_value, posix=True)
            value = parts[0] if parts else ""
        except ValueError:
            value = raw_value.strip().strip('"')
        values[key] = value
    return {
        "name": values.get("NAME", ""),
        "version_id": values.get("VERSION_ID", ""),
        "pretty_name": values.get("PRETTY_NAME", ""),
    }


def _normalize_nvidia_pci_device_id(raw_value: str) -> str:
    compact = raw_value.strip().lower().removeprefix("0x")
    if not re.fullmatch(r"[0-9a-f]{8}", compact):
        raise ValueError(f"unexpected NVIDIA PCI device ID: {raw_value!r}")
    device_id, vendor_id = compact[:4], compact[4:]
    return f"{vendor_id}:{device_id}"


def parse_nvidia_smi(output: str) -> dict[str, object]:
    """Parse the single-GPU CSV query and normalize NVIDIA's PCI identifier."""
    rows = [line for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one NVIDIA GPU, got {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 6:
        raise ValueError(f"expected six NVIDIA fields, got {len(fields)}")
    model, pci_bus_id, raw_pci_device_id, memory_mib, compute_capability, driver = fields
    return {
        "available": True,
        "model": model,
        "pci_bus_id": pci_bus_id,
        "pci_device_id": _normalize_nvidia_pci_device_id(raw_pci_device_id),
        "memory_total_mib": int(memory_mib),
        "compute_capability": compute_capability,
        "driver_version": driver,
    }


def parse_nvcc(output: str) -> dict[str, object]:
    """Extract both CUDA release and complete compiler version from nvcc output."""
    match = re.search(r"release\s+([0-9.]+),\s+V([0-9.]+)", output)
    if match is None:
        raise ValueError("nvcc output did not contain release and compiler version")
    return {
        "available": True,
        "release": match.group(1),
        "compiler_version": match.group(2),
    }


def unavailable_probe(
    command: str, returncode: int, stdout: str, stderr: str
) -> dict[str, object]:
    """Preserve a failed probe's command, code, and best available diagnostic."""
    error = stderr.strip() or stdout.strip() or f"exit code {returncode}"
    return {
        "available": False,
        "command": command,
        "returncode": returncode,
        "error": error,
    }


def load_baseline(
    profile: str, baseline_root: Path = DEFAULT_BASELINE_ROOT
) -> dict[str, object]:
    """Load one named platform baseline from its authoritative YAML document."""
    path = Path(baseline_root) / profile / "baseline.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"platform baseline must be a mapping: {path}")
    return document


def _command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def _probe_result(
    command: list[str],
    result: subprocess.CompletedProcess[str],
    parser: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    display = _command_text(command)
    if result.returncode != 0:
        return unavailable_probe(display, result.returncode, result.stdout, result.stderr)
    try:
        return parser(result.stdout or result.stderr)
    except (TypeError, ValueError) as error:
        return unavailable_probe(display, result.returncode, "", f"parse error: {error}")


def _parse_lscpu(output: str) -> dict[str, object]:
    fields: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields.setdefault(key.strip(), value.strip())
    model = fields.get("Model name", "")
    logical_cpus = fields.get("CPU(s)", "")
    if not model or not logical_cpus:
        raise ValueError("lscpu output lacks Model name or CPU(s)")
    return {"available": True, "model": model, "logical_cpus": int(logical_cpus)}


def _parse_memory(output: str) -> dict[str, object]:
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] == "Mem:" and len(fields) >= 2:
            return {"available": True, "total_bytes": int(fields[1])}
    raise ValueError("free output lacks Mem total")


def _parse_prefixed_version(prefix: str) -> Callable[[str], dict[str, object]]:
    def parse(output: str) -> dict[str, object]:
        first_line = output.strip().splitlines()[0] if output.strip() else ""
        value = first_line.removeprefix(prefix).strip()
        if not value:
            raise ValueError(f"version output lacks {prefix.strip()}")
        return {"available": True, "version": value}

    return parse


def _parse_plain_version(output: str) -> dict[str, object]:
    value = output.strip().splitlines()[0] if output.strip() else ""
    if not value:
        raise ValueError("version output is empty")
    return {"available": True, "version": value}


def _parse_value(output: str) -> dict[str, object]:
    value = output.strip().rstrip("\x00").strip()
    if not value:
        raise ValueError("probe output is empty")
    return {"available": True, "value": value}


def _audit_output(output: str) -> str:
    """Render probe output on one line without losing tabs or line boundaries."""
    value = output.strip()
    if not value:
        return "<empty>"
    return value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def _parse_tensorrt_packages(output: str) -> dict[str, object]:
    """Select an installed libnvinfer runtime from dpkg-query package records."""
    installed: list[tuple[int, str, str]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 3:
            continue
        binary_package, status, version = (field.strip() for field in fields)
        package = binary_package.split(":", maxsplit=1)[0]
        match = re.fullmatch(r"libnvinfer(\d+)", package)
        if match and status == "ii" and version:
            installed.append((int(match.group(1)), package, version))
    if not installed:
        raise ValueError(
            "no installed TensorRT runtime in dpkg-query output; "
            f"raw output: {_audit_output(output)}"
        )
    _major, package, version = max(installed)
    return {"available": True, "package": package, "version": version}


def _parse_lspci_device_id(output: str) -> str:
    """Extract the normalized PCI vendor/device ID from one lspci device record."""
    matches = re.findall(
        r"(?<![0-9a-f])([0-9a-f]{4}):([0-9a-f]{4})(?![0-9a-f])",
        output.lower(),
    )
    identifiers = {f"{vendor}:{device}" for vendor, device in matches}
    if len(identifiers) != 1:
        raise ValueError("lspci output did not contain a vendor/device ID")
    return identifiers.pop()


def _parse_l4t(output: str) -> dict[str, object]:
    value = output.strip()
    direct = re.search(r"\bR\d+(?:\.\d+){2}\b", value)
    if direct:
        return {"available": True, "value": direct.group(0)}
    release = re.search(r"#\s*R(\d+).*?REVISION:[ \t]*([0-9]+(?:\.[0-9]+)?)", value)
    if release:
        revision = release.group(2)
        if revision.count(".") == 0:
            revision += ".0"
        return {"available": True, "value": f"R{release.group(1)}.{revision}"}
    raise ValueError("L4T release output is unrecognized")


def _parse_jetpack(output: str) -> dict[str, object]:
    match = re.search(r"\b(\d+\.\d+)\b", output)
    if match is None:
        raise ValueError("JetPack package version is unrecognized")
    return {"available": True, "value": match.group(1)}


def _read_os_release() -> dict[str, str]:
    try:
        contents = Path("/etc/os-release").read_text(encoding="utf-8")
    except OSError:
        contents = ""
    return parse_os_release(contents)


def capture_environment(
    profile: str, *, run: CommandRunner = run_command
) -> dict[str, object]:
    """Capture actual host values without applying baseline policy."""
    if profile not in PROFILES:
        raise ValueError(f"unsupported platform profile: {profile}")

    uname_arch_command = ["uname", "-m"]
    uname_arch = run(uname_arch_command)
    raw_architecture = uname_arch.stdout.strip() if uname_arch.returncode == 0 else ""
    architecture = {"x86_64": "amd64", "aarch64": "aarch64"}.get(
        raw_architecture, raw_architecture
    )

    uname_kernel = run(["uname", "-r"])
    kernel = uname_kernel.stdout.strip() if uname_kernel.returncode == 0 else ""

    cpu_command = ["lscpu"]
    cpu_result = run(cpu_command)
    cpu = _probe_result(cpu_command, cpu_result, _parse_lscpu)

    memory_command = ["free", "-b"]
    memory_result = run(memory_command)
    memory = _probe_result(memory_command, memory_result, _parse_memory)

    ros_distro = os.environ.get("ROS_DISTRO", "").strip()
    ros: dict[str, object]
    if ros_distro:
        ros = {"available": True, "distro": ros_distro}
    else:
        ros = unavailable_probe("ROS_DISTRO environment variable", 1, "", "not set")

    python_command = ["python3", "--version"]
    python_result = run(python_command)
    python = _probe_result(
        python_command, python_result, _parse_prefixed_version("Python ")
    )

    gcc_command = ["gcc", "-dumpfullversion"]
    gcc_result = run(gcc_command)
    gcc = _probe_result(gcc_command, gcc_result, _parse_plain_version)

    cmake_command = ["cmake", "--version"]
    cmake_result = run(cmake_command)
    cmake = _probe_result(
        cmake_command, cmake_result, _parse_prefixed_version("cmake version ")
    )

    gpu_command = [
        "nvidia-smi",
        "--query-gpu=name,pci.bus_id,pci.device_id,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    gpu_result = run(gpu_command)
    gpu = _probe_result(gpu_command, gpu_result, parse_nvidia_smi)
    if gpu.get("available") is True:
        lspci_command = ["lspci", "-Dnn", "-s", str(gpu["pci_bus_id"])]
        lspci_result = run(lspci_command)
        if lspci_result.returncode != 0:
            gpu = unavailable_probe(
                _command_text(lspci_command),
                lspci_result.returncode,
                lspci_result.stdout,
                lspci_result.stderr,
            )
        else:
            try:
                lspci_device_id = _parse_lspci_device_id(lspci_result.stdout)
            except ValueError as error:
                gpu = unavailable_probe(
                    _command_text(lspci_command),
                    lspci_result.returncode,
                    "",
                    f"parse error: {error}; raw output: {_audit_output(lspci_result.stdout)}",
                )
            else:
                nvidia_device_id = str(gpu["pci_device_id"])
                if lspci_device_id != nvidia_device_id:
                    gpu = unavailable_probe(
                        _command_text(lspci_command),
                        lspci_result.returncode,
                        "",
                        (
                            "PCI device ID mismatch: "
                            f"nvidia-smi reported {nvidia_device_id!r}, "
                            f"lspci reported {lspci_device_id!r}"
                        ),
                    )

    cuda_command = ["/usr/local/cuda/bin/nvcc", "--version"]
    cuda_result = run(cuda_command)
    cuda = _probe_result(cuda_command, cuda_result, parse_nvcc)

    tensorrt_command = [
        "dpkg-query",
        "-W",
        "-f=${binary:Package}\\t${db:Status-Abbrev}\\t${Version}\\n",
        "libnvinfer[0-9]*",
    ]
    tensorrt_result = run(tensorrt_command)
    tensorrt = _probe_result(
        tensorrt_command, tensorrt_result, _parse_tensorrt_packages
    )

    device_model_command = ["cat", "/proc/device-tree/model"]
    device_model_result = run(device_model_command)
    device_model = _probe_result(device_model_command, device_model_result, _parse_value)

    l4t_command = ["cat", "/etc/nv_tegra_release"]
    l4t_result = run(l4t_command)
    l4t = _probe_result(l4t_command, l4t_result, _parse_l4t)

    jetpack_command = ["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"]
    jetpack_result = run(jetpack_command)
    jetpack = _probe_result(jetpack_command, jetpack_result, _parse_jetpack)

    power_mode_command = ["nvpmodel", "-q"]
    power_mode_result = run(power_mode_command)
    power_mode = _probe_result(power_mode_command, power_mode_result, _parse_value)

    clocks_command = ["jetson_clocks", "--show"]
    clocks_result = run(clocks_command)
    clocks = _probe_result(clocks_command, clocks_result, _parse_value)

    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "captured_at_utc": captured_at,
        "os": _read_os_release(),
        "architecture": architecture,
        "kernel": kernel,
        "cpu": cpu,
        "memory": memory,
        "ros": ros,
        "python": python,
        "gcc": gcc,
        "cmake": cmake,
        "gpu": gpu,
        "cuda": cuda,
        "tensorrt": tensorrt,
        "device_model": device_model,
        "l4t": l4t,
        "jetpack": jetpack,
        "power_mode": power_mode,
        "clocks": clocks,
        "readiness": {"ready": False, "errors": []},
    }


def _nested(document: Mapping[str, object], *path: str) -> object:
    value: object = document
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _expect(errors: list[str], path: str, expected: object, actual: object) -> None:
    if actual != expected:
        errors.append(f"{path}: expected {expected!r}, got {actual!r}")


def _major_minor(version: object) -> object:
    if not isinstance(version, str):
        return version
    match = re.match(r"^(\d+)\.(\d+)", version)
    return f"{match.group(1)}.{match.group(2)}" if match else version


def validate_fingerprint(
    document: Mapping[str, object], baseline: Mapping[str, object]
) -> list[str]:
    """Return deterministic baseline mismatch errors for a captured document."""
    errors: list[str] = []
    _expect(errors, "schema_version", SCHEMA_VERSION, document.get("schema_version"))
    _expect(errors, "profile", baseline.get("profile"), document.get("profile"))

    baseline_os = baseline.get("os")
    if baseline_os == "Ubuntu 22.04 LTS":
        _expect(errors, "os.name", "Ubuntu", _nested(document, "os", "name"))
        _expect(errors, "os.version_id", "22.04", _nested(document, "os", "version_id"))
    _expect(errors, "architecture", baseline.get("architecture"), document.get("architecture"))
    _expect(errors, "ros.distro", baseline.get("ros_distro"), _nested(document, "ros", "distro"))

    profile = baseline.get("profile")
    if profile == "train_amd64_rtx4080_super":
        _expect(
            errors,
            "python.version",
            baseline.get("python"),
            _major_minor(_nested(document, "python", "version")),
        )
        _expect(errors, "gpu.available", True, _nested(document, "gpu", "available"))
        _expect(errors, "gpu.model", baseline.get("gpu_model"), _nested(document, "gpu", "model"))
        _expect(
            errors,
            "gpu.pci_device_id",
            baseline.get("gpu_pci_device_id"),
            _nested(document, "gpu", "pci_device_id"),
        )
        driver_version = _nested(document, "gpu", "driver_version")
        if not isinstance(driver_version, str) or not driver_version:
            errors.append(
                f"gpu.driver_version: expected a non-empty value, got {driver_version!r}"
            )
        _expect(errors, "cuda.available", True, _nested(document, "cuda", "available"))
    elif profile == "deploy_agx_orin_r36":
        _expect(
            errors,
            "device_model.available",
            True,
            _nested(document, "device_model", "available"),
        )
        _expect(
            errors,
            "device_model.value",
            baseline.get("device_model"),
            _nested(document, "device_model", "value"),
        )
        _expect(errors, "l4t.available", True, _nested(document, "l4t", "available"))
        _expect(errors, "l4t.value", baseline.get("l4t"), _nested(document, "l4t", "value"))
        _expect(
            errors,
            "jetpack.available",
            True,
            _nested(document, "jetpack", "available"),
        )
        _expect(
            errors,
            "jetpack.value",
            baseline.get("jetpack"),
            _nested(document, "jetpack", "value"),
        )
        _expect(errors, "cuda.available", True, _nested(document, "cuda", "available"))
        _expect(
            errors,
            "tensorrt.available",
            True,
            _nested(document, "tensorrt", "available"),
        )
    else:
        errors.append(f"profile: unsupported baseline profile {profile!r}")
    return errors


def write_fingerprint(document: Mapping[str, object], output: Path) -> None:
    """Write deterministic UTF-8 JSON with LF line endings and a trailing newline."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    target.write_text(payload, encoding="utf-8", newline="\n")


def _inside_repository(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Capture, persist, and then express readiness through the process exit code."""
    arguments = _argument_parser().parse_args(argv)
    if _inside_repository(arguments.output):
        print("error: fingerprint output must be outside the Git repository")
        return 2

    document = capture_environment(arguments.profile)
    baseline = load_baseline(arguments.profile, arguments.baseline_root)
    errors = validate_fingerprint(document, baseline)
    document["readiness"] = {"ready": not errors, "errors": errors}
    write_fingerprint(document, arguments.output)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

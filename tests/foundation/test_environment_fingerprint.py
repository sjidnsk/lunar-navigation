"""Behavioral tests for auditable platform environment fingerprints."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tools.capture_environment import (
    DEFAULT_BASELINE_ROOT,
    _parse_jetpack,
    _parse_l4t,
    _parse_tensorrt_packages,
    capture_environment,
    load_baseline,
    main,
    parse_nvidia_smi,
    parse_nvcc,
    parse_os_release,
    run_command,
    unavailable_probe,
    validate_fingerprint,
    write_fingerprint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOP_LEVEL_KEYS = {
    "schema_version",
    "profile",
    "captured_at_utc",
    "os",
    "architecture",
    "kernel",
    "cpu",
    "memory",
    "ros",
    "python",
    "gcc",
    "cmake",
    "gpu",
    "cuda",
    "tensorrt",
    "device_model",
    "l4t",
    "jetpack",
    "power_mode",
    "clocks",
    "readiness",
}
NVIDIA_COMMAND = (
    "nvidia-smi",
    "--query-gpu=name,pci.bus_id,pci.device_id,memory.total,compute_cap,driver_version",
    "--format=csv,noheader,nounits",
)
NVIDIA_COMMAND_TEXT = " ".join(NVIDIA_COMMAND)
LSPCI_COMMAND = ("lspci", "-Dnn", "-s", "00000000:01:00.0")
LSPCI_ALL_COMMAND = ("lspci", "-Dnn")
DEVICE_MODEL_COMMAND = ("cat", "/proc/device-tree/model")
NV_BOOT_CONTROL_COMMAND = ("cat", "/etc/nv_boot_control.conf")
TENSORRT_COMMAND = (
    "dpkg-query",
    "-W",
    "-f=${binary:Package}\\t${db:Status-Abbrev}\\t${Version}\\n",
    "libnvinfer[0-9]*",
)
AGX_MEMORY_BYTES = 65_000_000_000
AGX_MINIMUM_MEMORY_BYTES = 60 * 1024**3
AGX_MODEL = "NVIDIA Jetson AGX Orin Developer Kit"
AGX_TNSPEC = "3701-500-0005-K.2-1-1-jetson-agx-orin-devkit-"


def valid_training_fingerprint() -> dict[str, object]:
    """Return a hand-maintained valid training host fingerprint."""
    unavailable = {
        "available": False,
        "command": "not-applicable-on-training-host",
        "returncode": 127,
        "error": "not available",
    }
    return {
        "schema_version": "lunar-platform-fingerprint/v1",
        "profile": "train_amd64_rtx4080_super",
        "captured_at_utc": "2026-08-02T12:00:00Z",
        "os": {
            "name": "Ubuntu",
            "version_id": "22.04",
            "pretty_name": "Ubuntu 22.04.5 LTS",
        },
        "architecture": "amd64",
        "kernel": "6.8.0-124-generic",
        "cpu": {
            "available": True,
            "model": "Intel(R) Core(TM) i7-14700KF",
            "logical_cpus": 28,
        },
        "memory": {"available": True, "total_bytes": 67223306240},
        "ros": {"available": True, "distro": "humble"},
        "python": {"available": True, "version": "3.10.12"},
        "gcc": {"available": True, "version": "11.4.0"},
        "cmake": {"available": True, "version": "3.22.1"},
        "gpu": {
            "available": True,
            "model": "NVIDIA GeForce RTX 4080 SUPER",
            "pci_bus_id": "00000000:01:00.0",
            "pci_device_id": "10de:2702",
            "memory_total_mib": 16376,
            "compute_capability": "8.9",
            "driver_version": "595.84",
        },
        "cuda": {"available": True, "release": "13.2", "compiler_version": "13.2.78"},
        "tensorrt": unavailable.copy(),
        "device_model": unavailable.copy(),
        "l4t": unavailable.copy(),
        "jetpack": unavailable.copy(),
        "power_mode": unavailable.copy(),
        "clocks": unavailable.copy(),
        "readiness": {"ready": True, "errors": []},
    }


def valid_agx_identity() -> dict[str, object]:
    """Return independently specified evidence for a qualified 64 GB AGX module."""
    return {
        "available": True,
        "value": "Jetson AGX Orin 64GB",
        "raw_model": AGX_MODEL,
        "model_source": "/proc/device-tree/model",
        "module_sku": "P3701-0005",
        "tnspec": AGX_TNSPEC,
        "module_source": "/etc/nv_boot_control.conf",
        "observed_memory_bytes": AGX_MEMORY_BYTES,
        "minimum_memory_bytes": AGX_MINIMUM_MEMORY_BYTES,
        "memory_source": "free -b",
    }


def valid_agx_fingerprint() -> dict[str, object]:
    document = valid_training_fingerprint()
    document.update(
        {
            "profile": "deploy_agx_orin_r36",
            "architecture": "aarch64",
            "memory": {"available": True, "total_bytes": AGX_MEMORY_BYTES},
            "device_model": valid_agx_identity(),
            "l4t": {"available": True, "value": "R36.0.0"},
            "jetpack": {"available": True, "value": "6.0"},
            "tensorrt": {
                "available": True,
                "package": "libnvinfer8",
                "version": "8.6.2.3-1+cuda12.0",
            },
        }
    )
    return document


def capture_runner(
    overrides: dict[tuple[str, ...], subprocess.CompletedProcess[str]] | None = None,
):
    """Return complete command-shaped platform fixtures with selected probe overrides."""
    responses = {
        ("uname", "-m"): subprocess.CompletedProcess(
            ["uname", "-m"], 0, stdout="x86_64\n", stderr=""
        ),
        ("uname", "-r"): subprocess.CompletedProcess(
            ["uname", "-r"], 0, stdout="6.8.0-test\n", stderr=""
        ),
        ("lscpu",): subprocess.CompletedProcess(
            ["lscpu"], 0, stdout="Model name: Test CPU\nCPU(s): 28\n", stderr=""
        ),
        ("free", "-b"): subprocess.CompletedProcess(
            ["free", "-b"], 0, stdout="Mem: 67223306240 1 2 3 4 5\n", stderr=""
        ),
        ("python3", "--version"): subprocess.CompletedProcess(
            ["python3", "--version"], 0, stdout="Python 3.10.12\n", stderr=""
        ),
        ("gcc", "-dumpfullversion"): subprocess.CompletedProcess(
            ["gcc", "-dumpfullversion"], 0, stdout="11.4.0\n", stderr=""
        ),
        ("cmake", "--version"): subprocess.CompletedProcess(
            ["cmake", "--version"], 0, stdout="cmake version 3.22.1\n", stderr=""
        ),
        NVIDIA_COMMAND: subprocess.CompletedProcess(
            list(NVIDIA_COMMAND),
            0,
            stdout=(
                "NVIDIA GeForce RTX 4080 SUPER, 00000000:01:00.0, "
                "0x270210DE, 16376, 8.9, 595.84\n"
            ),
            stderr="",
        ),
        LSPCI_COMMAND: subprocess.CompletedProcess(
            list(LSPCI_COMMAND),
            0,
            stdout=(
                "0000:01:00.0 VGA compatible controller [0300]: "
                "NVIDIA Corporation Device [10de:2702] (rev a1)\n"
            ),
            stderr="",
        ),
        ("/usr/local/cuda/bin/nvcc", "--version"): subprocess.CompletedProcess(
            ["/usr/local/cuda/bin/nvcc", "--version"],
            0,
            stdout="Cuda compilation tools, release 13.2, V13.2.78\n",
            stderr="",
        ),
    }
    if overrides:
        responses.update(overrides)

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return responses.get(
            tuple(command),
            subprocess.CompletedProcess(command, 127, stdout="", stderr="not available"),
        )

    return run


def agx_capture_runner(
    overrides: dict[tuple[str, ...], subprocess.CompletedProcess[str]] | None = None,
):
    responses = {
        ("uname", "-m"): subprocess.CompletedProcess(
            ["uname", "-m"], 0, stdout="aarch64\n", stderr=""
        ),
        ("uname", "-r"): subprocess.CompletedProcess(
            ["uname", "-r"], 0, stdout="5.15.122-tegra\n", stderr=""
        ),
        ("lscpu",): subprocess.CompletedProcess(
            ["lscpu"], 0, stdout="Model name: ARMv8 Processor rev 1\nCPU(s): 12\n", stderr=""
        ),
        ("free", "-b"): subprocess.CompletedProcess(
            ["free", "-b"], 0, stdout=f"Mem: {AGX_MEMORY_BYTES} 1 2 3 4 5\n", stderr=""
        ),
        ("python3", "--version"): subprocess.CompletedProcess(
            ["python3", "--version"], 0, stdout="Python 3.10.12\n", stderr=""
        ),
        ("gcc", "-dumpfullversion"): subprocess.CompletedProcess(
            ["gcc", "-dumpfullversion"], 0, stdout="11.4.0\n", stderr=""
        ),
        ("cmake", "--version"): subprocess.CompletedProcess(
            ["cmake", "--version"], 0, stdout="cmake version 3.22.1\n", stderr=""
        ),
        NVIDIA_COMMAND: subprocess.CompletedProcess(
            list(NVIDIA_COMMAND), 9, stdout="", stderr="nvidia-smi not supported\n"
        ),
        LSPCI_ALL_COMMAND: subprocess.CompletedProcess(
            list(LSPCI_ALL_COMMAND), 0, stdout="", stderr=""
        ),
        ("/usr/local/cuda/bin/nvcc", "--version"): subprocess.CompletedProcess(
            ["/usr/local/cuda/bin/nvcc", "--version"],
            0,
            stdout="Cuda compilation tools, release 12.2, V12.2.140\n",
            stderr="",
        ),
        TENSORRT_COMMAND: subprocess.CompletedProcess(
            list(TENSORRT_COMMAND),
            0,
            stdout="libnvinfer8:arm64\tii \t8.6.2.3-1+cuda12.0\n",
            stderr="",
        ),
        DEVICE_MODEL_COMMAND: subprocess.CompletedProcess(
            list(DEVICE_MODEL_COMMAND), 0, stdout=f"{AGX_MODEL}\x00", stderr=""
        ),
        NV_BOOT_CONTROL_COMMAND: subprocess.CompletedProcess(
            list(NV_BOOT_CONTROL_COMMAND),
            0,
            stdout=f"TNSPEC {AGX_TNSPEC}\nCOMPATIBLE_SPEC 3701-300-0005--1--\n",
            stderr="",
        ),
        ("cat", "/etc/nv_tegra_release"): subprocess.CompletedProcess(
            ["cat", "/etc/nv_tegra_release"],
            0,
            stdout="# R36 (release), REVISION: 0.0, BOARD: generic, EABI: aarch64\n",
            stderr="",
        ),
        ("dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"): subprocess.CompletedProcess(
            ["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"],
            0,
            stdout="6.0+b106\n",
            stderr="",
        ),
    }
    if overrides:
        responses.update(overrides)

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return responses.get(
            tuple(command),
            subprocess.CompletedProcess(command, 127, stdout="", stderr="not available"),
        )

    return run


def test_authoritative_platform_baselines_match_frozen_profiles():
    """Changing a platform identity or responsibility would move the release boundary."""
    training = yaml.safe_load(
        (DEFAULT_BASELINE_ROOT / "train_amd64_rtx4080_super/baseline.yaml").read_text(
            encoding="utf-8"
        )
    )
    deployment = yaml.safe_load(
        (DEFAULT_BASELINE_ROOT / "deploy_agx_orin_r36/baseline.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert training == {
        "schema_version": "lunar-platform-baseline/v1",
        "profile": "train_amd64_rtx4080_super",
        "os": "Ubuntu 22.04 LTS",
        "architecture": "amd64",
        "ros_distro": "humble",
        "python": "3.10",
        "gpu_model": "NVIDIA GeForce RTX 4080 SUPER",
        "gpu_pci_device_id": "10de:2702",
        "responsibilities": ["ros_integration", "ppo_training", "onnx_export", "rosbag_replay"],
    }
    assert deployment == {
        "schema_version": "lunar-platform-baseline/v1",
        "profile": "deploy_agx_orin_r36",
        "os": "Ubuntu 22.04 LTS",
        "architecture": "aarch64",
        "ros_distro": "humble",
        "device_model": "Jetson AGX Orin 64GB",
        "l4t": "R36.0.0",
        "jetpack": "6.0",
        "responsibilities": [
            "native_build",
            "tensorrt_engine",
            "inference",
            "device_release_gate",
        ],
    }


def test_parse_os_release_preserves_structured_ubuntu_values():
    """Using PRETTY_NAME as the release identity would reject Ubuntu patch releases."""
    assert parse_os_release(
        'NAME="Ubuntu"\nVERSION_ID="22.04"\nPRETTY_NAME="Ubuntu 22.04.5 LTS"\n'
    ) == {
        "name": "Ubuntu",
        "version_id": "22.04",
        "pretty_name": "Ubuntu 22.04.5 LTS",
    }


def test_parse_nvidia_probe_normalizes_rtx4080_super():
    """Reading NVIDIA's device/vendor word in display order would produce the wrong PCI ID."""
    result = parse_nvidia_smi(
        "NVIDIA GeForce RTX 4080 SUPER, 00000000:01:00.0, "
        "0x270210DE, 16376, 8.9, 595.84\n"
    )
    assert result == {
        "available": True,
        "model": "NVIDIA GeForce RTX 4080 SUPER",
        "pci_bus_id": "00000000:01:00.0",
        "pci_device_id": "10de:2702",
        "memory_total_mib": 16376,
        "compute_capability": "8.9",
        "driver_version": "595.84",
    }


def test_parse_nvcc_returns_release_and_full_compiler_version():
    """Collapsing CUDA to a display string would lose the auditable compiler version."""
    result = parse_nvcc(
        "Cuda compilation tools, release 13.2, V13.2.78\n"
        "Build cuda_13.2.r13.2/compiler.37668154_0\n"
    )
    assert result == {"available": True, "release": "13.2", "compiler_version": "13.2.78"}


def test_parse_tensorrt_packages_accepts_jetpack_60_libnvinfer8_runtime():
    """Hard-coding libnvinfer10 would reject the TensorRT 8.6 runtime shipped by JetPack 6.0."""
    result = _parse_tensorrt_packages(
        "libnvinfer8:arm64\tii \t8.6.2.3-1+cuda12.0\n"
        "libnvinfer10:arm64\tun \t10.0.1-1+cuda12.4\n"
    )
    assert result == {
        "available": True,
        "package": "libnvinfer8",
        "version": "8.6.2.3-1+cuda12.0",
    }


def test_parse_tensorrt_packages_rejects_known_but_uninstalled_runtime():
    """Treating dpkg's uninstalled package records as available would bypass the AGX gate."""
    with pytest.raises(
        ValueError,
        match=(
            r"no installed TensorRT runtime in dpkg-query output; raw output: "
            r"libnvinfer8:arm64\\tun \\t8\.6\.2\.3-1"
        ),
    ):
        _parse_tensorrt_packages("libnvinfer8:arm64\tun \t8.6.2.3-1\n")


def test_parse_real_jetson_release_and_jetpack_package_versions():
    """Failing to normalize real NVIDIA release text would reject a matching frozen AGX."""
    assert _parse_l4t(
        "# R36 (release), REVISION: 0.0, GCID: 35084178, BOARD: generic, EABI: aarch64\n"
    ) == {"available": True, "value": "R36.0.0"}
    assert _parse_jetpack("6.0+b106\n") == {"available": True, "value": "6.0"}


def test_failed_probe_preserves_command_error():
    """Discarding the failing command or stderr would make a fingerprint unauditable."""
    result = unavailable_probe("nvidia-smi", returncode=9, stdout="", stderr="driver unavailable")
    assert result == {
        "available": False,
        "command": "nvidia-smi",
        "returncode": 9,
        "error": "driver unavailable",
    }


def test_failed_probe_falls_back_to_stdout_then_exit_code():
    """A tool that reports failure outside stderr must still leave useful evidence."""
    assert unavailable_probe("probe", 3, "not installed\n", "")["error"] == "not installed"
    assert unavailable_probe("probe", 4, "", "")["error"] == "exit code 4"


def test_run_command_forces_stable_c_locale(monkeypatch):
    """Inheriting a localized shell would make parsers depend on operator locale."""
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert run_command(["probe", "--version"]).returncode == 0
    assert captured["env"]["LC_ALL"] == "C"
    assert captured["env"]["LANG"] == "C"
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is False


def test_capture_has_exact_schema_and_marks_failed_optional_probes(monkeypatch):
    """Adding ad-hoc keys or dropping failed optional probes would break stable evidence consumers."""
    monkeypatch.setenv("ROS_DISTRO", "humble")

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        outputs = {
            ("uname", "-m"): "x86_64\n",
            ("uname", "-r"): "6.8.0-test\n",
            ("lscpu",): "Model name: Test CPU\nCPU(s): 28\n",
            ("free", "-b"): "Mem: 67223306240 1 2 3 4 5\n",
            ("python3", "--version"): "Python 3.10.12\n",
            ("gcc", "-dumpfullversion"): "11.4.0\n",
            ("cmake", "--version"): "cmake version 3.22.1\n",
            NVIDIA_COMMAND: "NVIDIA GeForce RTX 4080 SUPER, 00000000:01:00.0, 0x270210DE, 16376, 8.9, 595.84\n",
            LSPCI_COMMAND: "0000:01:00.0 VGA [0300]: NVIDIA [10de:2702]\n",
            ("/usr/local/cuda/bin/nvcc", "--version"): "Cuda compilation tools, release 13.2, V13.2.78\n",
        }
        key = tuple(command)
        if key in outputs:
            return subprocess.CompletedProcess(command, 0, stdout=outputs[key], stderr="")
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="not available")

    document = capture_environment("train_amd64_rtx4080_super", run=run)

    assert set(document) == TOP_LEVEL_KEYS
    assert document["architecture"] == "amd64"
    assert document["cpu"] == {"available": True, "model": "Test CPU", "logical_cpus": 28}
    assert document["memory"] == {"available": True, "total_bytes": 67223306240}
    assert document["gpu"]["pci_device_id"] == "10de:2702"
    assert document["cuda"] == {
        "available": True,
        "release": "13.2",
        "compiler_version": "13.2.78",
    }
    assert document["tensorrt"] == {
        "available": False,
        "command": (
            "dpkg-query -W "
            "-f=${binary:Package}\\t${db:Status-Abbrev}\\t${Version}\\n "
            "libnvinfer[0-9]*"
        ),
        "returncode": 127,
        "error": "not available",
    }
    assert document["readiness"] == {"ready": False, "errors": []}


def test_capture_cross_checks_matching_lspci_identity(monkeypatch):
    """Skipping the independent PCI cross-check would trust one driver-reported identity."""
    monkeypatch.setenv("ROS_DISTRO", "humble")
    document = capture_environment(
        "train_amd64_rtx4080_super", run=capture_runner()
    )
    assert document["gpu"] == {
        "available": True,
        "model": "NVIDIA GeForce RTX 4080 SUPER",
        "pci_bus_id": "00000000:01:00.0",
        "pci_device_id": "10de:2702",
        "memory_total_mib": 16376,
        "compute_capability": "8.9",
        "driver_version": "595.84",
    }


def test_driver_failure_preserves_one_display_adapter_identity(monkeypatch):
    """A broken driver must not erase the sole independently visible NVIDIA adapter."""
    monkeypatch.setenv("ROS_DISTRO", "humble")
    driver_failure = subprocess.CompletedProcess(
        list(NVIDIA_COMMAND), 9, stdout="", stderr="NVIDIA driver unavailable\n"
    )
    fallback = subprocess.CompletedProcess(
        list(LSPCI_ALL_COMMAND),
        0,
        stdout=(
            "0000:00:02.0 VGA compatible controller [0300]: Intel Corporation Device [8086:a780]\n"
            "0000:01:00.0 VGA compatible controller [0300]: "
            "NVIDIA Corporation Device [10de:2702] (rev a1)\n"
        ),
        stderr="",
    )
    document = capture_environment(
        "train_amd64_rtx4080_super",
        run=capture_runner(
            {NVIDIA_COMMAND: driver_failure, LSPCI_ALL_COMMAND: fallback}
        ),
    )

    assert document["gpu"] == {
        "available": False,
        "command": NVIDIA_COMMAND_TEXT,
        "returncode": 9,
        "error": "NVIDIA driver unavailable",
        "pci_bus_id": "00000000:01:00.0",
        "pci_device_id": "10de:2702",
        "pci_evidence": {
            "available": True,
            "command": "lspci -Dnn",
            "returncode": 0,
            "source": "lspci -Dnn",
            "pci_bus_id": "00000000:01:00.0",
            "pci_device_id": "10de:2702",
            "raw_record": (
                "0000:01:00.0 VGA compatible controller [0300]: "
                "NVIDIA Corporation Device [10de:2702] (rev a1)"
            ),
        },
    }
    assert "gpu.available: expected True, got False" in validate_fingerprint(
        document, load_baseline("train_amd64_rtx4080_super")
    )


@pytest.mark.parametrize(
    ("fallback", "expected_pci_evidence"),
    [
        (
            subprocess.CompletedProcess(
                list(LSPCI_ALL_COMMAND),
                0,
                stdout=(
                    "0000:00:02.0 VGA compatible controller [0300]: "
                    "Intel Corporation Device [8086:a780]\n"
                ),
                stderr="",
            ),
            {
                "available": False,
                "command": "lspci -Dnn",
                "returncode": 0,
                "error": (
                    "parse error: expected exactly one NVIDIA display-class PCI "
                    "candidate, got 0; raw output: 0000:00:02.0 VGA compatible "
                    "controller [0300]: Intel Corporation Device [8086:a780]"
                ),
            },
        ),
        (
            subprocess.CompletedProcess(
                list(LSPCI_ALL_COMMAND),
                0,
                stdout=(
                    "0000:01:00.0 VGA compatible controller [0300]: NVIDIA Device [10de:2702]\n"
                    "0000:02:00.0 3D controller [0302]: NVIDIA Device [10de:2684]\n"
                ),
                stderr="",
            ),
            {
                "available": False,
                "command": "lspci -Dnn",
                "returncode": 0,
                "error": (
                    "parse error: expected exactly one NVIDIA display-class PCI "
                    "candidate, got 2; raw output: 0000:01:00.0 VGA compatible "
                    "controller [0300]: NVIDIA Device [10de:2702]\\n0000:02:00.0 "
                    "3D controller [0302]: NVIDIA Device [10de:2684]"
                ),
            },
        ),
        (
            subprocess.CompletedProcess(
                list(LSPCI_ALL_COMMAND), 2, stdout="", stderr="PCI inventory denied\n"
            ),
            {
                "available": False,
                "command": "lspci -Dnn",
                "returncode": 2,
                "error": "PCI inventory denied",
            },
        ),
        (
            subprocess.CompletedProcess(
                list(LSPCI_ALL_COMMAND),
                0,
                stdout=(
                    "0000:01:00.0 VGA compatible controller [0300]: "
                    "NVIDIA Corporation malformed-device-record\n"
                ),
                stderr="",
            ),
            {
                "available": False,
                "command": "lspci -Dnn",
                "returncode": 0,
                "error": (
                    "parse error: malformed NVIDIA display-class PCI record(s): "
                    "0000:01:00.0 VGA compatible controller [0300]: NVIDIA "
                    "Corporation malformed-device-record"
                ),
            },
        ),
    ],
)
def test_driver_failure_fallback_rejects_ambiguous_pci_evidence(
    monkeypatch, fallback, expected_pci_evidence
):
    """No, multiple, failed, or malformed fallback candidates must stay unavailable."""
    monkeypatch.setenv("ROS_DISTRO", "humble")
    driver_failure = subprocess.CompletedProcess(
        list(NVIDIA_COMMAND), 9, stdout="", stderr="NVIDIA driver unavailable\n"
    )
    document = capture_environment(
        "train_amd64_rtx4080_super",
        run=capture_runner(
            {NVIDIA_COMMAND: driver_failure, LSPCI_ALL_COMMAND: fallback}
        ),
    )
    assert document["gpu"] == {
        "available": False,
        "command": NVIDIA_COMMAND_TEXT,
        "returncode": 9,
        "error": "NVIDIA driver unavailable",
        "pci_evidence": expected_pci_evidence,
    }


def test_capture_preserves_failed_lspci_and_makes_gpu_unavailable(monkeypatch):
    """Discarding lspci failure would incorrectly leave GPU compute readiness true."""
    monkeypatch.setenv("ROS_DISTRO", "humble")
    failed = subprocess.CompletedProcess(
        list(LSPCI_COMMAND), 2, stdout="", stderr="unable to access PCI configuration"
    )
    document = capture_environment(
        "train_amd64_rtx4080_super",
        run=capture_runner({LSPCI_COMMAND: failed}),
    )
    assert document["gpu"] == {
        "available": False,
        "model": "NVIDIA GeForce RTX 4080 SUPER",
        "pci_bus_id": "00000000:01:00.0",
        "pci_device_id": "10de:2702",
        "memory_total_mib": 16376,
        "compute_capability": "8.9",
        "driver_version": "595.84",
        "error": "PCI cross-check failed: unable to access PCI configuration",
        "pci_evidence": {
            "available": False,
            "command": "lspci -Dnn -s 00000000:01:00.0",
            "returncode": 2,
            "error": "unable to access PCI configuration",
        },
    }
    assert "gpu.available: expected True, got False" in validate_fingerprint(
        document, load_baseline("train_amd64_rtx4080_super")
    )


def test_capture_preserves_unparseable_lspci_and_makes_gpu_unavailable(monkeypatch):
    """Accepting lspci output without a PCI ID would turn a missing cross-check into success."""
    monkeypatch.setenv("ROS_DISTRO", "humble")
    malformed = subprocess.CompletedProcess(
        list(LSPCI_COMMAND), 0, stdout="unrecognized PCI record\n", stderr=""
    )
    document = capture_environment(
        "train_amd64_rtx4080_super",
        run=capture_runner({LSPCI_COMMAND: malformed}),
    )
    assert document["gpu"] == {
        "available": False,
        "model": "NVIDIA GeForce RTX 4080 SUPER",
        "pci_bus_id": "00000000:01:00.0",
        "pci_device_id": "10de:2702",
        "memory_total_mib": 16376,
        "compute_capability": "8.9",
        "driver_version": "595.84",
        "error": (
            "PCI cross-check failed: parse error: lspci output did not contain "
            "a vendor/device ID; raw output: unrecognized PCI record"
        ),
        "pci_evidence": {
            "available": False,
            "command": "lspci -Dnn -s 00000000:01:00.0",
            "returncode": 0,
            "error": (
                "parse error: lspci output did not contain a vendor/device ID; "
                "raw output: unrecognized PCI record"
            ),
        },
    }
    assert "gpu.available: expected True, got False" in validate_fingerprint(
        document, load_baseline("train_amd64_rtx4080_super")
    )


def test_capture_rejects_lspci_identity_mismatch(monkeypatch):
    """A disagreement between driver and PCI inventory must not pass GPU readiness."""
    monkeypatch.setenv("ROS_DISTRO", "humble")
    mismatch = subprocess.CompletedProcess(
        list(LSPCI_COMMAND),
        0,
        stdout="0000:01:00.0 VGA controller [0300]: NVIDIA Device [10de:2684]\n",
        stderr="",
    )
    document = capture_environment(
        "train_amd64_rtx4080_super",
        run=capture_runner({LSPCI_COMMAND: mismatch}),
    )
    assert document["gpu"] == {
        "available": False,
        "model": "NVIDIA GeForce RTX 4080 SUPER",
        "pci_bus_id": "00000000:01:00.0",
        "pci_device_id": "10de:2702",
        "memory_total_mib": 16376,
        "compute_capability": "8.9",
        "driver_version": "595.84",
        "error": (
            "PCI cross-check failed: PCI device ID mismatch: "
            "nvidia-smi reported '10de:2702', "
            "lspci reported '10de:2684'"
        ),
        "pci_evidence": {
            "available": True,
            "command": "lspci -Dnn -s 00000000:01:00.0",
            "returncode": 0,
            "source": "lspci -Dnn -s 00000000:01:00.0",
            "pci_bus_id": "00000000:01:00.0",
            "pci_device_id": "10de:2684",
            "raw_record": (
                "0000:01:00.0 VGA controller [0300]: NVIDIA Device [10de:2684]"
            ),
        },
    }
    assert "gpu.available: expected True, got False" in validate_fingerprint(
        document, load_baseline("train_amd64_rtx4080_super")
    )


def test_capture_preserves_failed_tensorrt_package_query(monkeypatch):
    """A failed wildcard package query must retain the exact dpkg diagnostic."""
    monkeypatch.setenv("ROS_DISTRO", "humble")
    failed = subprocess.CompletedProcess(
        list(TENSORRT_COMMAND),
        1,
        stdout="",
        stderr="dpkg-query: no packages found matching libnvinfer[0-9]*",
    )
    document = capture_environment(
        "train_amd64_rtx4080_super",
        run=capture_runner({TENSORRT_COMMAND: failed}),
    )
    assert document["tensorrt"] == {
        "available": False,
        "command": (
            "dpkg-query -W "
            "-f=${binary:Package}\\t${db:Status-Abbrev}\\t${Version}\\n "
            "libnvinfer[0-9]*"
        ),
        "returncode": 1,
        "error": "dpkg-query: no packages found matching libnvinfer[0-9]*",
    }


def test_capture_preserves_uninstalled_tensorrt_package_record(monkeypatch):
    """A dpkg record without installed status must remain visible in failure evidence."""
    monkeypatch.setenv("ROS_DISTRO", "humble")
    uninstalled = subprocess.CompletedProcess(
        list(TENSORRT_COMMAND),
        0,
        stdout="libnvinfer8:arm64\tun \t8.6.2.3-1\n",
        stderr="",
    )
    document = capture_environment(
        "train_amd64_rtx4080_super",
        run=capture_runner({TENSORRT_COMMAND: uninstalled}),
    )
    assert document["tensorrt"] == {
        "available": False,
        "command": (
            "dpkg-query -W "
            "-f=${binary:Package}\\t${db:Status-Abbrev}\\t${Version}\\n "
            "libnvinfer[0-9]*"
        ),
        "returncode": 0,
        "error": (
            "parse error: no installed TensorRT runtime in dpkg-query output; "
            "raw output: libnvinfer8:arm64\\tun \\t8.6.2.3-1"
        ),
    }


def test_jetpack_60_tensorrt8_package_satisfies_agx_readiness():
    """An installed libnvinfer8 runtime on the frozen AGX must satisfy TensorRT readiness."""
    document = valid_agx_fingerprint()
    assert validate_fingerprint(
        document, load_baseline("deploy_agx_orin_r36")
    ) == []


def test_agx_capture_normalizes_only_audited_64gb_module(monkeypatch):
    """Model, P3701-0005 SKU, and visible RAM must jointly establish the canonical value."""
    monkeypatch.setenv("ROS_DISTRO", "humble")
    document = capture_environment(
        "deploy_agx_orin_r36", run=agx_capture_runner()
    )
    assert document["device_model"] == valid_agx_identity()
    assert validate_fingerprint(
        document, load_baseline("deploy_agx_orin_r36")
    ) == []


@pytest.mark.parametrize("module_sku", ["0000", "0004"])
def test_agx_capture_rejects_non_64gb_p3701_skus(monkeypatch, module_sku):
    """A nearby AGX Orin module SKU must never be relabeled as the 64 GB module."""
    monkeypatch.setenv("ROS_DISTRO", "humble")
    tnspec = f"3701-500-{module_sku}-K.2-1-1-jetson-agx-orin-devkit-"
    module_result = subprocess.CompletedProcess(
        list(NV_BOOT_CONTROL_COMMAND),
        0,
        stdout=f"TNSPEC {tnspec}\n",
        stderr="",
    )
    document = capture_environment(
        "deploy_agx_orin_r36",
        run=agx_capture_runner({NV_BOOT_CONTROL_COMMAND: module_result}),
    )
    assert document["device_model"] == {
        "available": False,
        "command": "AGX identity validation",
        "returncode": 0,
        "error": f"module SKU must be P3701-0005, got P3701-{module_sku}",
        "raw_model": AGX_MODEL,
        "model_source": "/proc/device-tree/model",
        "module_sku": f"P3701-{module_sku}",
        "tnspec": tnspec,
        "module_source": "/etc/nv_boot_control.conf",
        "observed_memory_bytes": AGX_MEMORY_BYTES,
        "minimum_memory_bytes": AGX_MINIMUM_MEMORY_BYTES,
        "memory_source": "free -b",
    }
    assert "device_model.available: expected True, got False" in validate_fingerprint(
        document, load_baseline("deploy_agx_orin_r36")
    )


def test_agx_capture_rejects_missing_tnspec_evidence(monkeypatch):
    """An unreadable nv_boot_control file must keep the model evidence but fail identity."""
    monkeypatch.setenv("ROS_DISTRO", "humble")
    missing = subprocess.CompletedProcess(
        list(NV_BOOT_CONTROL_COMMAND), 1, stdout="", stderr="file not found\n"
    )
    document = capture_environment(
        "deploy_agx_orin_r36",
        run=agx_capture_runner({NV_BOOT_CONTROL_COMMAND: missing}),
    )
    assert document["device_model"] == {
        "available": False,
        "command": "cat /etc/nv_boot_control.conf",
        "returncode": 1,
        "error": "file not found",
        "raw_model": AGX_MODEL,
        "model_source": "/proc/device-tree/model",
        "observed_memory_bytes": AGX_MEMORY_BYTES,
        "minimum_memory_bytes": AGX_MINIMUM_MEMORY_BYTES,
        "memory_source": "free -b",
    }
    assert "device_model.available: expected True, got False" in validate_fingerprint(
        document, load_baseline("deploy_agx_orin_r36")
    )


def test_agx_capture_rejects_low_os_visible_memory(monkeypatch):
    """A valid 64 GB SKU with 32 GB-like visible RAM must not pass the release identity."""
    monkeypatch.setenv("ROS_DISTRO", "humble")
    low_memory = 31 * 1024**3
    result = subprocess.CompletedProcess(
        ["free", "-b"],
        0,
        stdout=f"Mem: {low_memory} 1 2 3 4 5\n",
        stderr="",
    )
    document = capture_environment(
        "deploy_agx_orin_r36",
        run=agx_capture_runner({("free", "-b"): result}),
    )
    assert document["device_model"] == {
        "available": False,
        "command": "AGX identity validation",
        "returncode": 0,
        "error": (
            f"OS-visible memory must be at least {AGX_MINIMUM_MEMORY_BYTES} bytes, "
            f"got {low_memory}"
        ),
        "raw_model": AGX_MODEL,
        "model_source": "/proc/device-tree/model",
        "module_sku": "P3701-0005",
        "tnspec": AGX_TNSPEC,
        "module_source": "/etc/nv_boot_control.conf",
        "observed_memory_bytes": low_memory,
        "minimum_memory_bytes": AGX_MINIMUM_MEMORY_BYTES,
        "memory_source": "free -b",
    }
    assert "device_model.available: expected True, got False" in validate_fingerprint(
        document, load_baseline("deploy_agx_orin_r36")
    )


def test_training_readiness_requires_exact_gpu_and_cuda():
    """Accepting a nearby GPU SKU would make training evidence non-reproducible."""
    baseline = load_baseline("train_amd64_rtx4080_super")
    fingerprint = valid_training_fingerprint()
    assert validate_fingerprint(fingerprint, baseline) == []
    fingerprint["gpu"]["model"] = "NVIDIA GeForce RTX 4080"
    assert validate_fingerprint(fingerprint, baseline) == [
        "gpu.model: expected 'NVIDIA GeForce RTX 4080 SUPER', got 'NVIDIA GeForce RTX 4080'"
    ]


def test_training_readiness_requires_driver_and_cuda_availability():
    """A matching GPU label without a driver or CUDA compiler is not training-ready."""
    fingerprint = valid_training_fingerprint()
    fingerprint["gpu"]["driver_version"] = ""
    fingerprint["cuda"] = unavailable_probe("nvcc", 127, "", "not found")
    assert validate_fingerprint(
        fingerprint, load_baseline("train_amd64_rtx4080_super")
    ) == [
        "gpu.driver_version: expected a non-empty value, got ''",
        "cuda.available: expected True, got False",
    ]


@pytest.mark.parametrize(
    ("path", "value", "expected_error"),
    [
        (("os", "name"), "Debian GNU/Linux", "os.name: expected 'Ubuntu', got 'Debian GNU/Linux'"),
        (("os", "version_id"), "24.04", "os.version_id: expected '22.04', got '24.04'"),
        (("architecture",), "aarch64", "architecture: expected 'amd64', got 'aarch64'"),
        (("ros", "distro"), "jazzy", "ros.distro: expected 'humble', got 'jazzy'"),
        (("python", "version"), "3.11.9", "python.version: expected '3.10', got '3.11'"),
        (("gpu", "pci_device_id"), "10de:2684", "gpu.pci_device_id: expected '10de:2702', got '10de:2684'"),
    ],
)
def test_training_readiness_rejects_core_baseline_drift(path, value, expected_error):
    """Any frozen training identity drift must produce its precise readiness error."""
    document = valid_training_fingerprint()
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert validate_fingerprint(
        document, load_baseline("train_amd64_rtx4080_super")
    ) == [expected_error]


def test_training_profile_allows_tensorrt_to_be_unavailable():
    """Treating deployment-only TensorRT as required would incorrectly block PPO training."""
    fingerprint = valid_training_fingerprint()
    fingerprint["tensorrt"] = unavailable_probe(
        "dpkg-query libnvinfer", 1, "", "not installed"
    )
    assert validate_fingerprint(
        fingerprint, load_baseline("train_amd64_rtx4080_super")
    ) == []


def test_agx_readiness_requires_device_versions_cuda_and_tensorrt():
    """An AGX identity alone must not pass the native TensorRT release gate."""
    document = valid_agx_fingerprint()
    baseline = load_baseline("deploy_agx_orin_r36")
    assert validate_fingerprint(document, baseline) == []

    document["tensorrt"] = unavailable_probe("dpkg-query", 1, "", "not installed")
    assert validate_fingerprint(document, baseline) == [
        "tensorrt.available: expected True, got False"
    ]


def test_agx_readiness_rejects_unavailable_identity_with_stale_values():
    """Stale matching values must not hide failed AGX identity probes."""
    document = valid_agx_fingerprint()
    document.update(
        {
            "device_model": {
                "available": False,
                "value": "Jetson AGX Orin 64GB",
            },
            "l4t": {"available": False, "value": "R36.0.0"},
            "jetpack": {"available": False, "value": "6.0"},
        }
    )

    assert validate_fingerprint(
        document, load_baseline("deploy_agx_orin_r36")
    ) == [
        "device_model.available: expected True, got False",
        "l4t.available: expected True, got False",
        "jetpack.available: expected True, got False",
    ]


@pytest.mark.parametrize(
    ("path", "value", "expected_error"),
    [
        (("architecture",), "amd64", "architecture: expected 'aarch64', got 'amd64'"),
        (("device_model", "value"), "Jetson Orin Nano", "device_model.value: expected 'Jetson AGX Orin 64GB', got 'Jetson Orin Nano'"),
        (("l4t", "value"), "R35.4.1", "l4t.value: expected 'R36.0.0', got 'R35.4.1'"),
        (("jetpack", "value"), "5.1", "jetpack.value: expected '6.0', got '5.1'"),
        (("cuda",), unavailable_probe("nvcc", 127, "", "not found"), "cuda.available: expected True, got False"),
    ],
)
def test_agx_readiness_rejects_core_release_drift(path, value, expected_error):
    """An AGX with wrong identity or missing CUDA must not pass the device release gate."""
    document = valid_agx_fingerprint()
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert validate_fingerprint(
        document, load_baseline("deploy_agx_orin_r36")
    ) == [expected_error]


def test_write_fingerprint_is_sorted_utf8_json_with_terminal_newline(tmp_path):
    """Non-deterministic or unterminated JSON would weaken reviewable evidence diffs."""
    output = tmp_path / "nested/fingerprint.json"
    write_fingerprint({"z": "月", "a": 1}, output)
    raw = output.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.decode("utf-8") == '{\n  "a": 1,\n  "z": "月"\n}\n'


def test_cli_writes_actual_mismatch_before_returning_one(tmp_path, monkeypatch):
    """Returning failure before persistence would discard the evidence needed to diagnose drift."""
    baseline_root = tmp_path / "platform"
    profile_dir = baseline_root / "train_amd64_rtx4080_super"
    profile_dir.mkdir(parents=True)
    baseline = copy.deepcopy(load_baseline("train_amd64_rtx4080_super"))
    baseline["gpu_model"] = "NVIDIA GeForce RTX 4090"
    (profile_dir / "baseline.yaml").write_text(
        yaml.safe_dump(baseline, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        "tools.capture_environment.capture_environment",
        lambda profile: valid_training_fingerprint(),
    )
    output = tmp_path / "fingerprints/actual.json"

    returncode = main(
        [
            "--profile",
            "train_amd64_rtx4080_super",
            "--output",
            str(output),
            "--baseline-root",
            str(baseline_root),
        ]
    )

    assert returncode == 1
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["gpu"]["model"] == "NVIDIA GeForce RTX 4080 SUPER"
    assert written["readiness"] == {
        "ready": False,
        "errors": [
            "gpu.model: expected 'NVIDIA GeForce RTX 4090', got 'NVIDIA GeForce RTX 4080 SUPER'"
        ],
    }


def test_cli_writes_ready_fingerprint_before_returning_zero(tmp_path, monkeypatch):
    """A matching capture must be persisted with ready=true before a successful exit."""
    monkeypatch.setattr(
        "tools.capture_environment.capture_environment",
        lambda profile: valid_training_fingerprint(),
    )
    output = tmp_path / "fingerprints/ready.json"

    assert main(
        [
            "--profile",
            "train_amd64_rtx4080_super",
            "--output",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["readiness"] == {
        "ready": True,
        "errors": [],
    }


def test_cli_refuses_output_inside_git_repository(monkeypatch):
    """Allowing a live fingerprint in source control would violate repository artifact boundaries."""
    output = REPOSITORY_ROOT / "forbidden-fingerprint.json"
    monkeypatch.setattr(
        "tools.capture_environment.capture_environment",
        lambda profile: valid_training_fingerprint(),
    )

    assert main(
        [
            "--profile",
            "train_amd64_rtx4080_super",
            "--output",
            str(output),
        ]
    ) == 2
    assert not output.exists()

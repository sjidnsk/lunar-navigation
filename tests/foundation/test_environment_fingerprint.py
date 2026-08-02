"""Behavioral tests for auditable platform environment fingerprints."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import yaml

from tools.capture_environment import (
    DEFAULT_BASELINE_ROOT,
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
            (
                "nvidia-smi",
                "--query-gpu=name,pci.bus_id,pci.device_id,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ): "NVIDIA GeForce RTX 4080 SUPER, 00000000:01:00.0, 0x270210DE, 16376, 8.9, 595.84\n",
            ("lspci", "-Dnns", "00000000:01:00.0"): "0000:01:00.0 VGA [0300]: NVIDIA [10de:2702]\n",
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
        "command": "dpkg-query -W -f=${Version} libnvinfer10",
        "returncode": 127,
        "error": "not available",
    }
    assert document["readiness"] == {"ready": False, "errors": []}


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
    document = valid_training_fingerprint()
    document.update(
        {
            "profile": "deploy_agx_orin_r36",
            "architecture": "aarch64",
            "device_model": {"available": True, "value": "Jetson AGX Orin 64GB"},
            "l4t": {"available": True, "value": "R36.0.0"},
            "jetpack": {"available": True, "value": "6.0"},
            "tensorrt": {"available": True, "value": "10.0.1"},
        }
    )
    baseline = load_baseline("deploy_agx_orin_r36")
    assert validate_fingerprint(document, baseline) == []

    document["tensorrt"] = unavailable_probe("dpkg-query", 1, "", "not installed")
    assert validate_fingerprint(document, baseline) == [
        "tensorrt.available: expected True, got False"
    ]


def test_agx_readiness_rejects_unavailable_identity_with_stale_values():
    """Stale matching values must not hide failed AGX identity probes."""
    document = valid_training_fingerprint()
    document.update(
        {
            "profile": "deploy_agx_orin_r36",
            "architecture": "aarch64",
            "device_model": {
                "available": False,
                "value": "Jetson AGX Orin 64GB",
            },
            "l4t": {"available": False, "value": "R36.0.0"},
            "jetpack": {"available": False, "value": "6.0"},
            "tensorrt": {"available": True, "value": "10.0.1"},
        }
    )

    assert validate_fingerprint(
        document, load_baseline("deploy_agx_orin_r36")
    ) == [
        "device_model.available: expected True, got False",
        "l4t.available: expected True, got False",
        "jetpack.available: expected True, got False",
    ]


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

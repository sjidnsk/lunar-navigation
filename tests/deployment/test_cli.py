from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from deployment.luna_runtime.cli import run_cli
from deployment.luna_runtime.host import HostFacts


AMD64_FACTS = HostFacts(
    os_id="ubuntu", os_version="22.04", architecture="x86_64", ros_distro="humble"
)
ORIN_FACTS = HostFacts(
    os_id="ubuntu",
    os_version="22.04",
    architecture="aarch64",
    ros_distro="humble",
    l4t="R36.2.0",
    jetpack="6.0",
)


class FakeRunner:
    def __init__(self, installed_packages: set[str] | None = None) -> None:
        self.installed_packages = installed_packages or set()
        self.commands: list[tuple[str, ...]] = []

    def is_installed(self, package: str) -> bool:
        return package in self.installed_packages

    def run(self, command: tuple[str, ...]) -> None:
        self.commands.append(command)
        if command[:4] == ("sudo", "apt-get", "install", "--no-install-recommends"):
            self.installed_packages.update(command[4:])

    def package_versions(self, packages: tuple[str, ...]) -> dict[str, str]:
        return {package: "1.0" for package in packages if package in self.installed_packages}


def test_init_writes_only_runtime_paths_and_never_overwrites(tmp_path: Path) -> None:
    home = tmp_path / "home"
    result = run_cli(
        ["init", "--profile", "ubuntu22-humble-amd64"], facts=AMD64_FACTS, home=home
    )
    assert result.exit_code == 0
    assert (home / ".config/luna/runtime.yaml").exists()
    assert not (tmp_path / "build").exists()

    repeated = run_cli(
        ["init", "--profile", "ubuntu22-humble-amd64"], facts=AMD64_FACTS, home=home
    )
    assert repeated.exit_code == 2
    assert repeated.payload["reason"] == "CONFIG_EXISTS"


def test_luna_script_initializes_from_outside_repository(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    home = tmp_path / "luna-home"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "luna"),
            "init",
            "--profile",
            "ubuntu22-humble-amd64",
        ],
        cwd=tmp_path,
        env={**os.environ, "LUNA_HOME": str(home), "ROS_DISTRO": "humble"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert (home / "config" / "runtime.yaml").is_file()


def test_doctor_reports_host_mismatch_without_running_build(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        (Path(__file__).resolve().parents[2] / "deployment/config/runtime.default.yaml").read_text(encoding="utf-8").replace(
            "ubuntu22-humble-amd64", "jetson-orin-r36"
        ),
        encoding="utf-8",
    )
    runner = FakeRunner()
    result = run_cli(["doctor", "--config", str(config)], facts=AMD64_FACTS, runner=runner)
    assert result.exit_code == 3
    assert result.payload["reasons"] == ["ARCHITECTURE_MISMATCH", "L4T_MISMATCH", "JETPACK_MISMATCH"]
    assert runner.commands == []


def test_prepare_dry_run_is_read_only_and_never_invokes_sudo(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        (Path(__file__).resolve().parents[2] / "deployment/config/runtime.default.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    runner = FakeRunner({"build-essential"})
    result = run_cli(
        ["prepare", "--dry-run", "--config", str(config)],
        facts=AMD64_FACTS,
        runner=runner,
        home=tmp_path / "home",
    )
    assert result.exit_code == 3
    assert result.payload["missing_apt_packages"]
    assert runner.commands == []
    assert not (tmp_path / "home").exists()


def test_prepare_apply_installs_only_locked_packages_then_records_manifest(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        (Path(__file__).resolve().parents[2] / "deployment/config/runtime.default.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    runner = FakeRunner()
    result = run_cli(
        ["prepare", "--apply", "--yes", "--config", str(config)],
        facts=AMD64_FACTS,
        runner=runner,
        home=tmp_path / "home",
    )
    assert result.exit_code == 0
    assert ("sudo", "apt-get", "update") in runner.commands
    installed = next(command[4:] for command in runner.commands if command[:4] == ("sudo", "apt-get", "install", "--no-install-recommends"))
    assert "ros-humble-nav2-core" not in installed
    manifest = json.loads((tmp_path / "home/.local/share/luna/dev/environment-manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile_id"] == "ubuntu22-humble-amd64"
    assert manifest["apt_packages"]


def test_orin_prepare_never_attempts_to_install_nvidia_cuda_or_tensorrt(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        (Path(__file__).resolve().parents[2] / "deployment/config/runtime.default.yaml").read_text(encoding="utf-8").replace(
            "ubuntu22-humble-amd64", "jetson-orin-r36"
        ),
        encoding="utf-8",
    )
    runner = FakeRunner()
    result = run_cli(
        ["prepare", "--apply", "--yes", "--config", str(config)],
        facts=ORIN_FACTS,
        runner=runner,
        home=tmp_path / "home",
    )
    assert result.exit_code == 0
    invoked = " ".join(" ".join(command) for command in runner.commands).lower()
    assert all(token not in invoked for token in ("nvidia", "cuda", "tensorrt", "flash"))

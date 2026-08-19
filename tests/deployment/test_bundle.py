from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from deployment.luna_runtime.bundle import BundleRequest, build_bundle
from deployment.luna_runtime.cli import run_cli
from deployment.luna_runtime.host import HostFacts


REPO_ROOT = Path(__file__).resolve().parents[2]


def _names(archive: Path) -> list[str]:
    with tarfile.open(archive, "r:gz") as tar:
        return sorted(member.name for member in tar.getmembers() if member.isfile())


def _manifest(archive: Path) -> dict[str, object]:
    with tarfile.open(archive, "r:gz") as tar:
        member = next(item for item in tar.getmembers() if item.name.endswith("release-manifest.json"))
        return json.loads(tar.extractfile(member).read())


def test_bundle_has_only_allowlisted_current_runtime_source(tmp_path: Path) -> None:
    result = build_bundle(BundleRequest(REPO_ROOT, "HEAD", "ubuntu22-humble-amd64", tmp_path))
    names = _names(result.archive)
    root = "lunar-runtime-ubuntu22-humble-amd64-src/"
    assert root + "luna" in names
    assert root + "luna_runtime/cli.py" in names
    assert root + "config/runtime.schema.json" in names
    assert root + "profiles/ubuntu22-humble-amd64.environment.yaml" in names
    assert root + "runtime_source_allowlist.yaml" in names
    assert not any("profiles/jetson-orin-r36" in name for name in names)
    assert all("/training/" not in name and "/test/" not in name and "/tests/" not in name for name in names)
    assert not any("lunar_planner_training_bridge" in name for name in names)


def test_bundles_are_deterministic_and_profiles_are_distinct(tmp_path: Path) -> None:
    amd_first = build_bundle(BundleRequest(REPO_ROOT, "HEAD", "ubuntu22-humble-amd64", tmp_path)).archive
    amd_second = build_bundle(BundleRequest(REPO_ROOT, "HEAD", "ubuntu22-humble-amd64", tmp_path / "again")).archive
    orin = build_bundle(BundleRequest(REPO_ROOT, "HEAD", "jetson-orin-r36", tmp_path)).archive
    assert hashlib.sha256(amd_first.read_bytes()).hexdigest() == hashlib.sha256(amd_second.read_bytes()).hexdigest()
    assert _manifest(amd_first)["source_commit"] == _manifest(orin)["source_commit"]
    assert _manifest(amd_first)["target_profile"] != _manifest(orin)["target_profile"]


def test_luna_bundle_writes_one_archive_without_runtime_initialization(tmp_path: Path) -> None:
    result = run_cli(
        ["bundle", "--target", "ubuntu22-humble-amd64", "--output", str(tmp_path)],
        facts=HostFacts("ubuntu", "22.04", "x86_64", "humble"),
        repo_root=REPO_ROOT,
        home=tmp_path / "home",
    )
    assert result.exit_code == 0
    assert Path(result.payload["archive"]).is_file()
    assert not (tmp_path / "home").exists()


def test_bundle_root_contains_exactly_two_markdown_documents(tmp_path: Path) -> None:
    result = build_bundle(BundleRequest(REPO_ROOT, "HEAD", "ubuntu22-humble-amd64", tmp_path))
    markdown = [name for name in _names(result.archive) if name.endswith(".md")]
    assert markdown == [
        "lunar-runtime-ubuntu22-humble-amd64-src/COMMANDS.md",
        "lunar-runtime-ubuntu22-humble-amd64-src/README.md",
    ]

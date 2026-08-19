from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from deployment.luna_runtime.bundle import BundleRequest, build_bundle


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

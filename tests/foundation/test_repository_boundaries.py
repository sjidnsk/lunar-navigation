from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from check_repository_boundaries import check_repository


def _run_git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )


def _initialize_repository(root: Path) -> None:
    _run_git(root, "init", "-q")
    _run_git(root, "config", "user.email", "tests@example.invalid")
    _run_git(root, "config", "user.name", "Repository Boundary Tests")


def test_rejects_nested_git_and_windows_absolute_paths(tmp_path: Path) -> None:
    """Removing either nested-repository or source-path validation must fail this test."""
    (tmp_path / "nested" / ".git").mkdir(parents=True)
    source = tmp_path / "bad.py"
    source.write_text('MODEL = "D:/models/policy.onnx"\n', encoding="utf-8")

    errors = check_repository(tmp_path)

    assert any("nested Git" in error for error in errors)
    assert any("Windows absolute path" in error for error in errors)


def test_rejects_nested_git_worktree_file(tmp_path: Path) -> None:
    """Removing nested-worktree detection must fail this test."""
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / ".git").write_text("gitdir: ../.git/worktrees/nested\n", encoding="utf-8")

    errors = check_repository(tmp_path)

    assert any("nested Git" in error for error in errors)


def test_rejects_nested_git_below_an_ignored_directory(tmp_path: Path) -> None:
    """Skipping ignored directories during Git discovery must fail this test."""
    (tmp_path / "build" / "third_party" / ".git").mkdir(parents=True)

    errors = check_repository(tmp_path)

    assert any("nested Git" in error and "build/third_party/.git" in error for error in errors)


def test_rejects_tracked_gitlinks_forbidden_artifacts_and_nonexecutable_shell_scripts(
    tmp_path: Path,
) -> None:
    """Removing tracked-file checks must fail this test."""
    _initialize_repository(tmp_path)
    (tmp_path / "model.pt").write_text("not a model", encoding="utf-8")
    script = tmp_path / "run.sh"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    _run_git(tmp_path, "add", "model.pt", "run.sh")
    _run_git(tmp_path, "update-index", "--chmod=-x", "run.sh")
    _run_git(
        tmp_path,
        "update-index",
        "--add",
        "--cacheinfo",
        "160000,0123456789012345678901234567890123456789,legacy-link",
    )

    errors = check_repository(tmp_path)

    assert any("gitlink" in error for error in errors)
    assert any("forbidden artifact" in error for error in errors)
    assert any("shell script" in error for error in errors)


def test_rejects_tracked_models_training_data_artifact_directories_and_large_files(
    tmp_path: Path,
) -> None:
    """Removing artifact suffix, directory, or size checks must fail this test."""
    _initialize_repository(tmp_path)
    (tmp_path / "policy.onnx").write_text("model", encoding="utf-8")
    (tmp_path / "checkpoint.ckpt").write_text("checkpoint", encoding="utf-8")
    (tmp_path / "weights.safetensors").write_text("weights", encoding="utf-8")
    (tmp_path / "training.npz").write_text("data", encoding="utf-8")
    output = tmp_path / "training-output"
    output.mkdir()
    (output / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "large.bin").write_bytes(b"x" * (1_048_576 + 1))
    _run_git(tmp_path, "add", ".")

    errors = check_repository(tmp_path)

    assert any("forbidden artifact" in error and "policy.onnx" in error for error in errors)
    assert any("forbidden artifact" in error and "checkpoint.ckpt" in error for error in errors)
    assert any("forbidden artifact" in error and "weights.safetensors" in error for error in errors)
    assert any("forbidden artifact" in error and "training.npz" in error for error in errors)
    assert any("forbidden artifact directory" in error and "training-output" in error for error in errors)
    assert any("size limit" in error and "large.bin" in error for error in errors)


def test_rejects_windows_paths_in_yaml_and_json_configuration(tmp_path: Path) -> None:
    """Removing YAML or JSON configuration scanning must fail this test."""
    (tmp_path / "runtime.yaml").write_text(
        "model_path: D:/models/policy.onnx\n",
        encoding="utf-8",
    )
    (tmp_path / "runtime.json").write_text(
        '{"cache_path": "C:\\\\Users\\\\operator\\\\cache"}\n',
        encoding="utf-8",
    )

    errors = check_repository(tmp_path)

    assert any("Windows absolute path" in error and "runtime.yaml" in error for error in errors)
    assert any("Windows absolute path" in error and "runtime.json" in error for error in errors)


def test_allows_windows_paths_in_approved_migration_inventories(tmp_path: Path) -> None:
    """Removing the narrow migration-inventory exemption must fail this test."""
    migration = tmp_path / "migration"
    migration.mkdir()
    for name in ("source_inventory.yaml", "fixture_inventory.yaml"):
        (migration / name).write_text(
            "captured_from: C:/legacy/lunar-navigation\n",
            encoding="utf-8",
        )

    assert check_repository(tmp_path) == []


def test_allows_documented_windows_path_counterexamples(tmp_path: Path) -> None:
    """Treating documentation examples as source violations must fail this test."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "portability.md").write_text(
        "Counterexample: do not hard-code D:/models/policy.onnx in source code.\n",
        encoding="utf-8",
    )

    assert check_repository(tmp_path) == []


def test_rejects_second_lunar_navigation_msgs_package(tmp_path: Path) -> None:
    canonical = tmp_path / "ros2_ws/src/lunar_navigation_msgs"
    duplicate = tmp_path / "vendor/lunar_navigation_msgs"
    canonical.mkdir(parents=True)
    duplicate.mkdir(parents=True)
    package_xml = "<package format='3'><name>lunar_navigation_msgs</name></package>"
    (canonical / "package.xml").write_text(package_xml, encoding="utf-8")
    (duplicate / "package.xml").write_text(package_xml, encoding="utf-8")
    assert any("duplicate lunar_navigation_msgs" in error for error in check_repository(tmp_path))


def test_rejects_noncanonical_lunar_navigation_msgs_package(tmp_path: Path) -> None:
    package = tmp_path / "vendor/lunar_navigation_msgs"
    package.mkdir(parents=True)
    (package / "package.xml").write_text(
        "<package format='3'><name>lunar_navigation_msgs</name></package>",
        encoding="utf-8",
    )

    assert any("noncanonical lunar_navigation_msgs" in error for error in check_repository(tmp_path))


def test_rejects_unapproved_provisional_interface(tmp_path: Path) -> None:
    package = tmp_path / "ros2_ws/src/lunar_navigation_msgs"
    (package / "action").mkdir(parents=True)
    (package / "action/Unexpected.action").write_text(
        "string input\n---\nbool ok\n",
        encoding="utf-8",
    )
    assert any("unapproved lunar_navigation_msgs interface" in error for error in check_repository(tmp_path))

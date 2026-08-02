from __future__ import annotations

import hashlib
import subprocess

import pytest

import tools.create_source_inventory as source_inventory
from tools.create_source_inventory import create_inventory, source_file_selections


def test_inventory_records_each_repository_and_sha256(tmp_path):
    repository = tmp_path / "root"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/legacy.git"],
        cwd=repository,
        check=True,
    )
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )

    output = create_inventory(
        repositories={"legacy_root": repository},
        selected_files=("README.md",),
    )

    assert output["schema_version"] == "lunar-migration-source-inventory/v1"
    assert len(output["repositories"]["legacy_root"]["commit"]) == 40
    assert len(output["files"][0]["sha256"]) == 64
    assert "absolute_path" not in output["files"][0]


def test_inventory_hashes_head_blob_not_clean_crlf_checkout(tmp_path):
    repository = tmp_path / "root"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/legacy.git"],
        cwd=repository,
        check=True,
    )
    (repository / ".gitattributes").write_text("README.md text\n", encoding="utf-8")
    (repository / "README.md").write_bytes(b"fixture\n")
    subprocess.run(["git", "add", ".gitattributes", "README.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=repository, check=True)
    (repository / "README.md").unlink()
    subprocess.run(["git", "checkout", "--", "README.md"], cwd=repository, check=True)

    assert (repository / "README.md").read_bytes() == b"fixture\r\n"
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout == ""

    output = create_inventory(
        repositories={"legacy_root": repository}, selected_files=("README.md",)
    )

    assert output["files"][0]["sha256"] == hashlib.sha256(b"fixture\n").hexdigest()
    assert output["files"][0]["size_bytes"] == len(b"fixture\n")


def test_inventory_rejects_ignored_selected_file_absent_from_head(tmp_path):
    repository = tmp_path / "root"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/legacy.git"],
        cwd=repository,
        check=True,
    )
    (repository / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "ignore fixture",
        ],
        cwd=repository,
        check=True,
    )
    (repository / "ignored.txt").write_text("not committed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="HEAD"):
        create_inventory(
            repositories={"legacy_root": repository}, selected_files=("ignored.txt",)
        )


def test_inventory_rejects_assume_unchanged_selected_file(tmp_path):
    repository = tmp_path / "root"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/legacy.git"],
        cwd=repository,
        check=True,
    )
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "update-index", "--assume-unchanged", "README.md"],
        cwd=repository,
        check=True,
    )
    (repository / "README.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="assume-unchanged"):
        create_inventory(
            repositories={"legacy_root": repository}, selected_files=("README.md",)
        )


def test_source_selection_includes_migration_categories_and_excludes_governance(tmp_path):
    planner = tmp_path / "path-planner"
    planner.mkdir()
    subprocess.run(["git", "init"], cwd=planner, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/planner.git"],
        cwd=planner,
        check=True,
    )
    selected_paths = (
        "cpp/include/lunar_path_planner/v3/wheel/wheel_planner.hpp",
        "cpp/include/lunar_path_planner/v3/legged/legged_planner.hpp",
        "cpp/include/lunar_path_planner/v3/hopper/hopper_planner.hpp",
        "cpp/src/wheel/wheel_planner.cpp",
        "cpp/src/legged/legged_planner.cpp",
        "cpp/src/hopper/hopper_planner.cpp",
        "cpp/tests/integration/wheel/wheel_planner_test.cpp",
        "cpp/src/bindings/python_projection.cpp",
        "cpp/include/lunar_path_planner/v3/bindings/python_projection.hpp",
        "cpp/benchmarks/wheel_planner_benchmark.cpp",
    )
    for relative_path in selected_paths:
        path = planner / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=planner, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "planner fixture",
        ],
        cwd=planner,
        check=True,
    )

    selections = source_file_selections({"path_planner": planner})

    assert "path_planner:cpp/include/lunar_path_planner/v3/wheel/wheel_planner.hpp" in selections
    assert "path_planner:cpp/src/legged/legged_planner.cpp" in selections
    assert "path_planner:cpp/src/hopper/hopper_planner.cpp" in selections
    assert "path_planner:cpp/tests/integration/wheel/wheel_planner_test.cpp" in selections
    assert "path_planner:cpp/src/bindings/python_projection.cpp" not in selections
    assert "path_planner:cpp/include/lunar_path_planner/v3/bindings/python_projection.hpp" not in selections
    assert "path_planner:cpp/benchmarks/wheel_planner_benchmark.cpp" not in selections
    assert "dev_platform_constraints:src/dev_platform_constraints/path_planning/astar.py" in selections
    assert not any("scripts/run_" in selection for selection in selections)


def test_inventory_marks_legacy_python_astar_as_differential_reference(tmp_path):
    repository = tmp_path / "platform"
    source = repository / "src/dev_platform_constraints/path_planning/astar.py"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repository, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/platform.git"],
        cwd=repository,
        check=True,
    )
    source.write_text("class AStarPlanner: pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "astar fixture",
        ],
        cwd=repository,
        check=True,
    )

    output = create_inventory(
        repositories={"dev_platform_constraints": repository},
        selected_files=("src/dev_platform_constraints/path_planning/astar.py",),
    )

    assert output["files"][0]["migration_role"] == "tests/differential_reference"


def test_inventory_reads_blobs_from_captured_commit_when_head_advances(tmp_path, monkeypatch):
    repository = tmp_path / "root"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/legacy.git"],
        cwd=repository,
        check=True,
    )
    readme = repository / "README.md"
    readme.write_text("first\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "first",
        ],
        cwd=repository,
        check=True,
    )
    first_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    readme.write_text("second\n", encoding="utf-8")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-am",
            "second",
        ],
        cwd=repository,
        check=True,
    )
    second_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "checkout", "--detach", first_commit], cwd=repository, check=True)

    original_metadata = source_inventory._repository_metadata

    def capture_then_advance_head(path):
        metadata = original_metadata(path)
        subprocess.run(["git", "checkout", "--detach", second_commit], cwd=path, check=True)
        return metadata

    monkeypatch.setattr(source_inventory, "_repository_metadata", capture_then_advance_head)

    output = create_inventory(
        repositories={"legacy_root": repository}, selected_files=("README.md",)
    )

    assert output["repositories"]["legacy_root"]["commit"] == first_commit
    assert output["files"][0]["sha256"] == hashlib.sha256(b"first\n").hexdigest()


def test_inventory_resolves_unqualified_selection_from_captured_commit(tmp_path, monkeypatch):
    repository = tmp_path / "root"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/legacy.git"],
        cwd=repository,
        check=True,
    )
    readme = repository / "README.md"
    readme.write_text("first\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "first",
        ],
        cwd=repository,
        check=True,
    )
    first_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    readme.unlink()
    subprocess.run(["git", "add", "-u"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "remove README",
        ],
        cwd=repository,
        check=True,
    )
    second_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "checkout", "--detach", first_commit], cwd=repository, check=True)

    original_metadata = source_inventory._repository_metadata

    def capture_then_remove_selected_path(path):
        metadata = original_metadata(path)
        subprocess.run(["git", "checkout", "--detach", second_commit], cwd=path, check=True)
        return metadata

    monkeypatch.setattr(source_inventory, "_repository_metadata", capture_then_remove_selected_path)

    output = create_inventory(
        repositories={"legacy_root": repository}, selected_files=("README.md",)
    )

    assert output["repositories"]["legacy_root"]["commit"] == first_commit
    assert output["files"][0]["path"] == "README.md"
    assert output["files"][0]["sha256"] == hashlib.sha256(b"first\n").hexdigest()

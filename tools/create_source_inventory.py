from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any


SOURCE_SCHEMA = "lunar-migration-source-inventory/v1"
FIXTURE_SCHEMA = "lunar-migration-fixture-inventory/v1"
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

SOURCE_FILES = (
    "legacy_root:configs/ppo_highres_frontier_foundation_v1.json",
    "legacy_root:configs/platforms/v3/README.md",
    "legacy_root:configs/platforms/v3/hopper_ballistic_v3_example_v1.json",
    "legacy_root:configs/platforms/v3/legged_body_v3_example_v1.json",
    "legacy_root:configs/platforms/v3/wheeled_skid_steer_v3_example_v1.json",
    "legacy_root:src/lunar_exploration_ppo/env/terrain_proxy.py",
    "legacy_root:src/lunar_exploration_ppo/policy/cross_attention.py",
    "legacy_root:src/lunar_exploration_ppo/ppo/rollout.py",
    "legacy_root:src/lunar_exploration_ppo/ppo/trainer.py",
    "path_planner:cpp/CMakeLists.txt",
    "path_planner:cpp/include/lunar_path_planner/v3/api/planner_v3.hpp",
    "path_planner:cpp/include/lunar_path_planner/v3/contracts/planning_request.hpp",
    "path_planner:cpp/include/lunar_path_planner/v3/contracts/planning_response.hpp",
    "path_planner:cpp/src/api/planner_v3.cpp",
    "dev_platform_constraints:configs/platforms/yutu.json",
    "dev_platform_constraints:configs/platforms/yutu2.json",
    "dev_platform_constraints:src/dev_platform_constraints/core/contracts.py",
    "dev_platform_constraints:src/dev_platform_constraints/mapping/constraints.py",
    "dev_platform_constraints:src/dev_platform_constraints/path_planning/astar.py",
    "dev_platform_constraints:src/dev_platform_constraints/platforms/model.py",
)

FIXTURE_FILES = (
    "path_planner:cpp/tests/fixtures/minimal_wheeled_request.json",
    "path_planner:cpp/tests/fixtures/shared_core_case_a.expected.json",
    "path_planner:cpp/tests/fixtures/shared_core_case_a.json",
)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _normalise_relative_path(value: str) -> str:
    candidate = PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts or str(candidate) in {"", "."}:
        raise ValueError(f"selected file must be a repository-relative path: {value}")
    return candidate.as_posix()


def _resolve_selection(
    repositories: dict[str, Path], selection: str
) -> tuple[str, str]:
    repository_key, separator, file_path = selection.partition(":")
    if separator and repository_key in repositories:
        return repository_key, _normalise_relative_path(file_path)

    relative_path = _normalise_relative_path(selection)
    matching_keys = [
        key for key, repository in repositories.items() if (repository / relative_path).is_file()
    ]
    if len(matching_keys) != 1:
        raise ValueError(
            f"selected file must resolve in exactly one repository: {relative_path}"
        )
    return matching_keys[0], relative_path


def _repository_metadata(repository: Path) -> dict[str, str]:
    commit = _git(repository, "rev-parse", "HEAD").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError("repository HEAD is not a 40-character commit")
    return {"commit": commit, "origin": _git(repository, "remote", "get-url", "origin")}


def _assert_selected_file_is_clean(repository: Path, relative_path: str) -> None:
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        relative_path,
    )
    if status:
        raise ValueError(f"refusing selected dirty file: {relative_path}")


def _create_document(
    *,
    schema_version: str,
    repositories: dict[str, Path],
    selected_files: Sequence[str],
) -> dict[str, object]:
    resolved_repositories = {
        key: Path(path).resolve() for key, path in sorted(repositories.items())
    }
    if not resolved_repositories:
        raise ValueError("at least one repository is required")

    repository_records: dict[str, dict[str, str]] = {}
    for key, repository in resolved_repositories.items():
        if not repository.is_dir():
            raise ValueError(f"repository directory is missing for key: {key}")
        repository_records[key] = _repository_metadata(repository)

    file_records: list[dict[str, object]] = []
    for selection in selected_files:
        key, relative_path = _resolve_selection(resolved_repositories, selection)
        repository = resolved_repositories[key]
        file_path = (repository / relative_path).resolve()
        if not file_path.is_relative_to(repository) or not file_path.is_file():
            raise ValueError(f"selected file is missing or escapes repository: {relative_path}")
        _assert_selected_file_is_clean(repository, relative_path)
        data = file_path.read_bytes()
        file_records.append(
            {
                "repository": key,
                "path": relative_path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )

    return {
        "schema_version": schema_version,
        "repositories": repository_records,
        "files": sorted(file_records, key=lambda record: (record["repository"], record["path"])),
    }


def create_inventory(
    *, repositories: dict[str, Path], selected_files: Sequence[str]
) -> dict[str, object]:
    """Return a deterministic source inventory without local absolute paths."""
    return _create_document(
        schema_version=SOURCE_SCHEMA,
        repositories=repositories,
        selected_files=selected_files,
    )


def create_fixture_inventory(
    *, repositories: dict[str, Path], selected_files: Sequence[str]
) -> dict[str, object]:
    return _create_document(
        schema_version=FIXTURE_SCHEMA,
        repositories=repositories,
        selected_files=selected_files,
    )


def write_inventory(document: dict[str, object], output: Path) -> None:
    """Write a UTF-8, LF-only JSON document, which is valid YAML 1.2."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze approved legacy migration sources.")
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    legacy_root = arguments.legacy_root.resolve()
    repositories = {
        "legacy_root": legacy_root,
        "path_planner": legacy_root / "path-planner",
        "dev_platform_constraints": legacy_root / "dev-platform-constraints",
    }
    write_inventory(
        create_inventory(repositories=repositories, selected_files=SOURCE_FILES),
        arguments.output,
    )
    write_inventory(
        create_fixture_inventory(repositories=repositories, selected_files=FIXTURE_FILES),
        arguments.fixture_output,
    )


if __name__ == "__main__":
    main()

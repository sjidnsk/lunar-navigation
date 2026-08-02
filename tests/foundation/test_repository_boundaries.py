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


def test_allows_documented_windows_path_counterexamples(tmp_path: Path) -> None:
    """Treating documentation examples as source violations must fail this test."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "portability.md").write_text(
        "Counterexample: do not hard-code D:/models/policy.onnx in source code.\n",
        encoding="utf-8",
    )

    assert check_repository(tmp_path) == []

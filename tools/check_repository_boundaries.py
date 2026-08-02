"""Validate that lunar_navigation remains a single, portable Git repository."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


IGNORED_DIRS = {".git", "build", "install", "log", ".venv", "__pycache__"}
FORBIDDEN_SUFFIXES = {".obj", ".engine", ".pt", ".pth", ".bag", ".db3"}

_LARGE_ARTIFACT_SUFFIXES = FORBIDDEN_SUFFIXES | {".mcap"}
_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hxx",
    ".py",
    ".sh",
    ".toml",
    ".xml",
}
_WINDOWS_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|[\\/]Users[\\/])")


def _tracked_files(root: Path) -> list[tuple[str, str]]:
    """Return tracked file modes and paths, or no entries outside a Git root."""
    if not (root / ".git").exists():
        return []

    completed = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []

    entries: list[tuple[str, str]] = []
    for record in completed.stdout.split("\0"):
        if not record:
            continue
        metadata, path = record.split("\t", maxsplit=1)
        mode, _object_id, _stage = metadata.split(maxsplit=2)
        entries.append((mode, path))
    return entries


def _is_source(path: Path) -> bool:
    return path.name == "CMakeLists.txt" or path.suffix.lower() in _SOURCE_SUFFIXES


def _find_nested_git_directories(root: Path) -> list[Path]:
    nested: list[Path] = []
    for directory, names, files in os.walk(root):
        current = Path(directory)
        if current != root and ".git" in names:
            nested.append(current / ".git")
            names.remove(".git")
        if current != root and ".git" in files:
            nested.append(current / ".git")
        ignored = [name for name in names if name in IGNORED_DIRS]
        for name in ignored:
            names.remove(name)
    return nested


def _find_windows_paths(root: Path) -> list[Path]:
    violations: list[Path] = []
    for directory, names, filenames in os.walk(root):
        names[:] = [name for name in names if name not in IGNORED_DIRS]
        for filename in filenames:
            path = Path(directory) / filename
            relative_path = path.relative_to(root)
            if (
                relative_path.parts[0] == "tests"
                or path.suffix.lower() == ".md"
                or not _is_source(path)
            ):
                continue
            try:
                contents = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if _WINDOWS_PATH.search(contents):
                violations.append(path)
    return violations


def check_repository(root: Path) -> list[str]:
    """Return repository-boundary violations; an empty list means the root is valid."""
    root = root.resolve()
    errors: list[str] = []

    for directory in _find_nested_git_directories(root):
        errors.append(f"nested Git directory: {directory.relative_to(root).as_posix()}")

    for source in _find_windows_paths(root):
        errors.append(f"Windows absolute path in source: {source.relative_to(root).as_posix()}")

    for mode, relative_path in _tracked_files(root):
        path = Path(relative_path)
        if mode == "160000":
            errors.append(f"tracked gitlink is forbidden: {relative_path}")
        if path.suffix.lower() in _LARGE_ARTIFACT_SUFFIXES:
            errors.append(f"tracked forbidden artifact: {relative_path}")
        if path.suffix.lower() == ".sh" and mode != "100755":
            errors.append(f"tracked shell script is not executable: {relative_path}")

    return sorted(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    arguments = parser.parse_args()
    errors = check_repository(arguments.root)
    if errors:
        for error in errors:
            print(f"repository boundary error: {error}")
        return 1
    print("repository boundaries: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

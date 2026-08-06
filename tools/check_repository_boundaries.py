"""Validate that lunar_navigation remains a single, portable Git repository."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import xml.etree.ElementTree as element_tree
from pathlib import Path


IGNORED_DIRS = {".git", "build", "install", "log", ".venv", "__pycache__"}
FORBIDDEN_SUFFIXES = {
    ".obj",
    ".engine",
    ".pt",
    ".pth",
    ".bag",
    ".db3",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".safetensors",
    ".tfrecord",
}
FORBIDDEN_ARTIFACT_DIRS = {
    "artifacts",
    "checkpoints",
    "datasets",
    "device-output",
    "rosbags",
    "training-output",
}
MAX_TRACKED_FILE_BYTES = 1_048_576
MIGRATION_INVENTORY_PATHS = {
    "migration/fixture_inventory.yaml",
    "migration/source_inventory.yaml",
}
PROVISIONAL_PACKAGE_PATH = Path("ros2_ws/src/lunar_navigation_msgs")
ALLOWED_PROVISIONAL_INTERFACES = {
    Path("msg/LocalizationStatus.msg"),
    Path("msg/ScienceTargetRegion.msg"),
    Path("msg/ExplorationTask.msg"),
    Path("msg/MotionExecutionFeedback.msg"),
}

_LARGE_ARTIFACT_SUFFIXES = FORBIDDEN_SUFFIXES | {".mcap"}
_SCANNED_TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hxx",
    ".ini",
    ".json",
    ".py",
    ".sh",
    ".toml",
    ".xml",
    ".yaml",
    ".yml",
}
_WINDOWS_PATH = re.compile(
    r"(?:(?<![A-Za-z0-9_])[A-Za-z]:[\\/]|[\\/]Users[\\/])"
)


def _tracked_files(root: Path) -> list[tuple[str, str, str]]:
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

    entries: list[tuple[str, str, str]] = []
    for record in completed.stdout.split("\0"):
        if not record:
            continue
        metadata, path = record.split("\t", maxsplit=1)
        mode, object_id, _stage = metadata.split(maxsplit=2)
        entries.append((mode, object_id, path))
    return entries


def _is_scanned_text(path: Path) -> bool:
    return path.name == "CMakeLists.txt" or path.suffix.lower() in _SCANNED_TEXT_SUFFIXES


def _find_nested_git_directories(root: Path) -> list[Path]:
    nested: list[Path] = []
    for directory, names, files in os.walk(root):
        current = Path(directory)
        if ".git" in names:
            if current != root:
                nested.append(current / ".git")
            names.remove(".git")
        if current != root and ".git" in files:
            nested.append(current / ".git")
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
                or relative_path.as_posix() in MIGRATION_INVENTORY_PATHS
                or path.suffix.lower() == ".md"
                or not _is_scanned_text(path)
            ):
                continue
            try:
                contents = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if _WINDOWS_PATH.search(contents):
                violations.append(path)
    return violations


def _find_provisional_package_paths(root: Path) -> list[Path]:
    """Return paths of all non-ignored packages named lunar_navigation_msgs."""
    packages: list[Path] = []
    for directory, names, filenames in os.walk(root):
        names[:] = [name for name in names if name not in IGNORED_DIRS]
        if "package.xml" not in filenames:
            continue
        package_path = Path(directory) / "package.xml"
        try:
            package_name = element_tree.parse(package_path).getroot().findtext("name")
        except element_tree.ParseError:
            continue
        if package_name == "lunar_navigation_msgs":
            packages.append(package_path.parent.relative_to(root))
    return sorted(packages)


def _find_unapproved_provisional_interfaces(root: Path) -> list[Path]:
    """Return unapproved ROS interface sources below the canonical package path."""
    package_root = root / PROVISIONAL_PACKAGE_PATH
    if not package_root.is_dir():
        return []
    interfaces: list[Path] = []
    for directory, names, filenames in os.walk(package_root):
        names[:] = [name for name in names if name not in IGNORED_DIRS]
        for filename in filenames:
            path = Path(directory) / filename
            relative_interface = path.relative_to(package_root)
            if (
                path.suffix in {".msg", ".srv", ".action"}
                and relative_interface not in ALLOWED_PROVISIONAL_INTERFACES
            ):
                interfaces.append(path.relative_to(root))
    return sorted(interfaces)


def _tracked_blob_size(root: Path, object_id: str) -> int | None:
    completed = subprocess.run(
        ["git", "cat-file", "-s", object_id],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip().isdigit():
        return None
    return int(completed.stdout)


def _is_in_forbidden_artifact_directory(path: Path) -> bool:
    return any(part in FORBIDDEN_ARTIFACT_DIRS for part in path.parts[:-1])


def check_repository(root: Path) -> list[str]:
    """Return repository-boundary violations; an empty list means the root is valid."""
    root = root.resolve()
    errors: list[str] = []

    for directory in _find_nested_git_directories(root):
        errors.append(f"nested Git directory: {directory.relative_to(root).as_posix()}")

    for source in _find_windows_paths(root):
        errors.append(f"Windows absolute path in source: {source.relative_to(root).as_posix()}")

    provisional_packages = _find_provisional_package_paths(root)
    for package in provisional_packages:
        if package != PROVISIONAL_PACKAGE_PATH:
            if PROVISIONAL_PACKAGE_PATH in provisional_packages:
                errors.append(f"duplicate lunar_navigation_msgs package: {package.as_posix()}")
            else:
                errors.append(
                    "noncanonical lunar_navigation_msgs package: "
                    f"{package.as_posix()} (expected {PROVISIONAL_PACKAGE_PATH.as_posix()})"
                )

    for interface in _find_unapproved_provisional_interfaces(root):
        errors.append(f"unapproved lunar_navigation_msgs interface: {interface.as_posix()}")

    for mode, object_id, relative_path in _tracked_files(root):
        path = Path(relative_path)
        if mode == "160000":
            errors.append(f"tracked gitlink is forbidden: {relative_path}")
        if path.suffix.lower() in _LARGE_ARTIFACT_SUFFIXES:
            errors.append(f"tracked forbidden artifact: {relative_path}")
        if _is_in_forbidden_artifact_directory(path):
            errors.append(f"tracked forbidden artifact directory: {relative_path}")
        blob_size = _tracked_blob_size(root, object_id)
        if blob_size is not None and blob_size > MAX_TRACKED_FILE_BYTES:
            errors.append(
                f"tracked file exceeds size limit ({MAX_TRACKED_FILE_BYTES} bytes): {relative_path}"
            )
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

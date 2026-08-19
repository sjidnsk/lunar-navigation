"""Immutable staging for validated policy model artifacts.

This module deliberately stages artifacts only.  The current ROS runtime has
no policy adapter, so a staged model never changes planner behaviour.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .state import atomic_write_json


def _load_validator():
    """Import the bundled ament package when running from source or a bundle."""
    for root in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parent.parent):
        package_root = root / "ros2_ws" / "src" / "lunar_policy_runtime"
        if package_root.is_dir() and str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
    from lunar_policy_runtime.manifest import ModelManifestError, validate_model_package

    return ModelManifestError, validate_model_package


class ModelInstallError(ValueError):
    pass


@dataclass(frozen=True)
class InstalledModel:
    model_id: str
    model_sha256: str
    path: Path


@dataclass(frozen=True)
class ModelStatus:
    active_model_id: str | None
    active_model_sha256: str | None
    previous_model_id: str | None
    model_binding: str


def _directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


class ModelStore:
    """A content-addressed model store with atomic active/previous pointers."""

    def __init__(self, root: Path, *, probe: Callable[[Path], None] | None = None) -> None:
        self.root = root
        self.probe = probe or (lambda _path: None)

    @property
    def _active_path(self) -> Path:
        return self.root / "active-model.json"

    @property
    def _previous_path(self) -> Path:
        return self.root / "previous-model.json"

    def _validated_source(self, source: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
        if source.is_dir():
            return source, None
        if not source.is_file() or not source.name.endswith(".tar.gz"):
            raise ModelInstallError("MODEL_PACKAGE_SOURCE_INVALID")
        temporary = tempfile.TemporaryDirectory(prefix="luna-model-")
        destination = Path(temporary.name)
        try:
            with tarfile.open(source, "r:gz") as archive:
                members = archive.getmembers()
                if any(member.issym() or member.islnk() or Path(member.name).is_absolute() or ".." in Path(member.name).parts for member in members):
                    raise ModelInstallError("MODEL_PACKAGE_LAYOUT_INVALID")
                archive.extractall(destination, members=members)
        except (tarfile.TarError, OSError) as error:
            temporary.cleanup()
            raise ModelInstallError("MODEL_PACKAGE_SOURCE_INVALID") from error
        entries = list(destination.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            return entries[0], temporary
        return destination, temporary

    def install(self, source: Path) -> InstalledModel:
        package_root, temporary = self._validated_source(source)
        try:
            manifest_error, validate = _load_validator()
            try:
                package = validate(package_root)
            except manifest_error as error:
                raise ModelInstallError(str(error)) from error
            model_hash = _directory_sha256(package_root)
            destination = self.root / package.manifest.model_id / model_hash
            if destination.exists():
                return InstalledModel(package.manifest.model_id, model_hash, destination)
            self.probe(package_root / "policy.onnx")
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = destination.parent / f".{model_hash}.staging"
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(package_root, staging)
            os.replace(staging, destination)
            return InstalledModel(package.manifest.model_id, model_hash, destination)
        finally:
            if temporary is not None:
                temporary.cleanup()

    def _read_pointer(self, path: Path) -> dict[str, str] | None:
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelInstallError("MODEL_POINTER_INVALID") from error
        if not isinstance(value, dict) or set(value) != {"model_id", "model_sha256"}:
            raise ModelInstallError("MODEL_POINTER_INVALID")
        if not all(isinstance(value[key], str) and value[key] for key in value):
            raise ModelInstallError("MODEL_POINTER_INVALID")
        return {"model_id": value["model_id"], "model_sha256": value["model_sha256"]}

    def _installed_versions(self, model_id: str) -> list[Path]:
        root = self.root / model_id
        return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name) if root.is_dir() else []

    def activate(self, model_id: str) -> ModelStatus:
        versions = self._installed_versions(model_id)
        if not versions:
            raise ModelInstallError("MODEL_NOT_INSTALLED")
        if len(versions) != 1:
            raise ModelInstallError("MODEL_VERSION_AMBIGUOUS")
        target = {"model_id": model_id, "model_sha256": versions[0].name}
        active = self._read_pointer(self._active_path)
        if active is not None and active != target:
            atomic_write_json(self._previous_path, active)
        atomic_write_json(self._active_path, target)
        return self.status()

    def rollback(self) -> ModelStatus:
        previous = self._read_pointer(self._previous_path)
        if previous is None:
            self._active_path.unlink(missing_ok=True)
            return self.status()
        atomic_write_json(self._active_path, previous)
        self._previous_path.unlink(missing_ok=True)
        return self.status()

    def status(self) -> ModelStatus:
        active = self._read_pointer(self._active_path)
        previous = self._read_pointer(self._previous_path)
        return ModelStatus(
            active_model_id=None if active is None else active["model_id"],
            active_model_sha256=None if active is None else active["model_sha256"],
            previous_model_id=None if previous is None else previous["model_id"],
            model_binding="staged_not_connected" if active is not None else "fallback",
        )


def model_store_root(config_path: Path, data_path: Path) -> Path:
    """Honor the portable LUNA_HOME layout while keeping the XDG default isolated."""
    if config_path.parent.name == "config":
        return config_path.parent.parent / "models"
    return data_path / "models"

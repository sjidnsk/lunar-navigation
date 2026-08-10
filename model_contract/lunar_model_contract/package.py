"""interface-v1 精确四文件模型包验证。"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from .action import ActionContractError, ActionContractV2
from .hashing import sha256_file
from .interface_manifest import (
    InterfaceModelManifest,
    InterfaceModelManifestError,
    MODEL_CONTENT_FILES,
)
from .observation import ObservationContractError, validate_observation_inputs


INTERFACE_MODEL_PACKAGE_FILES: Final = frozenset(
    {*MODEL_CONTENT_FILES, "manifest.json"}
)


class InterfaceModelPackageError(ValueError):
    """interface-v1 模型目录不满足物理或张量边界。"""


def _load_npz(path: Path, names: set[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != names:
                raise InterfaceModelPackageError(
                    f"{path.name} keys must equal {sorted(names)}"
                )
            arrays = {name: archive[name] for name in archive.files}
    except InterfaceModelPackageError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise InterfaceModelPackageError(
            f"cannot load {path.name}: {error}"
        ) from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise InterfaceModelPackageError(f"{path.name} object dtype is forbidden")
    return arrays


def validate_interface_model_package(
    path: str | Path,
) -> InterfaceModelManifest:
    """先验证目录与 hash，再验证三平台 golden 张量。"""
    try:
        root = Path(path).resolve(strict=True)
    except OSError as error:
        raise InterfaceModelPackageError(
            f"model package is unavailable: {error}"
        ) from error
    if not root.is_dir():
        raise InterfaceModelPackageError("model package must be a directory")
    entries = tuple(root.iterdir())
    names = {entry.name for entry in entries}
    if names != INTERFACE_MODEL_PACKAGE_FILES:
        raise InterfaceModelPackageError(
            "model package has missing or unexpected file(s): "
            f"missing={sorted(INTERFACE_MODEL_PACKAGE_FILES - names)}, "
            f"unexpected={sorted(names - INTERFACE_MODEL_PACKAGE_FILES)}"
        )
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise InterfaceModelPackageError("model package entries must be regular files")
    if any(entry.stat().st_size == 0 for entry in entries):
        raise InterfaceModelPackageError("model package entries must not be empty")
    try:
        manifest = InterfaceModelManifest.from_path(root / "manifest.json")
    except InterfaceModelManifestError as error:
        raise InterfaceModelPackageError(str(error)) from error
    for name in MODEL_CONTENT_FILES:
        actual = sha256_file(root / name)
        if actual != manifest.files[name]:
            raise InterfaceModelPackageError(
                f"SHA-256 mismatch for {name}: expected {manifest.files[name]}, got {actual}"
            )
    inputs = _load_npz(
        root / "golden_inputs.npz", {record.name for record in manifest.inputs}
    )
    outputs = _load_npz(
        root / "golden_outputs.npz", {record.name for record in manifest.outputs}
    )
    try:
        validate_observation_inputs(inputs)
        ActionContractV2.validate_outputs(outputs)
    except (ObservationContractError, ActionContractError) as error:
        raise InterfaceModelPackageError(str(error)) from error
    if inputs[manifest.inputs[0].name].shape[0] != 3:
        raise InterfaceModelPackageError("golden batch must contain exactly three platforms")
    if not np.array_equal(
        inputs["platform_context"], np.eye(3, dtype=np.float32)
    ):
        raise InterfaceModelPackageError(
            "golden platform order must be WHEELED, LEGGED, HOPPER"
        )
    if outputs[manifest.outputs[0].name].shape[0] != 3:
        raise InterfaceModelPackageError("golden input/output batch sizes must match")
    return manifest


__all__ = [
    "INTERFACE_MODEL_PACKAGE_FILES",
    "InterfaceModelPackageError",
    "validate_interface_model_package",
]

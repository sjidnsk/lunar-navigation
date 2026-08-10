from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lunar_model_contract.package import (
    INTERFACE_MODEL_PACKAGE_FILES,
    InterfaceModelPackageError,
    validate_interface_model_package,
)
from test_interface_manifest import valid_interface_manifest


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _inputs() -> dict[str, np.ndarray]:
    arrays = {
        "prior_channels": np.zeros((3, 4, 256, 256), np.float32),
        "coverage_summary": np.zeros((3, 3, 256, 256), np.float32),
        "local_crop": np.zeros((3, 4, 32, 32), np.float32),
        "frontier_features": np.zeros((3, 64, 12), np.float32),
        "pose_features": np.zeros((3, 5), np.float32),
        "candidate_mask": np.zeros((3, 64), np.bool_),
        "platform_context": np.eye(3, dtype=np.float32),
    }
    arrays["candidate_mask"][:, 0] = True
    return arrays


def _outputs() -> dict[str, np.ndarray]:
    arrays = {
        "frontier_logits": np.zeros((3, 64), np.float32),
        "theta_mu": np.zeros((3, 64), np.float32),
        "theta_kappa": np.ones((3, 64), np.float32),
        "value": np.zeros((3,), np.float32),
    }
    return arrays


def _package(root: Path) -> Path:
    root.mkdir()
    (root / "policy.onnx").write_bytes(b"test-onnx")
    np.savez(root / "golden_inputs.npz", **_inputs())
    np.savez(root / "golden_outputs.npz", **_outputs())
    document = valid_interface_manifest()
    document["files"] = {
        name: _sha(root / name)
        for name in (
            "policy.onnx",
            "golden_inputs.npz",
            "golden_outputs.npz",
        )
    }
    (root / "manifest.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    return root


def test_valid_package_has_exact_four_regular_files(tmp_path: Path) -> None:
    root = _package(tmp_path / "model")

    manifest = validate_interface_model_package(
        root, require_approved_identity=False
    )

    assert {item.name for item in root.iterdir()} == INTERFACE_MODEL_PACKAGE_FILES
    assert manifest.qualification == "integration_only"


def test_runtime_validator_rejects_coherently_replaced_four_file_package(
    tmp_path: Path,
) -> None:
    root = _package(tmp_path / "model")

    with pytest.raises(InterfaceModelPackageError, match="approved identity"):
        validate_interface_model_package(root)


@pytest.mark.parametrize("extra", ("checkpoint.pt", "optimizer.json"))
def test_package_rejects_training_state_or_extra_file(
    tmp_path: Path, extra: str
) -> None:
    root = _package(tmp_path / "model")
    (root / extra).write_bytes(b"forbidden")

    with pytest.raises(InterfaceModelPackageError, match="unexpected"):
        validate_interface_model_package(root, require_approved_identity=False)


def test_package_rejects_model_hash_drift(tmp_path: Path) -> None:
    root = _package(tmp_path / "model")
    (root / "policy.onnx").write_bytes(b"mutated")

    with pytest.raises(InterfaceModelPackageError, match="SHA-256"):
        validate_interface_model_package(root, require_approved_identity=False)


def test_package_rejects_non_three_platform_golden_batch(tmp_path: Path) -> None:
    root = _package(tmp_path / "model")
    inputs = _inputs()
    inputs["platform_context"][[1, 2]] = inputs["platform_context"][[2, 1]]
    np.savez(root / "golden_inputs.npz", **inputs)
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    document["files"]["golden_inputs.npz"] = _sha(root / "golden_inputs.npz")
    (root / "manifest.json").write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(InterfaceModelPackageError, match="platform order"):
        validate_interface_model_package(root, require_approved_identity=False)

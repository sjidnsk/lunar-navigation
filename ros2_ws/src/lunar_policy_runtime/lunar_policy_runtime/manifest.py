from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class ModelManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    source_commit: str
    observation_contract: str
    action_contract: str
    onnx_sha256: str
    normalization_sha256: str


@dataclass(frozen=True)
class ValidatedModelPackage:
    root: Path
    manifest: ModelManifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(path: Path, expected: str, reason: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or _sha256(path) != expected:
        raise ModelManifestError(reason)


def validate_model_package(root: Path) -> ValidatedModelPackage:
    allowed = {"model-manifest.json", "policy.onnx", "normalization.npz"}
    names = {path.name for path in root.iterdir()} if root.is_dir() else set()
    if names != allowed:
        raise ModelManifestError("MODEL_PACKAGE_LAYOUT_INVALID")
    try:
        data = json.loads((root / "model-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelManifestError("MODEL_MANIFEST_INVALID") from error
    expected = {
        "schema_version", "model_id", "source_commit", "observation_contract", "action_contract",
        "onnx_sha256", "normalization_sha256",
    }
    if not isinstance(data, dict) or set(data) != expected or data["schema_version"] != "luna-model-manifest/v1":
        raise ModelManifestError("MODEL_MANIFEST_INVALID")
    if data["observation_contract"] != "lunar-observation-contract/v4":
        raise ModelManifestError("OBSERVATION_CONTRACT_UNSUPPORTED")
    if data["action_contract"] != "lunar-action-contract/v2":
        raise ModelManifestError("ACTION_CONTRACT_UNSUPPORTED")
    _require_hash(root / "policy.onnx", data["onnx_sha256"], "ONNX_SHA256_MISMATCH")
    _require_hash(root / "normalization.npz", data["normalization_sha256"], "NORMALIZATION_SHA256_MISMATCH")
    return ValidatedModelPackage(
        root=root,
        manifest=ModelManifest(
            model_id=str(data["model_id"]),
            source_commit=str(data["source_commit"]),
            observation_contract=str(data["observation_contract"]),
            action_contract=str(data["action_contract"]),
            onnx_sha256=str(data["onnx_sha256"]),
            normalization_sha256=str(data["normalization_sha256"]),
        ),
    )

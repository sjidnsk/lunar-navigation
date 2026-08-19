from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deployment.luna_runtime.cli import run_cli
from deployment.luna_runtime.model_store import ModelInstallError, ModelStore


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_package(root: Path, *, normalization: bytes = b"normalization") -> Path:
    root.mkdir(parents=True)
    onnx = b"portable-onnx-model"
    (root / "policy.onnx").write_bytes(onnx)
    (root / "normalization.npz").write_bytes(normalization)
    manifest = {
        "schema_version": "luna-model-manifest/v1",
        "model_id": "demo-v4",
        "source_commit": "a" * 40,
        "observation_contract": "lunar-observation-contract/v4",
        "action_contract": "lunar-action-contract/v2",
        "onnx_sha256": _sha256(onnx),
        "normalization_sha256": _sha256(normalization),
    }
    (root / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_install_rejects_hash_mismatch_without_changing_active_model(tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "models")
    store.install(_model_package(tmp_path / "good"))
    store.activate("demo-v4")

    corrupt = _model_package(tmp_path / "bad", normalization=b"corrupt")
    manifest_path = corrupt / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["normalization_sha256"] = _sha256(b"normalization")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelInstallError, match="NORMALIZATION_SHA256_MISMATCH"):
        store.install(corrupt)
    assert store.status().active_model_id == "demo-v4"


def test_install_is_immutable_and_rollback_atomically_restores_previous_model(tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "models")
    installed = store.install(_model_package(tmp_path / "good"))
    assert (tmp_path / "models" / "demo-v4" / installed.model_sha256 / "policy.onnx").is_file()

    store.activate("demo-v4")
    restored = store.rollback()
    assert restored.active_model_id is None
    assert restored.model_binding == "fallback"
    assert not (tmp_path / "models" / "active-model.json").exists()


def test_cli_stages_and_activates_model_without_binding_it_to_ros(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        (Path(__file__).resolve().parents[2] / "deployment/config/runtime.default.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    package = _model_package(tmp_path / "package")
    staged = run_cli(["model", "install", str(package), "--config", str(config)], home=tmp_path / "home")
    active = run_cli(["model", "activate", "demo-v4", "--config", str(config)], home=tmp_path / "home")
    assert staged.exit_code == 0
    assert active.payload["model_binding"] == "staged_not_connected"

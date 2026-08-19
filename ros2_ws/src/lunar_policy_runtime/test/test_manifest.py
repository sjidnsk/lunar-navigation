from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_policy_runtime.manifest import ModelManifestError, validate_model_package


def test_model_package_rejects_normalization_hash_mismatch(tmp_path: Path) -> None:
    package = tmp_path / "model"
    package.mkdir()
    (package / "policy.onnx").write_bytes(b"onnx")
    (package / "normalization.npz").write_bytes(b"normalization")
    (package / "model-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "luna-model-manifest/v1",
                "model_id": "demo-v4",
                "source_commit": "a" * 40,
                "observation_contract": "lunar-observation-contract/v4",
                "action_contract": "lunar-action-contract/v2",
                "onnx_sha256": hashlib.sha256(b"onnx").hexdigest(),
                "normalization_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelManifestError, match="NORMALIZATION_SHA256_MISMATCH"):
        validate_model_package(package)

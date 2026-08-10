from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from lunar_model_contract.interface_manifest import (
    InterfaceModelManifest,
    InterfaceModelManifestError,
)


SOURCE_COMMIT = "fed9ea92c8ae55eb8423a008598935f515e801a3"
SHA = "a" * 64


def valid_interface_manifest() -> dict[str, object]:
    inputs = [
        ("prior_channels", [None, 4, 256, 256], "float32"),
        ("coverage_summary", [None, 3, 256, 256], "float32"),
        ("local_crop", [None, 4, 32, 32], "float32"),
        ("frontier_features", [None, 64, 12], "float32"),
        ("pose_features", [None, 5], "float32"),
        ("candidate_mask", [None, 64], "bool"),
        ("platform_context", [None, 3], "float32"),
    ]
    outputs = [
        ("frontier_logits", [None, 64], "float32"),
        ("theta_mu", [None, 64], "float32"),
        ("theta_kappa", [None, 64], "float32"),
        ("value", [None], "float32"),
    ]
    return {
        "schema_version": "lunar-policy-interface-manifest/v1",
        "qualification": "integration_only",
        "model_id": "fed9-step251-interface",
        "version": "1.0.0",
        "source_commit": SOURCE_COMMIT,
        "checkpoint": {
            "schema_version": "lunar-ppo-checkpoint/v6",
            "step": 251,
            "file_sha256": "5c19113953bc91854abac88835a833e7bebe25ed6865cf0a09f36e828b359727",
            "body_sha256": "001ad541eb4ba19930350c70cfd86d392c6443e73f0914bcde5f32dacdadb101",
        },
        "onnx_opset": 17,
        "observation_contract": "lunar-observation-contract/v3",
        "action_contract": "lunar-action-contract/v2",
        "training_identity": {
            "semantics": "lunar-training-semantics/sensor-30m-360-theta-mask-roi95-unbounded-common-start-subset/v5",
            "semantics_sha256": "93797e459cfdbeedf59ca68642122c9e3546730cd1dd4c9d5ea78bb495786e44",
            "capability_freeze_sha256": "60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95",
            "reward_sha256": "ec2ba24c8afd2a8c4416d4ece079155bcf33f326ed16453b0aded6e1bd77eeec",
            "environment_sha256": "8fb5d43781ff372ad2398d91d646caebb1c2ca79580244add22902eeec4687cb",
            "config_sha256": "1891988a68c5fdc08806c4918e99b159bbdf1fba867ed51dbf5fb95d0a9da891",
        },
        "capability_profiles": {
            name: {"capability_version": f"{name.lower()}-v1", "sha256": SHA}
            for name in ("WHEELED", "LEGGED", "HOPPER")
        },
        "inputs": [
            {"name": name, "shape": shape, "dtype": dtype}
            for name, shape, dtype in inputs
        ],
        "outputs": [
            {"name": name, "shape": shape, "dtype": dtype}
            for name, shape, dtype in outputs
        ],
        "normalization": {"applied_by_observation_builder": True},
        "files": {
            "policy.onnx": SHA,
            "golden_inputs.npz": SHA,
            "golden_outputs.npz": SHA,
        },
        "tolerances": {"atol": 1e-5, "rtol": 1e-4},
        "export_environment": {
            "python": "3.10",
            "torch": "2.x",
            "onnx": "1.x",
        },
    }


def test_valid_manifest_is_explicitly_integration_only(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(valid_interface_manifest()), encoding="utf-8"
    )

    manifest = InterfaceModelManifest.from_path(path)

    assert manifest.qualification == "integration_only"
    assert manifest.source_commit == SOURCE_COMMIT
    assert manifest.checkpoint_step == 251
    assert manifest.release_qualified is False


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda value: value.update(qualification="release"),
            "qualification",
        ),
        (
            lambda value: value.update(
                release_evaluation={"passed": True, "report_sha256": SHA}
            ),
            "unknown keys.*release_evaluation",
        ),
        (
            lambda value: value["checkpoint"].update(step=250),
            "checkpoint.step",
        ),
        (
            lambda value: value.update(source_commit="b" * 40),
            "source_commit",
        ),
        (
            lambda value: value["training_identity"].pop("reward_sha256"),
            "training_identity missing keys.*reward_sha256",
        ),
    ],
)
def test_manifest_rejects_relabeling_or_incomplete_old_identity(
    mutate, message: str
) -> None:
    document = copy.deepcopy(valid_interface_manifest())
    mutate(document)

    with pytest.raises(InterfaceModelManifestError, match=message):
        InterfaceModelManifest.from_dict(document)


def test_manifest_rejects_tensor_order_drift() -> None:
    document = valid_interface_manifest()
    document["inputs"][0], document["inputs"][1] = (
        document["inputs"][1],
        document["inputs"][0],
    )

    with pytest.raises(InterfaceModelManifestError, match="inputs.*order"):
        InterfaceModelManifest.from_dict(document)


def test_json_schema_keeps_release_claim_out_of_interface_package() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schema/interface-manifest.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    document = valid_interface_manifest()

    jsonschema.Draft202012Validator(schema).validate(document)
    document["release_evaluation"] = {
        "passed": True,
        "report_sha256": SHA,
    }

    with pytest.raises(jsonschema.ValidationError, match="not allowed"):
        jsonschema.Draft202012Validator(schema).validate(document)

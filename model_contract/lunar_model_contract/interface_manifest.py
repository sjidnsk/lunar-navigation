"""fed9/step251 接口联调模型的无歧义 manifest 合同。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Final, Mapping

from .action import ActionContractV2
from .observation import ObservationContractV3


SCHEMA_VERSION: Final = "lunar-policy-interface-manifest/v1"
QUALIFICATION: Final = "integration_only"
SOURCE_COMMIT: Final = "fed9ea92c8ae55eb8423a008598935f515e801a3"
CHECKPOINT_SCHEMA: Final = "lunar-ppo-checkpoint/v6"
CHECKPOINT_STEP: Final = 251
CHECKPOINT_FILE_SHA256: Final = "5c19113953bc91854abac88835a833e7bebe25ed6865cf0a09f36e828b359727"
CHECKPOINT_BODY_SHA256: Final = "001ad541eb4ba19930350c70cfd86d392c6443e73f0914bcde5f32dacdadb101"
MODEL_CONTENT_FILES: Final = frozenset(
    {"policy.onnx", "golden_inputs.npz", "golden_outputs.npz"}
)
PLATFORMS: Final = ("WHEELED", "LEGGED", "HOPPER")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TRAINING_IDENTITY: Final = {
    "semantics": "lunar-training-semantics/sensor-30m-360-theta-mask-roi95-unbounded-common-start-subset/v5",
    "semantics_sha256": "93797e459cfdbeedf59ca68642122c9e3546730cd1dd4c9d5ea78bb495786e44",
    "capability_freeze_sha256": "60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95",
    "reward_sha256": "ec2ba24c8afd2a8c4416d4ece079155bcf33f326ed16453b0aded6e1bd77eeec",
    "environment_sha256": "8fb5d43781ff372ad2398d91d646caebb1c2ca79580244add22902eeec4687cb",
    "config_sha256": "1891988a68c5fdc08806c4918e99b159bbdf1fba867ed51dbf5fb95d0a9da891",
}


class InterfaceModelManifestError(ValueError):
    """interface-v1 manifest 不完整、被重标注或与旧定义不兼容。"""


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int | None, ...]
    dtype: str


@dataclass(frozen=True)
class CapabilityRecord:
    capability_version: str
    sha256: str


@dataclass(frozen=True)
class InterfaceModelManifest:
    schema_version: str
    qualification: str
    model_id: str
    version: str
    source_commit: str
    checkpoint_step: int
    onnx_opset: int
    observation_contract: str
    action_contract: str
    capability_profiles: dict[str, CapabilityRecord]
    inputs: tuple[TensorRecord, ...]
    outputs: tuple[TensorRecord, ...]
    files: dict[str, str]
    atol: float
    rtol: float

    @property
    def release_qualified(self) -> bool:
        """联调包永远不能被解释为正式发布包。"""
        return False

    @classmethod
    def from_path(cls, path: str | Path) -> "InterfaceModelManifest":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise InterfaceModelManifestError(
                f"cannot read manifest.json: {error}"
            ) from error
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InterfaceModelManifest":
        root = _mapping(value, "manifest")
        _exact_keys(root, {
            "schema_version", "qualification", "model_id", "version",
            "source_commit", "checkpoint", "onnx_opset",
            "observation_contract", "action_contract", "training_identity",
            "capability_profiles", "inputs", "outputs", "normalization",
            "files", "tolerances", "export_environment",
        }, "manifest")
        _equal(root["schema_version"], SCHEMA_VERSION, "schema_version")
        _equal(root["qualification"], QUALIFICATION, "qualification")
        model_id = _identity(root["model_id"], "model_id")
        version = _identity(root["version"], "version")
        _equal(root["source_commit"], SOURCE_COMMIT, "source_commit")

        checkpoint = _mapping(root["checkpoint"], "checkpoint")
        _exact_keys(checkpoint, {
            "schema_version", "step", "file_sha256", "body_sha256",
        }, "checkpoint")
        _equal(checkpoint["schema_version"], CHECKPOINT_SCHEMA, "checkpoint.schema_version")
        _equal(checkpoint["step"], CHECKPOINT_STEP, "checkpoint.step")
        _equal(checkpoint["file_sha256"], CHECKPOINT_FILE_SHA256, "checkpoint.file_sha256")
        _equal(checkpoint["body_sha256"], CHECKPOINT_BODY_SHA256, "checkpoint.body_sha256")
        _equal(root["onnx_opset"], 17, "onnx_opset")
        _equal(root["observation_contract"], ObservationContractV3.version, "observation_contract")
        _equal(root["action_contract"], ActionContractV2.version, "action_contract")

        training = _mapping(root["training_identity"], "training_identity")
        _exact_keys(training, set(_TRAINING_IDENTITY), "training_identity")
        for name, expected in _TRAINING_IDENTITY.items():
            _equal(training[name], expected, f"training_identity.{name}")

        capabilities = _capabilities(root["capability_profiles"])
        inputs = _tensor_records(root["inputs"], ObservationContractV3, "inputs")
        outputs = _tensor_records(root["outputs"], ActionContractV2, "outputs")

        normalization = _mapping(root["normalization"], "normalization")
        _exact_keys(normalization, {"applied_by_observation_builder"}, "normalization")
        _equal(normalization["applied_by_observation_builder"], True, "normalization.applied_by_observation_builder")

        files = _mapping(root["files"], "files")
        _exact_keys(files, MODEL_CONTENT_FILES, "files")
        file_hashes = {name: _sha256(files[name], f"files.{name}") for name in files}

        tolerances = _mapping(root["tolerances"], "tolerances")
        _exact_keys(tolerances, {"atol", "rtol"}, "tolerances")
        atol = _positive(tolerances["atol"], "tolerances.atol")
        rtol = _positive(tolerances["rtol"], "tolerances.rtol")

        environment = _mapping(root["export_environment"], "export_environment")
        _exact_keys(environment, {"python", "torch", "onnx"}, "export_environment")
        for name in environment:
            _nonempty(environment[name], f"export_environment.{name}")

        return cls(
            SCHEMA_VERSION, QUALIFICATION, model_id, version, SOURCE_COMMIT,
            CHECKPOINT_STEP, 17, ObservationContractV3.version,
            ActionContractV2.version, capabilities, inputs, outputs,
            file_hashes, atol, rtol,
        )


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise InterfaceModelManifestError(f"{path} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], path: str) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise InterfaceModelManifestError(f"{path} missing keys: {', '.join(sorted(missing))}")
    if unknown:
        raise InterfaceModelManifestError(f"{path} unknown keys: {', '.join(sorted(unknown))}")


def _equal(value: Any, expected: Any, path: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise InterfaceModelManifestError(f"{path} must equal {expected}")


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise InterfaceModelManifestError(f"{path} must be a non-empty string")
    return value


def _identity(value: Any, path: str) -> str:
    text = _nonempty(value, path)
    if not _IDENTITY.fullmatch(text):
        raise InterfaceModelManifestError(f"{path} has invalid characters")
    return text


def _sha256(value: Any, path: str) -> str:
    text = _nonempty(value, path)
    if not _SHA.fullmatch(text):
        raise InterfaceModelManifestError(f"{path} must be lowercase SHA-256")
    return text


def _positive(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterfaceModelManifestError(f"{path} must be positive and finite")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise InterfaceModelManifestError(f"{path} must be positive and finite")
    return result


def _capabilities(value: Any) -> dict[str, CapabilityRecord]:
    root = _mapping(value, "capability_profiles")
    _exact_keys(root, set(PLATFORMS), "capability_profiles")
    result: dict[str, CapabilityRecord] = {}
    for platform in PLATFORMS:
        record = _mapping(root[platform], f"capability_profiles.{platform}")
        _exact_keys(record, {"capability_version", "sha256"}, f"capability_profiles.{platform}")
        result[platform] = CapabilityRecord(
            _nonempty(record["capability_version"], f"capability_profiles.{platform}.capability_version"),
            _sha256(record["sha256"], f"capability_profiles.{platform}.sha256"),
        )
    return result


def _tensor_records(value: Any, contract: Any, path: str) -> tuple[TensorRecord, ...]:
    if not isinstance(value, list):
        raise InterfaceModelManifestError(f"{path} must be an array")
    names = contract.input_names if path == "inputs" else contract.output_names
    actual = tuple(item.get("name") if isinstance(item, Mapping) else None for item in value)
    if actual != tuple(names):
        raise InterfaceModelManifestError(f"{path} tensor order must match contract")
    records: list[TensorRecord] = []
    for item, name in zip(value, names, strict=True):
        record = _mapping(item, f"{path}.{name}")
        _exact_keys(record, {"name", "shape", "dtype"}, f"{path}.{name}")
        shape = contract.shapes[name]
        if record["shape"] != list(shape):
            raise InterfaceModelManifestError(f"{path}.{name}.shape is invalid")
        dtype = "bool" if name == "candidate_mask" else "float32"
        _equal(record["dtype"], dtype, f"{path}.{name}.dtype")
        records.append(TensorRecord(name, tuple(shape), dtype))
    return tuple(records)


__all__ = ["InterfaceModelManifest", "InterfaceModelManifestError", "MODEL_CONTENT_FILES", "PLATFORMS"]

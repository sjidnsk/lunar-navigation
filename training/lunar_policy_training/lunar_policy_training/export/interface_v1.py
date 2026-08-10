"""从冻结的 fed9/step251 检查点生成 interface-v1 四文件包。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
import yaml

from lunar_model_contract import ActionContractV2, ObservationContractV3
from lunar_model_contract.hashing import sha256_file
from lunar_model_contract.interface_manifest import (
    CHECKPOINT_BODY_SHA256,
    CHECKPOINT_FILE_SHA256,
    CHECKPOINT_SCHEMA,
    CHECKPOINT_STEP,
    SOURCE_COMMIT,
)
from lunar_model_contract.package import validate_interface_model_package
from lunar_model_contract.observation import validate_observation_inputs

from ..checkpoint import TrainingCheckpointV6, load_checkpoint
from ..policy.cross_attention import CrossAttentionPolicy
from ..policy.observation import PolicyBatch


class InterfaceExportError(RuntimeError):
    """旧检查点身份、导出图或黄金等价性不满足联调合同。"""


class PolicyOnnxWrapper(nn.Module):
    """把 PolicyBatch 边界展开为 ONNX 的七个命名张量。"""

    def __init__(self, policy: CrossAttentionPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        prior_channels: torch.Tensor,
        coverage_summary: torch.Tensor,
        local_crop: torch.Tensor,
        frontier_features: torch.Tensor,
        pose_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        platform_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.policy(
            PolicyBatch(
                prior_channels=prior_channels,
                coverage_summary=coverage_summary,
                local_crop=local_crop,
                frontier_features=frontier_features,
                pose_features=pose_features,
                candidate_mask=candidate_mask,
                platform_context=platform_context,
            )
        )
        return (
            output.frontier_logits,
            output.theta_mu,
            output.theta_kappa,
            output.value,
        )


def require_checkpoint_file_identity(path: str | Path) -> Path:
    """在 torch 反序列化之前核对旧检查点的原始文件身份。"""
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise InterfaceExportError("checkpoint must be an existing regular file")
    actual = sha256_file(target)
    if actual != CHECKPOINT_FILE_SHA256:
        raise InterfaceExportError(
            "checkpoint file SHA-256 mismatch: "
            f"expected {CHECKPOINT_FILE_SHA256}, got {actual}"
        )
    return target


def make_golden_inputs() -> dict[str, np.ndarray]:
    """生成三平台固定顺序、无随机源的合法 Observation V3 批。"""
    arrays = {
        name: np.zeros(
            tuple(3 if size is None else size for size in shape),
            dtype=np.bool_ if name == "candidate_mask" else np.float32,
        )
        for name, shape in ObservationContractV3.shapes.items()
    }
    arrays["candidate_mask"][:, :4] = True
    arrays["platform_context"] = np.eye(3, dtype=np.float32)
    arrays["frontier_features"][:, :4, 0] = np.asarray(
        (0.20, 0.35, 0.50, 0.65), dtype=np.float32
    )
    arrays["frontier_features"][:, :4, 1] = np.asarray(
        (0.25, 0.40, 0.55, 0.70), dtype=np.float32
    )
    arrays["frontier_features"][:, :4, 4] = 1.0
    arrays["frontier_features"][:, :4, 5] = np.asarray(
        (0.04, 0.03, 0.02, 0.01), dtype=np.float32
    )
    arrays["frontier_features"][:, :4, 8] = 1.0
    arrays["frontier_features"][:, :4, 10] = 0.5
    arrays["frontier_features"][:, :4, 11] = 0.75
    arrays["pose_features"][:, 0:2] = 0.5
    arrays["pose_features"][:, 3] = 1.0
    arrays["pose_features"][:, 4] = 0.1
    validate_observation_inputs(arrays)
    return arrays


def _torch_outputs(
    policy: CrossAttentionPolicy,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    wrapper = PolicyOnnxWrapper(policy).eval()
    inputs = tuple(
        torch.from_numpy(arrays[name])
        for name in ObservationContractV3.input_names
    )
    with torch.inference_mode():
        values = wrapper(*inputs)
    return {
        name: value.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, value in zip(ActionContractV2.output_names, values, strict=True)
    }


def export_policy_onnx(
    policy: CrossAttentionPolicy,
    arrays: Mapping[str, np.ndarray],
    output_path: str | Path,
) -> dict[str, np.ndarray]:
    """导出 opset17 图并返回同一批次的 PyTorch 权威输出。"""
    validate_observation_inputs(arrays)
    target = Path(output_path)
    if target.exists():
        raise InterfaceExportError("ONNX output path already exists")
    policy.eval()
    wrapper = PolicyOnnxWrapper(policy).eval()
    inputs = tuple(
        torch.from_numpy(arrays[name])
        for name in ObservationContractV3.input_names
    )
    dynamic_axes = {
        name: {0: "batch"}
        for name in (
            *ObservationContractV3.input_names,
            *ActionContractV2.output_names,
        )
    }
    try:
        torch.onnx.export(
            wrapper,
            inputs,
            str(target),
            input_names=list(ObservationContractV3.input_names),
            output_names=list(ActionContractV2.output_names),
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    except Exception as error:
        raise InterfaceExportError(f"ONNX export failed: {error}") from error
    return _torch_outputs(policy, arrays)


def load_frozen_policy(path: str | Path) -> CrossAttentionPolicy:
    """只加载完整 model_state；不触碰 optimizer、scheduler 或 RNG。"""
    target = require_checkpoint_file_identity(path)
    checkpoint = load_checkpoint(target)
    if not isinstance(checkpoint, TrainingCheckpointV6):
        raise InterfaceExportError("checkpoint must use v6 schema")
    expected = {
        "schema_version": CHECKPOINT_SCHEMA,
        "global_step": CHECKPOINT_STEP,
        "source_commit": SOURCE_COMMIT,
        "contract_version": ObservationContractV3.version,
        "payload_sha256": CHECKPOINT_BODY_SHA256,
    }
    for name, value in expected.items():
        if getattr(checkpoint, name) != value:
            raise InterfaceExportError(f"checkpoint {name} mismatch")
    policy = CrossAttentionPolicy()
    try:
        policy.load_state_dict(dict(checkpoint.model_state), strict=True)
    except (RuntimeError, ValueError, TypeError) as error:
        raise InterfaceExportError(
            f"checkpoint model_state is incompatible: {error}"
        ) from error
    return policy.eval()


def _profile_identity(path: Path, expected_platform: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise InterfaceExportError(f"{expected_platform} profile is unavailable")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        platform = document["platform"]
    except (OSError, UnicodeError, TypeError, KeyError, yaml.YAMLError) as error:
        raise InterfaceExportError(
            f"{expected_platform} profile is invalid: {error}"
        ) from error
    if platform.get("platform_type") != expected_platform:
        raise InterfaceExportError(f"{expected_platform} profile type mismatch")
    version = platform.get("capability_version")
    if not isinstance(version, str) or not version:
        raise InterfaceExportError(f"{expected_platform} capability version is invalid")
    return {"capability_version": version, "sha256": sha256_file(path)}


def _manifest(
    files: Mapping[str, str],
    profiles: Mapping[str, Path],
) -> dict[str, object]:
    def tensors(names, shapes):
        return [
            {
                "name": name,
                "shape": list(shapes[name]),
                "dtype": "bool" if name == "candidate_mask" else "float32",
            }
            for name in names
        ]

    return {
        "schema_version": "lunar-policy-interface-manifest/v1",
        "qualification": "integration_only",
        "model_id": "fed9-step251-interface",
        "version": "1.0.0",
        "source_commit": SOURCE_COMMIT,
        "checkpoint": {
            "schema_version": CHECKPOINT_SCHEMA,
            "step": CHECKPOINT_STEP,
            "file_sha256": CHECKPOINT_FILE_SHA256,
            "body_sha256": CHECKPOINT_BODY_SHA256,
        },
        "onnx_opset": 17,
        "observation_contract": ObservationContractV3.version,
        "action_contract": ActionContractV2.version,
        "training_identity": {
            "semantics": "lunar-training-semantics/sensor-30m-360-theta-mask-roi95-unbounded-common-start-subset/v5",
            "semantics_sha256": "93797e459cfdbeedf59ca68642122c9e3546730cd1dd4c9d5ea78bb495786e44",
            "capability_freeze_sha256": "60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95",
            "reward_sha256": "ec2ba24c8afd2a8c4416d4ece079155bcf33f326ed16453b0aded6e1bd77eeec",
            "environment_sha256": "8fb5d43781ff372ad2398d91d646caebb1c2ca79580244add22902eeec4687cb",
            "config_sha256": "1891988a68c5fdc08806c4918e99b159bbdf1fba867ed51dbf5fb95d0a9da891",
        },
        "capability_profiles": {
            platform: _profile_identity(profiles[platform], platform)
            for platform in ("WHEELED", "LEGGED", "HOPPER")
        },
        "inputs": tensors(ObservationContractV3.input_names, ObservationContractV3.shapes),
        "outputs": tensors(ActionContractV2.output_names, ActionContractV2.shapes),
        "normalization": {"applied_by_observation_builder": True},
        "files": dict(files),
        "tolerances": {"atol": 1e-5, "rtol": 1e-4},
        "export_environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "onnx": onnx.__version__,
        },
    }


def export_interface_v1_package(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    profiles: Mapping[str, str | Path],
) -> Path:
    """生成、等价验证并原子发布一个四文件目录。"""
    target = Path(output_dir)
    if target.exists():
        raise InterfaceExportError("model package output already exists")
    if set(profiles) != {"WHEELED", "LEGGED", "HOPPER"}:
        raise InterfaceExportError("three platform profiles are required")
    profile_paths = {name: Path(path) for name, path in profiles.items()}
    policy = load_frozen_policy(checkpoint_path)
    arrays = make_golden_inputs()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        expected = export_policy_onnx(policy, arrays, staging / "policy.onnx")
        np.savez(staging / "golden_inputs.npz", **arrays)
        np.savez(staging / "golden_outputs.npz", **expected)
        session = ort.InferenceSession(
            str(staging / "policy.onnx"), providers=["CPUExecutionProvider"]
        )
        actual_values = session.run(
            list(ActionContractV2.output_names),
            {name: arrays[name] for name in ObservationContractV3.input_names},
        )
        for name, actual in zip(ActionContractV2.output_names, actual_values, strict=True):
            if not np.allclose(actual, expected[name], atol=1e-5, rtol=1e-4):
                raise InterfaceExportError(f"ONNX equivalence failed for {name}")
        files = {
            name: sha256_file(staging / name)
            for name in ("policy.onnx", "golden_inputs.npz", "golden_outputs.npz")
        }
        (staging / "manifest.json").write_text(
            json.dumps(_manifest(files, profile_paths), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # 导出阶段验证结构与内部 hash；批准身份由审阅后的固定 hash 清单提供。
        validate_interface_model_package(
            staging, require_approved_identity=False
        )
        os.rename(staging, target)
    except Exception as error:
        raise InterfaceExportError(
            f"interface package export failed; staging kept at {staging}: {error}"
        ) from error
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wheeled-profile", required=True, type=Path)
    parser.add_argument("--legged-profile", required=True, type=Path)
    parser.add_argument("--hopper-profile", required=True, type=Path)
    args = parser.parse_args(argv)
    output = export_interface_v1_package(
        args.checkpoint,
        args.output,
        {
            "WHEELED": args.wheeled_profile,
            "LEGGED": args.legged_profile,
            "HOPPER": args.hopper_profile,
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

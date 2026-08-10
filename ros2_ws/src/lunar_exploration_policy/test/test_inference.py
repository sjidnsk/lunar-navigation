from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pytest

from lunar_exploration_policy.inference import (
    OnnxPolicyRuntime,
    PolicyRuntimeError,
    validate_session_signature,
)


@dataclass
class _TensorMetadata:
    name: str
    shape: list[object]
    type: str


def _inputs() -> list[_TensorMetadata]:
    return [
        _TensorMetadata("prior_channels", ["batch", 4, 256, 256], "tensor(float)"),
        _TensorMetadata("coverage_summary", ["batch", 3, 256, 256], "tensor(float)"),
        _TensorMetadata("local_crop", ["batch", 4, 32, 32], "tensor(float)"),
        _TensorMetadata("frontier_features", ["batch", 64, 12], "tensor(float)"),
        _TensorMetadata("pose_features", ["batch", 5], "tensor(float)"),
        _TensorMetadata("candidate_mask", ["batch", 64], "tensor(bool)"),
        _TensorMetadata("platform_context", ["batch", 3], "tensor(float)"),
    ]


def _outputs() -> list[_TensorMetadata]:
    return [
        _TensorMetadata("frontier_logits", ["batch", 64], "tensor(float)"),
        _TensorMetadata("theta_mu", ["batch", 64], "tensor(float)"),
        _TensorMetadata("theta_kappa", ["batch", 64], "tensor(float)"),
        _TensorMetadata("value", ["batch"], "tensor(float)"),
    ]


def test_session_signature_accepts_exact_dynamic_batch_contract() -> None:
    validate_session_signature(_inputs(), _outputs())


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda inputs, outputs: inputs.reverse(), "input names"),
        (lambda inputs, outputs: setattr(inputs[4], "shape", ["batch", 6]), "pose_features"),
        (lambda inputs, outputs: setattr(inputs[5], "type", "tensor(float)"), "candidate_mask"),
        (lambda inputs, outputs: outputs.pop(), "output names"),
    ],
)
def test_session_signature_rejects_name_shape_or_dtype_drift(
    mutate, message: str
) -> None:
    inputs, outputs = _inputs(), _outputs()
    mutate(inputs, outputs)

    with pytest.raises(PolicyRuntimeError, match=message):
        validate_session_signature(inputs, outputs)


def test_real_four_file_package_reproduces_golden_outputs() -> None:
    configured = os.environ.get("LUNAR_INTERFACE_V1_MODEL_DIR")
    if configured is None:
        pytest.skip("LUNAR_INTERFACE_V1_MODEL_DIR is not configured")
    root = Path(configured)
    with np.load(root / "golden_inputs.npz", allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    with np.load(root / "golden_outputs.npz", allow_pickle=False) as archive:
        expected = {name: archive[name] for name in archive.files}

    runtime = OnnxPolicyRuntime(root)
    actual = runtime.infer(inputs)

    for name in expected:
        np.testing.assert_allclose(
            actual[name],
            expected[name],
            atol=runtime.manifest.atol,
            rtol=runtime.manifest.rtol,
        )

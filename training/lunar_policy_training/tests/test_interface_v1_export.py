from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from lunar_model_contract import ActionContractV2, ObservationContractV3
from lunar_policy_training.export.interface_v1 import (
    InterfaceExportError,
    PolicyOnnxWrapper,
    export_policy_onnx,
    make_golden_inputs,
    require_checkpoint_file_identity,
)
from lunar_policy_training.policy.cross_attention import CrossAttentionPolicy


def test_checkpoint_file_identity_is_checked_before_deserialization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.pt"
    path.write_bytes(b"not-a-checkpoint")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(InterfaceExportError, match=f"got {actual}"):
        require_checkpoint_file_identity(path)


def test_wrapper_exposes_contract_order_and_four_outputs() -> None:
    policy = CrossAttentionPolicy().eval()
    arrays = make_golden_inputs()
    tensors = tuple(
        torch.from_numpy(arrays[name])
        for name in ObservationContractV3.input_names
    )

    outputs = PolicyOnnxWrapper(policy)(*tensors)

    assert len(outputs) == 4
    assert tuple(output.shape for output in outputs) == (
        (3, 64),
        (3, 64),
        (3, 64),
        (3,),
    )


def test_onnx_runtime_matches_pytorch_for_three_platform_batch(
    tmp_path: Path,
) -> None:
    torch.manual_seed(4080)
    policy = CrossAttentionPolicy().eval()
    arrays = make_golden_inputs()
    onnx_path = tmp_path / "policy.onnx"

    expected = export_policy_onnx(policy, arrays, onnx_path)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    actual = dict(
        zip(
            ActionContractV2.output_names,
            session.run(
                list(ActionContractV2.output_names),
                {
                    name: arrays[name]
                    for name in ObservationContractV3.input_names
                },
            ),
            strict=True,
        )
    )

    assert tuple(item.name for item in session.get_inputs()) == (
        ObservationContractV3.input_names
    )
    assert tuple(item.name for item in session.get_outputs()) == (
        ActionContractV2.output_names
    )
    for name in ActionContractV2.output_names:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-5, rtol=1e-4)

    single = session.run(
        list(ActionContractV2.output_names),
        {
            name: arrays[name][:1]
            for name in ObservationContractV3.input_names
        },
    )
    for name, value in zip(ActionContractV2.output_names, single, strict=True):
        np.testing.assert_allclose(
            value, expected[name][:1], atol=1e-5, rtol=1e-4
        )

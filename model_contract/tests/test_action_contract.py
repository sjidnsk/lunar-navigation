from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lunar_model_contract.action import ActionContractError, ActionContractV2  # noqa: E402


def _valid_outputs(batch_size: int = 2) -> dict[str, np.ndarray]:
    return {
        "frontier_logits": np.zeros((batch_size, 64), dtype=np.float32),
        "theta_mu": np.zeros((batch_size, 64), dtype=np.float32),
        "theta_kappa": np.full((batch_size, 64), 1.0, dtype=np.float32),
        "value": np.zeros((batch_size,), dtype=np.float32),
    }


def test_action_contract_freezes_exact_outputs_and_constraints() -> None:
    assert ActionContractV2.output_names == (
        "frontier_logits", "theta_mu", "theta_kappa", "value"
    )
    assert ActionContractV2.shapes == {
        "frontier_logits": (None, 64),
        "theta_mu": (None, 64),
        "theta_kappa": (None, 64),
        "value": (None,),
    }
    assert ActionContractV2.candidate_count == 64
    assert ActionContractV2.yaw_tolerance_rad == np.pi / 24.0
    assert ActionContractV2.theta_kappa_min == 0.05
    assert ActionContractV2.theta_kappa_max == 64.0


def test_action_contract_accepts_finite_batched_outputs() -> None:
    ActionContractV2.validate_outputs(_valid_outputs())


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda outputs: outputs.__setitem__(
                "theta_mu", np.zeros((2, 63), dtype=np.float32)
            ),
            "theta_mu must have shape [B,64]",
        ),
        (
            lambda outputs: outputs.__setitem__("value", np.zeros((2, 1), dtype=np.float32)),
            "value must have shape [B]",
        ),
        (
            lambda outputs: outputs.__setitem__(
                "frontier_logits", np.full((2, 64), np.nan, dtype=np.float32)
            ),
            "frontier_logits must contain only finite values",
        ),
        (
            lambda outputs: outputs.__setitem__(
                "theta_kappa", np.full((2, 64), 0.01, dtype=np.float32)
            ),
            "theta_kappa must be within [0.05, 64.0]",
        ),
        (
            lambda outputs: outputs.__setitem__(
                "theta_kappa", np.full((2, 64), 65.0, dtype=np.float32)
            ),
            "theta_kappa must be within [0.05, 64.0]",
        ),
    ),
)
def test_action_contract_rejects_invalid_outputs(mutate, message: str) -> None:
    outputs = _valid_outputs()
    mutate(outputs)

    with pytest.raises(ActionContractError, match=re.escape(message)):
        ActionContractV2.validate_outputs(outputs)

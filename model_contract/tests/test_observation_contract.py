from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lunar_model_contract.observation import (  # noqa: E402
    ObservationContractError,
    ObservationContractV1,
    PLATFORM_CONTEXTS,
    validate_platform_context,
)


EXPECTED_INPUTS = (
    "prior_channels",
    "coverage_summary",
    "local_crop",
    "frontier_features",
    "pose_features",
    "candidate_mask",
    "platform_context",
)


def test_observation_contract_exposes_the_exact_seven_inputs() -> None:
    """Would fail if deployment and training disagree on a required input."""
    assert ObservationContractV1.input_names == EXPECTED_INPUTS


@pytest.mark.parametrize("context", tuple(PLATFORM_CONTEXTS.values()))
def test_platform_context_accepts_each_defined_platform_one_hot(context: tuple[float, ...]) -> None:
    """Would fail if a supported platform could not enter the shared policy."""
    validate_platform_context(np.asarray([context], dtype=np.float32))


@pytest.mark.parametrize(
    ("array", "message"),
    (
        (np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32), "platform_context must be one-hot"),
        (np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64), "platform_context must be float32 [B,3]"),
        (np.asarray([[1.0, np.nan, 0.0]], dtype=np.float32), "platform_context must be finite one-hot"),
    ),
)
def test_platform_context_rejects_invalid_tensor_contract(
    array: np.ndarray, message: str
) -> None:
    """Would fail if malformed platform input reached policy inference."""
    with pytest.raises(ObservationContractError, match=re.escape(message)):
        validate_platform_context(array)

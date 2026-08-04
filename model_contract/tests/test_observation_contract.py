from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lunar_model_contract.observation import (  # noqa: E402
    ObservationContractError,
    ObservationContractV2,
    validate_observation_inputs,
)


def _valid_inputs(batch_size: int = 2) -> dict[str, np.ndarray]:
    return {
        "prior_channels": np.zeros((batch_size, 4, 256, 256), dtype=np.float32),
        "coverage_summary": np.zeros((batch_size, 3, 256, 256), dtype=np.float32),
        "local_crop": np.zeros((batch_size, 4, 32, 32), dtype=np.float32),
        "frontier_features": np.zeros((batch_size, 64, 12), dtype=np.float32),
        "pose_features": np.zeros((batch_size, 6), dtype=np.float32),
        "candidate_mask": np.zeros((batch_size, 64), dtype=np.bool_),
        "platform_context": np.tile(
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (batch_size, 1)
        ),
    }


def test_v2_contract_freezes_exact_shapes_channels_and_fields() -> None:
    assert ObservationContractV2.shapes == {
        "prior_channels": (None, 4, 256, 256),
        "coverage_summary": (None, 3, 256, 256),
        "local_crop": (None, 4, 32, 32),
        "frontier_features": (None, 64, 12),
        "pose_features": (None, 6),
        "candidate_mask": (None, 64),
        "platform_context": (None, 3),
    }
    assert ObservationContractV2.prior_channels == (
        "observed_relative_elevation",
        "mission_priority",
        "observed_physical_obstacle_ratio",
        "active_platform_traversable_ratio",
    )
    assert ObservationContractV2.coverage_summary_channels == (
        "observed_ratio",
        "mission_roi_ratio",
        "unobserved_priority_ratio",
    )
    assert ObservationContractV2.local_crop_channels == (
        "relative_elevation",
        "observed_mask",
        "observed_physical_obstacle",
        "active_platform_traversable",
    )
    assert ObservationContractV2.frontier_fields == (
        "x_norm",
        "y_norm",
        "distance_from_robot_norm",
        "bearing_sin",
        "bearing_cos",
        "potential_coverage_gain_ratio",
        "priority_weighted_gain_ratio",
        "normal_sin",
        "normal_cos",
        "normal_confidence",
        "clearance_margin_norm",
        "region_remaining_ratio",
    )
    assert ObservationContractV2.pose_fields == (
        "x_norm",
        "y_norm",
        "map_yaw_sin",
        "map_yaw_cos",
        "mission_observed_ratio",
        "remaining_decision_budget_ratio",
    )


def test_v2_validator_accepts_exact_inputs_with_all_false_candidate_mask() -> None:
    validate_observation_inputs(_valid_inputs())


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda inputs: inputs.__setitem__(
                "prior_channels", np.zeros((2, 3, 256, 256), dtype=np.float32)
            ),
            "prior_channels must have shape [B,4,256,256]",
        ),
        (
            lambda inputs: inputs.__setitem__(
                "local_crop", np.zeros((2, 4, 64, 64), dtype=np.float32)
            ),
            "local_crop must have shape [B,4,32,32]",
        ),
        (
            lambda inputs: inputs.__setitem__(
                "frontier_features", np.zeros((2, 63, 12), dtype=np.float32)
            ),
            "frontier_features must have shape [B,64,12]",
        ),
        (
            lambda inputs: inputs.__setitem__(
                "frontier_features", np.zeros((2, 64, 11), dtype=np.float32)
            ),
            "frontier_features must have shape [B,64,12]",
        ),
        (
            lambda inputs: inputs.__setitem__(
                "pose_features", np.zeros((2, 5), dtype=np.float32)
            ),
            "pose_features must have shape [B,6]",
        ),
        (
            lambda inputs: inputs.__setitem__(
                "candidate_mask", np.zeros((2, 64), dtype=np.float32)
            ),
            "candidate_mask must be bool [B,64]",
        ),
        (
            lambda inputs: inputs.__setitem__(
                "coverage_summary", np.full((2, 3, 256, 256), np.nan, dtype=np.float32)
            ),
            "coverage_summary must contain only finite values",
        ),
        (
            lambda inputs: inputs.__setitem__(
                "pose_features", np.full((2, 6), np.inf, dtype=np.float32)
            ),
            "pose_features must contain only finite values",
        ),
        (
            lambda inputs: inputs.__setitem__(
                "platform_context", np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
            ),
            "platform_context must be one-hot",
        ),
    ),
)
def test_v2_validator_rejects_malformed_inputs(mutate, message: str) -> None:
    inputs = _valid_inputs()
    mutate(inputs)

    with pytest.raises(ObservationContractError, match=re.escape(message)):
        validate_observation_inputs(inputs)


def test_v2_validator_rejects_wrong_names_and_inconsistent_batch() -> None:
    inputs = _valid_inputs()
    inputs["unexpected"] = inputs.pop("pose_features")
    with pytest.raises(ObservationContractError, match="names must exactly match"):
        validate_observation_inputs(inputs)

    inputs = _valid_inputs()
    inputs["candidate_mask"] = np.zeros((1, 64), dtype=np.bool_)
    with pytest.raises(ObservationContractError, match="candidate_mask batch size must match 2"):
        validate_observation_inputs(inputs)

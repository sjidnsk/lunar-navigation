from __future__ import annotations

import copy
import pathlib
import sys

import pytest


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

from lunar_policy_training.environment.formal_episode_state import (  # noqa: E402
    FormalWorkerState,
)


def _worker_state() -> dict[str, object]:
    identity = {
        "episode_id": "scene/wheeled/0/episode-4",
        "mission_revision": 1,
        "map_snapshot_id": "a" * 64,
        "robot_state_id": "b" * 64,
        "state_time_ns": 2_250_000_000,
        "execution_state": "DECISION_BOUNDARY",
        "candidate_set_id": "c" * 64,
    }
    pose = {
        "x_m": 128.5,
        "y_m": 256.5,
        "yaw_rad": 0.25,
        "elevation_m": 7.0,
        "frame_id": "map",
    }
    return {
        "scenario_schedule_id": "cache/train/v3",
        "platform_type": "WHEELED",
        "worker_index": 0,
        "platform_worker_index": 0,
        "platform_worker_count": 8,
        "episode_cursor": 4,
        "scene_id": "d" * 64,
        "scene_seed": "e" * 64,
        "start_seed": "f" * 64,
        "episode_seed": "1" * 64,
        "coverability_mask_sha256": "3" * 64,
        "start_cell": [64, 96],
        "current_pose": pose,
        "legged_body_z_m": 7.0,
        "execution_state": "DECISION_BOUNDARY",
        "observation_revision": 2,
        "state_time_ns": 2_250_000_000,
        "reveal_history": [
            {
                "pose": pose,
                "elapsed_s": 1.25,
                "path_samples": [
                    {
                        "pose": {
                            **pose,
                            "x_m": 127.5,
                        },
                        "elapsed_s": 0.5,
                    },
                    {
                        "pose": pose,
                        "elapsed_s": 0.75,
                    },
                ],
                "execution_state": "DECISION_BOUNDARY",
                "legged_body_z_m": 7.0,
            }
        ],
        "observation_identity": identity,
        "policy_batch_sha256": "2" * 64,
        "rejected_candidate_indices": [1, 3],
        "last_hop_available_delta_v_mps": 0.0,
    }


def test_formal_worker_state_roundtrips_as_strict_json() -> None:
    payload = _worker_state()

    state = FormalWorkerState.from_dict(payload)

    assert state.to_dict() == payload
    assert state.episode_cursor == 4
    assert state.rejected_candidate_indices == (1, 3)
    assert state.candidate_gain_resolution_m is None
    assert len(state.reveal_history[0].path_samples) == 2


def test_formal_worker_state_roundtrips_detail_gain_marker() -> None:
    payload = _worker_state()
    payload["candidate_gain_resolution_m"] = 0.2

    state = FormalWorkerState.from_dict(payload)

    assert state.to_dict() == payload
    assert state.candidate_gain_resolution_m == 0.2


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.pop("scene_id"), "structure"),
        (lambda value: value.__setitem__("extra", 1), "structure"),
        (
            lambda value: value["current_pose"].__setitem__("x_m", float("nan")),
            "finite",
        ),
        (lambda value: value.__setitem__("platform_type", "FLYING"), "platform"),
        (lambda value: value.__setitem__("episode_cursor", -1), "cursor"),
        (
            lambda value: value.__setitem__("execution_state", "IN_FLIGHT"),
            "stable",
        ),
        (
            lambda value: value.__setitem__("observation_revision", 8),
            "revision",
        ),
        (
            lambda value: value.__setitem__(
                "rejected_candidate_indices", [3, 1]
            ),
            "candidate",
        ),
        (
            lambda value: value.__setitem__(
                "candidate_gain_resolution_m", 1.0
            ),
            "candidate gain resolution",
        ),
    ),
)
def test_formal_worker_state_rejects_ambiguous_or_unstable_payloads(
    mutation, message: str
) -> None:
    payload = copy.deepcopy(_worker_state())
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        FormalWorkerState.from_dict(payload)

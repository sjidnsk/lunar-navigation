from __future__ import annotations

import hashlib
import pathlib
import sys
from dataclasses import replace

import numpy as np
import pytest
from lunar_planner_training_bridge import CandidateDisposition


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

from lunar_policy_training.environment.candidate_builder import (  # noqa: E402
    CandidateBuilderV2,
    PhysicalCandidateUniverse,
)
from lunar_policy_training.environment.formal_episode_state import (  # noqa: E402
    FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION,
    FormalWorkerState,
)
from lunar_policy_training.environment.macro_step import PolicyAction  # noqa: E402
from lunar_policy_training.environment.observation_boundary import (  # noqa: E402
    SensorBoundaryEvidence,
)
from test_formal_builder import _assembly  # noqa: E402


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
    candidate_ids = [f"{index + 10:064x}" for index in range(64)]
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
        "physical_snapshot_id": "4" * 64,
        "physical_evidence_generation": 3,
        "physical_evidence_sha256": "5" * 64,
        "physical_candidate_universe_sha256": "6" * 64,
        "planner_failed_candidate_ids": [],
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
                "defer_candidate_rebuild": False,
            }
        ],
        "replay_event_kinds": ["REVEAL"],
        "observation_identity": identity,
        "policy_batch_sha256": "2" * 64,
        "candidate_ids": candidate_ids,
        "candidate_mask": [True] * 64,
        "oracle_opportunity_count": 70,
        "oracle_opportunity_set_sha256": "7" * 64,
        "terminal_reason": None,
        "defer_candidate_rebuild": False,
        "last_hop_available_delta_v_mps": 0.0,
        "candidate_gain_resolution_m": 0.2,
    }


def _force_seventy_candidate_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CandidateBuilderV2.build_physical_universe

    def build_seventy(self, *args, **kwargs):
        universe = original(self, *args, **kwargs)
        candidates = universe.candidates[:70]
        assert len(candidates) == 70
        diagnostics = replace(
            universe.diagnostics,
            physical_candidate_universe_count=70,
            selected_policy_candidate_count=64,
            available_candidate_count=70,
            untried_reserve_count=6,
        )
        return PhysicalCandidateUniverse(
            physical_snapshot_id=universe.physical_snapshot_id,
            physical_reachability_algorithm_id=(
                universe.physical_reachability_algorithm_id
            ),
            candidates=candidates,
            universe_sha256=hashlib.sha256(
                "".join(item.candidate_id for item in candidates).encode("ascii")
            ).hexdigest(),
            diagnostics=diagnostics,
        )

    monkeypatch.setattr(
        CandidateBuilderV2,
        "build_physical_universe",
        build_seventy,
    )


def _worker_with_six_failures(tmp_path, monkeypatch):
    _force_seventy_candidate_universe(monkeypatch)
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory.create_for_episode(
        0,
        "WHEELED",
        4,
        platform_worker_index=0,
        platform_worker_count=8,
    )
    episode = worker.episode
    snapshot = episode._snapshot
    assert snapshot is not None
    failed_ids = tuple(
        str(snapshot.candidates.candidate_ids[index]) for index in range(6)
    )
    initial_ids = tuple(
        str(snapshot.candidates.candidate_ids[index]) for index in range(64)
    )
    for candidate_id in failed_ids:
        boundary = episode.refresh_after_planning_failure(
            candidate_id,
            CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT,
            snapshot.candidate_universe.physical_snapshot_id,
        )
        worker.environment._install_observation(boundary.next_observation)
    refreshed = episode._snapshot
    assert refreshed is not None
    assert refreshed.candidates.count == 64
    assert set(refreshed.candidates.candidate_ids) - set(initial_ids)
    return assembly, worker


def test_formal_worker_state_roundtrips_as_strict_json() -> None:
    payload = _worker_state()

    state = FormalWorkerState.from_dict(payload)

    assert FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION == (
        "lunar-formal-environment-state/v8"
    )
    assert state.to_dict() == payload
    assert state.episode_cursor == 4
    assert state.physical_evidence_generation == 3
    assert state.planner_failed_candidate_ids == ()
    assert state.candidate_gain_resolution_m == 0.2
    assert len(state.reveal_history[0].path_samples) == 2


@pytest.mark.parametrize(
    "missing",
    (
        "physical_snapshot_id",
        "physical_evidence_generation",
        "physical_evidence_sha256",
        "physical_candidate_universe_sha256",
        "planner_failed_candidate_ids",
        "replay_event_kinds",
        "defer_candidate_rebuild",
    ),
)
def test_formal_worker_state_rejects_missing_physical_identity(missing: str) -> None:
    payload = _worker_state()
    payload.pop(missing)

    with pytest.raises(ValueError, match="structure"):
        FormalWorkerState.from_dict(payload)


def test_formal_worker_state_rejects_retired_primitive_identity() -> None:
    payload = _worker_state()
    payload.update(
        {
            "primitive_graph_revision": 2,
            "primitive_graph_sha256": "8" * 64,
            "primitive_world_evidence_sha256": "9" * 64,
            "primitive_set_sha256": "a" * 64,
            "rejected_candidate_indices": [1, 3],
        }
    )

    with pytest.raises(ValueError, match="structure"):
        FormalWorkerState.from_dict(payload)


def test_formal_worker_state_rejects_promoted_defaults() -> None:
    payload = _worker_state()
    payload["physical_snapshot_id"] = "0" * 64
    payload["physical_evidence_generation"] = 0
    payload["physical_evidence_sha256"] = "0" * 64
    payload["physical_candidate_universe_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="physical"):
        FormalWorkerState.from_dict(payload)


def test_physical_worker_replay_roundtrips_reserve_failures_bit_exactly(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembly, worker = _worker_with_six_failures(tmp_path, monkeypatch)
    state = worker.snapshot_episode_state()
    before_snapshot = worker.episode._snapshot
    assert before_snapshot is not None
    before_observation = worker.environment.current_observation
    restored = assembly.factory.restore_for_episode(
        worker_index=0,
        platform_type="WHEELED",
        episode_cursor=4,
        platform_worker_index=0,
        platform_worker_count=8,
        state=state,
    )
    after_snapshot = restored.episode._snapshot
    assert after_snapshot is not None
    after_observation = restored.environment.current_observation

    assert len(after_snapshot.candidate_universe.candidates) == 70
    assert len(state["planner_failed_candidate_ids"]) == 6
    assert tuple(after_snapshot.candidates.candidate_ids) == tuple(
        before_snapshot.candidates.candidate_ids
    )
    assert np.array_equal(
        after_snapshot.candidates.mask,
        before_snapshot.candidates.mask,
    )
    assert after_snapshot.candidate_universe_sha256 == (
        before_snapshot.candidate_universe_sha256
    )
    assert after_snapshot.frontier_oracle.oracle_opportunity_count == (
        before_snapshot.frontier_oracle.oracle_opportunity_count
    )
    assert after_snapshot.frontier_oracle.oracle_opportunity_set_sha256 == (
        before_snapshot.frontier_oracle.oracle_opportunity_set_sha256
    )
    assert after_observation.observation_identities == (
        before_observation.observation_identities
    )
    assert all(
        np.array_equal(
            getattr(after_observation, name).numpy(),
            getattr(before_observation, name).numpy(),
        )
        for name in before_observation.input_names
    )
    assert restored.environment._audit_current_candidate_boundary() == (
        worker.environment._audit_current_candidate_boundary()
    )


def test_physical_worker_replay_preserves_failure_before_later_reveal(
    tmp_path: pathlib.Path,
) -> None:
    """A cleared current failure set must not erase an earlier revision event."""
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory(0, "WHEELED")
    episode = worker.episode
    initial_snapshot = episode._snapshot
    assert initial_snapshot is not None
    candidate_ids = tuple(
        str(initial_snapshot.candidates.candidate_ids[index])
        for index in np.flatnonzero(initial_snapshot.candidates.mask)[:3]
    )
    assert len(candidate_ids) == 3
    kept = episode.refresh_after_planning_failure(
        candidate_ids[0],
        CandidateDisposition.KEEP,
        initial_snapshot.planning_physical_snapshot_id,
    )
    worker.environment._install_observation(kept.next_observation)
    for candidate_id in candidate_ids:
        refreshed = episode.refresh_after_planning_failure(
            candidate_id,
            CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT,
            initial_snapshot.planning_physical_snapshot_id,
        )
        worker.environment._install_observation(refreshed.next_observation)
    evidence = SensorBoundaryEvidence(episode.current_pose, 1.0)
    episode._record_reveal(evidence, "DECISION_BOUNDARY")
    revealed = episode.controller.after_execution(
        platform_type="WHEELED",
        execution_state="DECISION_BOUNDARY",
        evidence=evidence,
    )
    worker.environment._install_observation(revealed.next_observation)

    snapshot = episode._snapshot
    assert snapshot is not None
    assert episode._planner_failed_candidate_ids == set()
    state = FormalWorkerState.from_dict(worker.snapshot_episode_state())
    assert state.replay_event_kinds == (
        "PLANNING_FAILURE_REBUILD",
        "PLANNING_FAILURE_SUPPRESS",
        "PLANNING_FAILURE_SUPPRESS",
        "PLANNING_FAILURE_SUPPRESS",
        "REVEAL",
    )
    restored = assembly.factory.restore_for_episode(
        worker_index=0,
        platform_type="WHEELED",
        episode_cursor=0,
        platform_worker_index=0,
        platform_worker_count=1,
        state=state.to_dict(),
    )

    assert restored.environment.current_observation.observation_identities == (
        worker.environment.current_observation.observation_identities
    )
    assert all(
        np.array_equal(
            getattr(restored.environment.current_observation, name).numpy(),
            getattr(worker.environment.current_observation, name).numpy(),
        )
        for name in worker.environment.current_observation.input_names
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("physical_snapshot_id", "0" * 64),
        lambda value: value.__setitem__(
            "physical_evidence_generation",
            value["physical_evidence_generation"] + 1,
        ),
        lambda value: value.__setitem__("physical_evidence_sha256", "0" * 64),
        lambda value: value.__setitem__(
            "physical_candidate_universe_sha256", "0" * 64
        ),
        lambda value: value["planner_failed_candidate_ids"].__setitem__(
            0, "0" * 64
        ),
    ),
)
def test_physical_worker_replay_rejects_identity_drift(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
) -> None:
    assembly, worker = _worker_with_six_failures(tmp_path, monkeypatch)
    state = worker.snapshot_episode_state()
    mutation(state)

    with pytest.raises(ValueError, match="physical|candidate|replay"):
        assembly.factory.restore_for_episode(
            worker_index=0,
            platform_type="WHEELED",
            episode_cursor=4,
            platform_worker_index=0,
            platform_worker_count=8,
            state=state,
        )


def test_formal_snapshot_rejects_active_ground_option(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_seventy_candidate_universe(monkeypatch)
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory(0, "WHEELED")
    observation = worker.environment.current_observation
    worker.episode.begin_ground_option(
        PolicyAction(0, 0.0),
        observation.observation_identities[0],
    )

    with pytest.raises(ValueError, match="active ground option"):
        worker.snapshot_episode_state()

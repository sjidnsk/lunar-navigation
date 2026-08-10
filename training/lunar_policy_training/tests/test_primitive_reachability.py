from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from lunar_policy_training.environment.primitive_reachability import (
    ObservedPrimitiveReachability,
)


@dataclass
class _NativeSnapshot:
    platform_type: str = "WHEELED"
    width: int = 2
    height: int = 1
    algorithm_id: str = "wheel-primitive-reachability/v1"
    state_schema: str = "wheel-primitive-state/v1"
    primitive_set_sha256: str = "1" * 64
    world_evidence_sha256: str = "2" * 64
    graph_sha256: str = "3" * 64
    revision: int = 1
    invalidated_edge_count: int = 0
    revalidated_edge_count: int = 0

    def __post_init__(self) -> None:
        self.state_ids = np.asarray([7, 9], dtype=np.uint64)
        self.positions_m = np.asarray(
            [[1.0, 2.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float64
        )
        self.yaw_rad = np.zeros(2, dtype=np.float64)
        self.cells = np.asarray([[0, 0], [0, 1]], dtype=np.int32)
        self.yaw_bin = np.zeros(2, dtype=np.int32)
        self.motion_mode = np.asarray([0, 1], dtype=np.int32)
        self.body_z_m = np.zeros((2, 2), dtype=np.float64)
        self.path_cost = np.asarray([0.0, 1.0], dtype=np.float64)
        self.forward_reachable = np.ones(2, dtype=np.bool_)
        self.returnable = np.ones(2, dtype=np.bool_)
        self.observation_state = np.ones(2, dtype=np.bool_)
        self.direct_successor = np.asarray([False, True], dtype=np.bool_)
        self.recoverable = np.ones(2, dtype=np.bool_)
        self.edge_source_ids = np.asarray([7], dtype=np.uint64)
        self.edge_target_ids = np.asarray([9], dtype=np.uint64)
        self.edge_primitive_indices = np.asarray([0], dtype=np.uint32)
        self.edge_primitive_ids = ["forward"]
        self.edge_cost = np.asarray([1.0], dtype=np.float64)


class _Engine:
    def __init__(self) -> None:
        self.calls = 0

    def update(self, request: object, maximum_action_distance_m: float) -> object:
        del request, maximum_action_distance_m
        self.calls += 1
        result = _NativeSnapshot(revision=self.calls)
        if self.calls > 1:
            result.world_evidence_sha256 = "4" * 64
            result.graph_sha256 = "5" * 64
            result.invalidated_edge_count = 1
            result.revalidated_edge_count = 1
        return result

    def reset(self) -> None:
        self.calls = 0


def _request(revision: int, *, request_id: str | None = None) -> object:
    identity = SimpleNamespace(
        w=1.0,
        x=0.0,
        y=0.0,
        z=0.0,
    )
    return SimpleNamespace(
        request_id=(
            request_id
            if request_id is not None
            else f"formal-observed/wheeled/test/{revision}"
        ),
        global_map_generation=revision,
        local_map_generation=revision,
        world=SimpleNamespace(
            global_map=SimpleNamespace(width=2, height=1, resolution_m=1.0),
            local_map=SimpleNamespace(width=2, height=1, resolution_m=1.0),
            map_from_odom=SimpleNamespace(
                translation_m=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                rotation=identity,
            ),
        ),
        config=SimpleNamespace(
            wheel=SimpleNamespace(xy_resolution_m=1.0),
            legged=SimpleNamespace(xy_resolution_m=1.0),
        ),
    )


def test_observed_wrapper_rejects_truth_request_before_native_engine() -> None:
    engine = _Engine()
    graph = ObservedPrimitiveReachability("WHEELED", native_engine=engine)

    with pytest.raises(ValueError, match="observed-only"):
        graph.update(
            _request(1, request_id="formal-cache/truth/wheeled"),
            observation_revision=1,
        )

    assert engine.calls == 0


def test_observed_wrapper_publishes_immutable_monotonic_snapshots() -> None:
    graph = ObservedPrimitiveReachability("WHEELED", native_engine=_Engine())

    first = graph.update(_request(1), observation_revision=1)
    second = graph.update(_request(2), observation_revision=2)

    assert first.revision == 1
    assert second.revision == 2
    assert first.graph_sha256 != second.graph_sha256
    assert second.invalidated_edge_count == 1
    assert second.revalidated_edge_count == 1
    assert first.positions_m.flags.writeable is False
    assert first.edge_source_ids.flags.writeable is False
    with pytest.raises(ValueError):
        first.positions_m[0, 0] = 99.0


def test_observed_wrapper_rejects_skipped_observation_revision() -> None:
    graph = ObservedPrimitiveReachability("WHEELED", native_engine=_Engine())
    graph.update(_request(1), observation_revision=1)

    with pytest.raises(ValueError, match="strictly consecutive"):
        graph.update(_request(3), observation_revision=3)

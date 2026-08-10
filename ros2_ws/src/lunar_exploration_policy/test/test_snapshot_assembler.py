from __future__ import annotations

import math

import numpy as np

from lunar_exploration_policy.grid_map_runtime import DecodedGridMap
from lunar_exploration_policy.snapshot_assembler import (
    MissionDefinition,
    SnapshotAssembler,
)
from lunar_policy_training.environment.candidate_builder import CandidateBuilderV2
from lunar_policy_training.environment.observation_builder import PlatformProjection, Pose2
from lunar_policy_training.environment.visibility import SensorGeometry


class _UnitGainEstimator:
    def __init__(self) -> None:
        self.sensor = SensorGeometry(30.0, 2.0 * math.pi)
        self.resolution_m = 4.0

    def estimate_candidate_gains(self, *args):
        cells = args[-1]
        return np.ones((cells.shape[0], 2), np.float32)


class _AllSafeProjector:
    def project(self, world, platform_type, **_maps):
        return PlatformProjection(
            world.canvas,
            np.ones((256, 256), np.float32),
            np.ones((32, 32), np.float32),
            np.full((256, 256), 0.5, np.float32),
            "cpp_v3/test-projection",
        )


def _grid(frame: str, cells: int, resolution: float, content: str):
    layers = {
        name: np.zeros((cells, cells), np.float32)
        for name in (
            "elevation", "valid_mask", "obstacle", "obstacle_height",
            "observation_age_s", "observation_quality",
            "elevation_variance", "obstacle_variance",
            "observation_count", "forbidden",
        )
    }
    layers["valid_mask"][:] = 1.0
    layers["observation_quality"][:] = 1.0
    return DecodedGridMap(
        frame_id=frame,
        stamp_ns=1_000_000_000,
        width=cells,
        height=cells,
        resolution_m=resolution,
        origin_xy_m=(0.0, 0.0),
        layers=layers,
        content_id=content * 64,
    )


def test_assembler_builds_old_observation_from_global_and_local_maps() -> None:
    global_map = _grid("map", 256, 4.0, "a")
    local_map = _grid("odom", 32, 0.2, "b")
    # 南向起始输入的这块未知区翻转后成为策略图顶部区域。
    global_map.layers["valid_mask"][:16] = 0.0
    assembler = SnapshotAssembler(
        CandidateBuilderV2(_UnitGainEstimator()), _AllSafeProjector()
    )

    snapshot = assembler.build(
        global_map=global_map,
        local_map=local_map,
        pose_map=Pose2(512.0, 512.0, 0.0),
        robot_state_id="odom-1",
        state_time_ns=1_000_000_000,
        mission=MissionDefinition("mission", 3, (0.0, 0.0, 1024.0, 1024.0)),
        platform_type="WHEELED",
    )

    assert snapshot.identity.mission_revision == 3
    assert snapshot.identity.map_snapshot_id == "a" * 64 + ":" + "b" * 64
    assert snapshot.arrays["prior_channels"].shape == (1, 4, 256, 256)
    assert snapshot.arrays["local_crop"].shape == (1, 4, 32, 32)
    assert snapshot.arrays["coverage_summary"][0, 0, :16].all()
    assert not snapshot.arrays["coverage_summary"][0, 0, -16:].any()

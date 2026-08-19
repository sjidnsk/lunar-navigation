from __future__ import annotations

import math
import pathlib
import sys

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.environment.candidate_builder import CandidateBuilderV2  # noqa: E402
from lunar_policy_training.environment.ground_opportunity_index import (  # noqa: E402
    GroundOpportunityIndex,
)
from lunar_policy_training.environment.observation_builder import (  # noqa: E402
    LocalObservation,
    MissionRaster,
    ObservedWorld,
)
from lunar_policy_training.environment.visibility import SensorGeometry  # noqa: E402
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer  # noqa: E402
from lunar_policy_training.polar_data.raster import MapCanvas  # noqa: E402


class _AllGainEstimator:
    def __init__(self) -> None:
        self.sensor = SensorGeometry(8.0, 2.0 * math.pi)

    def estimate_candidate_gains(
        self,
        observed_mask: np.ndarray,
        obstacle_ratio: np.ndarray,
        roi_ratio: np.ndarray,
        priority_weight: np.ndarray,
        candidate_cells: np.ndarray,
    ) -> np.ndarray:
        del observed_mask, obstacle_ratio, roi_ratio, priority_weight
        return np.ones((len(candidate_cells), 2), dtype=np.float32)

    def estimate_candidate_gains_at_positions(
        self,
        observed_mask: np.ndarray,
        obstacle_ratio: np.ndarray,
        roi_ratio: np.ndarray,
        priority_weight: np.ndarray,
        candidate_positions: np.ndarray,
    ) -> np.ndarray:
        del observed_mask, obstacle_ratio, roi_ratio, priority_weight
        return np.ones((len(candidate_positions), 2), dtype=np.float32)


def _world_and_mission(observed: np.ndarray) -> tuple[ObservedWorld, MissionRaster]:
    canvas = MapCanvas.from_roi_bounds("e" * 64, (0.0, 0.0, 25.6, 25.6))
    local = LocalObservation(
        canvas.identity,
        (9.6, 9.6, 16.0, 16.0),
        np.zeros((32, 32), dtype=np.float32),
        np.ones((32, 32), dtype=np.bool_),
        np.zeros((32, 32), dtype=np.float32),
    )
    world = ObservedWorld(
        canvas,
        np.zeros((256, 256), dtype=np.float32),
        observed,
        CanvasRatioLayer(canvas, np.zeros((256, 256), dtype=np.float32)),
        local,
    )
    roi = np.zeros((256, 256), dtype=np.float32)
    roi[100, 100] = 1.0
    roi[102, 100] = 1.0
    roi[100, 108] = 1.0
    return world, MissionRaster(canvas, roi.copy(), roi)


def _positions_for(mask: np.ndarray, canvas: MapCanvas) -> np.ndarray:
    return np.ascontiguousarray(
        [
            (*canvas.grid_center_world(int(row), int(column)), 0.0)
            for row, column in zip(*np.nonzero(mask), strict=True)
        ],
        dtype=np.float64,
    )


def _all_endpoints(positions: np.ndarray) -> np.ndarray:
    return np.ones(len(positions), dtype=np.bool_)


def _all_positive(
    pose_cells: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    del pose_cells
    return np.ones((len(positions), 2), dtype=np.float32)


def test_dirty_tile_update_matches_full_observed_only_reference() -> None:
    observed = np.ones((256, 256), dtype=np.bool_)
    observed[100, 100] = False
    observed[102, 100] = False
    observed[100, 108] = False
    world, mission = _world_and_mission(np.ascontiguousarray(observed))
    reachable = np.zeros((256, 256), dtype=np.bool_)
    reachable[100, 98] = True
    reachable[100, 110] = True
    reachable = np.ascontiguousarray(reachable)
    positions = _positions_for(reachable, world.canvas)
    index = GroundOpportunityIndex(
        task_roi_ratio=mission.roi_ratio,
        sensor_range_m=8.0,
        resolution_m=world.canvas.geometry.resolution_m,
    )

    initial = index.update(
        previous=None,
        observed_mask=world.observed_mask,
        reachable_pose_mask=reachable,
        observation_positions_m=positions,
        dirty_observed_mask=np.zeros_like(reachable),
        reachable_delta_mask=reachable,
        endpoint_feasibility=_all_endpoints,
        estimate_gains=_all_positive,
        observation_generation=1,
        physical_snapshot_id="a" * 64,
        max_pose_count=2,
    )
    reference = CandidateBuilderV2(_AllGainEstimator()).scan_ground_exhaustion_candidates(
        world,
        mission,
        reachable_pose_mask=reachable,
        observation_positions_m=positions,
        ground_endpoint_feasibility=_all_endpoints,
    )

    assert initial.pending_pose_count == 0
    assert initial.state.positive_pose_cells == reference.positive_pose_cells
    assert initial.state.positive_gain_pairs == reference.positive_gain_pairs

    changed_observed = observed.copy()
    changed_observed[100, 100] = True
    changed_world, changed_mission = _world_and_mission(
        np.ascontiguousarray(changed_observed)
    )
    dirty = np.zeros_like(reachable)
    dirty[100, 100] = True
    updated = index.update(
        previous=initial.state,
        observed_mask=changed_world.observed_mask,
        reachable_pose_mask=reachable,
        observation_positions_m=positions,
        dirty_observed_mask=np.ascontiguousarray(dirty),
        reachable_delta_mask=np.zeros_like(reachable),
        endpoint_feasibility=_all_endpoints,
        estimate_gains=_all_positive,
        observation_generation=2,
        physical_snapshot_id="b" * 64,
        max_pose_count=2,
    )
    changed_reference = CandidateBuilderV2(
        _AllGainEstimator()
    ).scan_ground_exhaustion_candidates(
        changed_world,
        changed_mission,
        reachable_pose_mask=reachable,
        observation_positions_m=positions,
        ground_endpoint_feasibility=_all_endpoints,
    )

    assert updated.endpoint_certified_pose_count == 1
    assert updated.exact_gain_evaluated_pose_count == 1
    assert updated.pending_pose_count == 0
    assert updated.state.positive_pose_cells == changed_reference.positive_pose_cells
    assert updated.state.positive_gain_pairs == changed_reference.positive_gain_pairs


def test_cold_index_reaches_reference_in_bounded_slices() -> None:
    cells = 20
    observed = np.ones((cells, cells), dtype=np.bool_)
    task_roi_ratio = np.zeros((cells, cells), dtype=np.float32)
    reachable = np.zeros((cells, cells), dtype=np.bool_)
    for column in (2, 5, 8, 11, 14):
        observed[10, column + 1] = False
        task_roi_ratio[10, column + 1] = 1.0
        reachable[10, column] = True
    observed = np.ascontiguousarray(observed)
    reachable = np.ascontiguousarray(reachable)
    positions = np.ascontiguousarray(
        [
            (float(row), float(column), 0.0)
            for row, column in zip(*np.nonzero(reachable), strict=True)
        ],
        dtype=np.float64,
    )
    index = GroundOpportunityIndex(
        task_roi_ratio=np.ascontiguousarray(task_roi_ratio),
        sensor_range_m=1.0,
        resolution_m=1.0,
    )
    common = dict(
        observed_mask=observed,
        reachable_pose_mask=reachable,
        observation_positions_m=positions,
        dirty_observed_mask=np.zeros_like(reachable),
        reachable_delta_mask=reachable,
        endpoint_feasibility=_all_endpoints,
        estimate_gains=_all_positive,
        observation_generation=1,
        physical_snapshot_id="c" * 64,
        max_pose_count=2,
    )

    first = index.update(previous=None, **common)
    second = index.update(previous=first.state, **common)
    third = index.update(previous=second.state, **common)

    assert first.endpoint_certified_pose_count == 2
    assert first.pending_pose_count == 3
    assert second.endpoint_certified_pose_count == 2
    assert second.pending_pose_count == 1
    assert third.endpoint_certified_pose_count == 1
    assert third.pending_pose_count == 0
    assert third.state.indexed_generation == 1
    assert len(third.state.positive_pose_cells) == 5

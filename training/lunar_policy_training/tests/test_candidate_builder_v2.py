from __future__ import annotations

from collections.abc import Collection
import pathlib
import sys
import math
from dataclasses import fields, replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.environment import candidate_builder as candidate_builder_module  # noqa: E402
from lunar_policy_training.environment.candidate_builder import (  # noqa: E402
    CANDIDATE_ID_SCHEMA,
    CANDIDATE_DIAGNOSTIC_FIELDS,
    PHYSICAL_SNAPSHOT_SCHEMA,
    CandidateBatch,
    CandidateBuildResult,
    CandidateBuilderV2,
    CandidateDiagnostics,
    PhysicalCandidate,
    PhysicalCandidateUniverse,
)
from lunar_policy_training.environment.observation_builder import LocalObservation, MissionRaster, ObservedWorld, PlatformProjection, Pose2  # noqa: E402
from lunar_policy_training.environment.platform_reachability import (  # noqa: E402
    HopperOpportunityAuthority,
    PHYSICAL_PROJECTION_SCHEMA,
    CandidateReachabilityResult,
    PhysicalReachabilityResult,
    PlatformCandidateReachability,
)


class _FakeOpportunityBridge:
    def __init__(
        self,
        *,
        direct_positive: tuple[int, int],
        transit: tuple[int, int],
    ) -> None:
        self.direct_positive = direct_positive
        self.transit = transit
        self.queries: list[np.ndarray] = []

    def query_hopper_opportunity_distance(
        self, context, positive, enumerate_all_reachable_opportunities
    ) -> object:
        del context, enumerate_all_reachable_opportunities
        north_up = np.ascontiguousarray(np.flipud(positive), dtype=np.bool_)
        self.queries.append(north_up.copy())
        progress = np.zeros_like(north_up)
        if bool(north_up.any()):
            target = (
                self.direct_positive
                if north_up[self.direct_positive]
                else self.transit
            )
            progress[target] = True
        return SimpleNamespace(
            direct_progress=np.ascontiguousarray(np.flipud(progress)),
            reachable_opportunities=np.ascontiguousarray(
                np.flipud(north_up)
            ),
            current_hop_distance=1 if north_up.any() else -1,
            has_reachable_opportunity=bool(north_up.any()),
            algorithm_id="cpp-hopper-opportunity-distance/v1",
        )


def _fake_opportunity_authority(
    shape: tuple[int, int],
    *,
    direct_positive: tuple[int, int],
    transit: tuple[int, int],
) -> tuple[HopperOpportunityAuthority, _FakeOpportunityBridge]:
    certified = np.ones(shape, dtype=np.bool_)
    direct = np.zeros(shape, dtype=np.bool_)
    direct[direct_positive] = True
    direct[transit] = True
    bridge = _FakeOpportunityBridge(
        direct_positive=direct_positive, transit=transit
    )
    context = SimpleNamespace(
        direct=np.ascontiguousarray(np.flipud(direct)),
        algorithm_id="cpp-hopper-opportunity-connectivity/v1",
    )
    canvas = _canvas()
    positions = np.asarray(
        [
            (*canvas.grid_center_world(int(row), int(column)), 0.0)
            for row, column in zip(*np.nonzero(certified), strict=True)
        ],
        dtype=np.float64,
    )
    return (
        HopperOpportunityAuthority(
            bridge=bridge,
            context=context,
            certified_mask=certified,
            certified_positions_m=positions,
        ),
        bridge,
    )


def _universe_with_gains(
    universe: PhysicalCandidateUniverse,
    gains: dict[tuple[int, int], float],
) -> PhysicalCandidateUniverse:
    candidates: list[PhysicalCandidate] = []
    for candidate in universe.candidates:
        feature = candidate.feature.copy()
        feature[5] = np.float32(gains.get(candidate.position_grid_key, 0.0))
        rank_key = (
            -float(feature[5]),
            *candidate.rank_key[1:],
        )
        candidates.append(replace(candidate, feature=feature, rank_key=rank_key))
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.rank_key))
    diagnostics = replace(
        universe.diagnostics,
        zero_gain_count=sum(float(candidate.feature[5]) == 0.0 for candidate in ordered),
    )
    return replace(
        universe,
        candidates=ordered,
        universe_sha256=sha256(
            "".join(candidate.candidate_id for candidate in ordered).encode("ascii")
        ).hexdigest(),
        diagnostics=diagnostics,
    )
from lunar_policy_training.environment.visibility import NativeVisibilityEstimator, SensorGeometry  # noqa: E402
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer  # noqa: E402
from lunar_policy_training.polar_data.raster import MapCanvas  # noqa: E402
from lunar_policy_training.training_semantics import FORMAL_SENSOR_FOV_RAD, FORMAL_SENSOR_RANGE_M  # noqa: E402


def test_physical_candidate_contract_symbols_are_public() -> None:
    expected = {
        "CANDIDATE_ID_SCHEMA",
        "PHYSICAL_SNAPSHOT_SCHEMA",
        "CandidateBuildResult",
        "PhysicalCandidate",
        "PhysicalCandidateUniverse",
    }

    assert expected <= set(candidate_builder_module.__all__)
    assert CANDIDATE_ID_SCHEMA == "lunar-physical-candidate-id/v1"
    assert PHYSICAL_SNAPSHOT_SCHEMA == "lunar-physical-snapshot/v1"
    assert CandidateBuildResult is candidate_builder_module.CandidateBuildResult
    assert PhysicalCandidate is candidate_builder_module.PhysicalCandidate
    assert PhysicalCandidateUniverse is candidate_builder_module.PhysicalCandidateUniverse


def _canvas() -> MapCanvas:
    return MapCanvas.from_roi_bounds("d" * 64, (500.0, 500.0, 524.0, 524.0))


def _obstacle_layer(canvas: MapCanvas, values: np.ndarray | None = None) -> CanvasRatioLayer:
    if values is None:
        values = np.zeros((256, 256), dtype=np.float32)
    return CanvasRatioLayer(canvas, values)


def _world_with_frontier(*, occlude: bool = False) -> ObservedWorld:
    observed = np.zeros((256, 256), dtype=bool)
    observed[120:136, 100:128] = True
    if occlude:
        observed[128, 110:120] = False
    canvas = _canvas()
    local = LocalObservation(canvas.identity, (508.8, 508.8, 515.2, 515.2), np.zeros((32, 32), dtype=np.float32), np.ones((32, 32), dtype=bool), np.zeros((32, 32), dtype=np.float32))
    return ObservedWorld(canvas, np.zeros((256, 256), dtype=np.float32), observed, _obstacle_layer(canvas), local)


def _world(observed: np.ndarray) -> ObservedWorld:
    canvas = _canvas()
    local = LocalObservation(canvas.identity, (508.8, 508.8, 515.2, 515.2), np.zeros((32, 32), dtype=np.float32), np.ones((32, 32), dtype=bool), np.zeros((32, 32), dtype=np.float32))
    return ObservedWorld(canvas, np.zeros((256, 256), dtype=np.float32), observed, _obstacle_layer(canvas), local)


def _mission_for_roi(roi: np.ndarray) -> MissionRaster:
    ratio = roi.astype(np.float32)
    return MissionRaster(_canvas(), ratio.copy(), ratio)


def _mission() -> MissionRaster:
    roi = np.zeros((256, 256), dtype=np.float32)
    roi[116:140, 96:144] = 1.0
    return MissionRaster(_canvas(), roi.copy(), roi)


def _projection() -> PlatformProjection:
    return PlatformProjection(
        _canvas(),
        traversable_ratio=np.ones((256, 256), dtype=np.float32),
        local_traversable_ratio=np.ones((32, 32), dtype=np.float32),
        clearance_margin_norm=np.full((256, 256), 0.3, dtype=np.float32),
        source="test_only/proxy",
    )


def _detour_fixture() -> tuple[ObservedWorld, MissionRaster, PlatformProjection, Pose2, tuple[int, int]]:
    canvas = _canvas()
    observed = np.zeros((256, 256), dtype=bool)
    observed[124:133, 124:136] = True
    roi = np.zeros((256, 256), dtype=bool)
    roi[124:133, 124:137] = True
    obstacles = np.zeros((256, 256), dtype=np.float32)
    obstacles[128, 130] = 1.0
    local = LocalObservation(
        canvas.identity,
        (508.8, 508.8, 515.2, 515.2),
        np.zeros((32, 32), dtype=np.float32),
        np.ones((32, 32), dtype=bool),
        np.zeros((32, 32), dtype=np.float32),
    )
    world = ObservedWorld(
        canvas,
        np.zeros((256, 256), dtype=np.float32),
        observed,
        _obstacle_layer(canvas, obstacles),
        local,
    )
    traversable = np.zeros((256, 256), dtype=np.float32)
    traversable[128, 128] = 1.0
    traversable[127, 128:135] = 1.0
    traversable[128, 134] = 1.0
    projection = PlatformProjection(
        canvas,
        traversable_ratio=traversable,
        local_traversable_ratio=np.ones((32, 32), dtype=np.float32),
        clearance_margin_norm=np.full((256, 256), 0.3, dtype=np.float32),
        source="test_only/proxy",
    )
    robot_x_m, robot_y_m = canvas.grid_center_world(128, 128)
    return world, _mission_for_roi(roi), projection, Pose2(robot_x_m, robot_y_m), (128, 134)


class _RecordingEstimator:
    def __init__(self) -> None:
        self.sensor = SensorGeometry(30.0, 2.0 * math.pi)
        self.calls: list[np.ndarray] = []

    def estimate_candidate_gains(
        self,
        observed_mask,
        obstacle_ratio,
        roi_ratio,
        priority_weight,
        candidate_cells,
    ) -> np.ndarray:
        self.calls.append(candidate_cells.copy())
        return np.ones((candidate_cells.shape[0], 2), dtype=np.float32)


def _builder(
    sensor: SensorGeometry | None = None,
) -> CandidateBuilderV2:
    geometry = sensor or SensorGeometry(
        range_m=FORMAL_SENSOR_RANGE_M,
        fov_rad=FORMAL_SENSOR_FOV_RAD,
    )
    return CandidateBuilderV2(
        NativeVisibilityEstimator(
            geometry,
            resolution_m=_canvas().geometry.resolution_m,
        )
    )


def _physical_reachability(
    world: ObservedWorld,
    *,
    platform_type: str = "WHEELED",
    mask: np.ndarray | None = None,
    offset_xy_m: tuple[float, float] = (0.0, 0.0),
    landing_z_m: float | None = None,
) -> PhysicalReachabilityResult:
    physical_mask = np.ascontiguousarray(
        world.observed_mask.copy() if mask is None else mask,
        dtype=np.bool_,
    )
    positions = np.asarray(
        [
            (
                world.canvas.grid_center_world(int(row), int(column))[0]
                + offset_xy_m[0],
                world.canvas.grid_center_world(int(row), int(column))[1]
                + offset_xy_m[1],
                (
                    float(world.elevation_m[row, column])
                    if landing_z_m is None
                    else landing_z_m
                ),
            )
            for row, column in zip(*np.nonzero(physical_mask), strict=True)
        ],
        dtype=np.float64,
    ).reshape((-1, 3))
    return PhysicalReachabilityResult(
        platform_type=platform_type,
        physical_observation_pose_mask=physical_mask,
        observation_positions_m=np.ascontiguousarray(positions),
        physical_projection_schema=PHYSICAL_PROJECTION_SCHEMA,
        physical_reachability_algorithm_id=(
            f"cpp-v3/{platform_type.lower()}-physical-reachability/v1"
        ),
        physical_evidence_algorithm_id=(
            f"cpp-v3/{platform_type.lower()}-physical-evidence/v1"
        ),
        physical_safe_pose_count=int(physical_mask.sum(dtype=np.int64)),
        physically_reachable_pose_count=int(
            physical_mask.sum(dtype=np.int64)
        ),
    )


def _formal_universe(
    builder: CandidateBuilderV2 | None = None,
    *,
    world: ObservedWorld | None = None,
    mission: MissionRaster | None = None,
    pose: Pose2 = Pose2(500.0, 512.0),
    projection: PlatformProjection | None = None,
    platform_type: str = "WHEELED",
    physical_reachability: PhysicalReachabilityResult | None = None,
    evidence_generation: int = 3,
    physical_evidence_sha256: str = "2" * 64,
    excluded_cells: Collection[tuple[int, int]] = (),
    backtrack_pose: Pose2 | None = None,
) -> candidate_builder_module.PhysicalCandidateUniverse:
    resolved_world = world or _world_with_frontier()
    resolved_physical = physical_reachability or _physical_reachability(
        resolved_world, platform_type=platform_type
    )
    return (builder or _builder()).build_physical_universe(
        resolved_world,
        mission or _mission(),
        pose,
        projection or _projection(),
        physical_reachability=resolved_physical,
        platform_type=platform_type,
        platform_id=f"unit-{platform_type.lower()}-1",
        capability_content_sha256="1" * 64,
        mission_revision=7,
        evidence_generation=evidence_generation,
        physical_evidence_sha256=physical_evidence_sha256,
        physical_reachability_algorithm_id=(
            resolved_physical.physical_reachability_algorithm_id
        ),
        goal_tolerance_mm=0 if platform_type == "HOPPER" else 200,
        excluded_cells=excluded_cells,
        backtrack_pose=backtrack_pose,
    )


def _remote_ground_opportunity_fixture() -> tuple[
    ObservedWorld,
    MissionRaster,
    Pose2,
    tuple[int, int],
]:
    canvas = _canvas()
    robot = (128, 125)
    remote = (128, 144)
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[127:130, 124:146] = True
    roi = observed.copy()
    roi[127:130, 146] = True
    robot_x_m, robot_y_m = canvas.grid_center_world(*robot)
    assert math.dist(robot, remote) * canvas.geometry.resolution_m == 76.0
    return (
        _world(observed),
        _mission_for_roi(roi),
        Pose2(robot_x_m, robot_y_m),
        remote,
    )


class _LiteralGainEstimator(_RecordingEstimator):
    def __init__(self, gains: dict[tuple[int, int], float]) -> None:
        super().__init__()
        self._gains = dict(gains)

    def estimate_candidate_gains(self, *args) -> np.ndarray:
        candidates = np.asarray(args[-1], dtype=np.int32)
        self.calls.append(candidates.copy())
        return np.asarray(
            [
                (self._gains.get(tuple(cell), 0.0), 0.0)
                for cell in candidates
            ],
            dtype=np.float32,
        ).reshape((-1, 2))


def test_ground_global_candidate_keeps_positive_opportunity_beyond_sensor_range() -> None:
    world, mission, pose, remote = _remote_ground_opportunity_fixture()
    estimator = _LiteralGainEstimator({remote: 1.0})
    physical = _physical_reachability(world, mask=world.observed_mask)

    universe = _formal_universe(
        CandidateBuilderV2(estimator),
        world=world,
        mission=mission,
        pose=pose,
        physical_reachability=physical,
    )

    assert remote in {
        candidate.position_grid_key for candidate in universe.candidates
    }
    assert all(float(candidate.feature[5]) > 0.0 for candidate in universe.candidates)
    assert any(remote in map(tuple, call) for call in estimator.calls)


def test_ground_global_all_zero_positions_are_diagnostic_only() -> None:
    world, mission, pose, _ = _remote_ground_opportunity_fixture()
    estimator = _LiteralGainEstimator({})
    physical = _physical_reachability(world, mask=world.observed_mask)

    universe = _formal_universe(
        CandidateBuilderV2(estimator),
        world=world,
        mission=mission,
        pose=pose,
        physical_reachability=physical,
    )

    assert universe.candidates == ()
    assert universe.diagnostics.physical_candidate_universe_count == 0
    assert universe.diagnostics.available_candidate_count == 0
    assert universe.diagnostics.selected_policy_candidate_count == 0
    assert universe.diagnostics.untried_reserve_count == 0
    assert universe.diagnostics.zero_gain_count > 0


def test_candidate_builder_requires_explicit_sensor_estimator() -> None:
    with pytest.raises(TypeError):
        CandidateBuilderV2()

    builder = _builder()
    assert builder.sensor == SensorGeometry(
        range_m=30.0,
        fov_rad=2.0 * math.pi,
    )


def test_candidate_diagnostics_own_only_physical_selection_fields() -> None:
    assert tuple(field.name for field in fields(CandidateDiagnostics)) == (
        "physical_snapshot_id",
        "physical_reachability_algorithm_id",
        "physical_candidate_universe_count",
        "selected_policy_candidate_count",
        "available_candidate_count",
        "untried_reserve_count",
        "planner_failed_current_snapshot_count",
        "zero_gain_count",
        "visited_excluded_count",
        "physical_unreachable_count",
    )
    assert CANDIDATE_DIAGNOSTIC_FIELDS == tuple(
        field.name for field in fields(CandidateDiagnostics)
    )


def test_physical_universe_requires_explicit_canonical_identity() -> None:
    with pytest.raises(TypeError):
        _builder().build_physical_universe(
            _world_with_frontier(),
            _mission(),
            Pose2(500.0, 512.0),
            _projection(),
            physical_reachability=_physical_reachability(
                _world_with_frontier()
            ),
            platform_type="WHEELED",
        )

    with pytest.raises(ValueError, match="capability content hash"):
        _builder().build_physical_universe(
            _world_with_frontier(),
            _mission(),
            Pose2(500.0, 512.0),
            _projection(),
            physical_reachability=_physical_reachability(
                _world_with_frontier()
            ),
            platform_type="WHEELED",
            platform_id="unit-wheel-1",
            capability_content_sha256="",
            mission_revision=7,
            evidence_generation=3,
            physical_evidence_sha256="2" * 64,
            physical_reachability_algorithm_id="cpp-v3/wheel/v1",
            goal_tolerance_mm=200,
        )


def test_hopper_candidates_bind_exact_certified_landing_positions_and_z() -> None:
    world = _world_with_frontier()
    physical = _physical_reachability(
        world,
        platform_type="HOPPER",
        offset_xy_m=(0.02, -0.02),
        landing_z_m=123.456,
    )
    universe = _formal_universe(
        world=world,
        platform_type="HOPPER",
        physical_reachability=physical,
    )
    position_by_cell = {
        tuple(cell): tuple(position)
        for cell, position in zip(
            zip(*np.nonzero(physical.physical_observation_pose_mask), strict=True),
            physical.observation_positions_m,
            strict=True,
        )
    }

    assert universe.candidates
    for candidate in universe.candidates:
        expected = position_by_cell[candidate.position_grid_key]
        assert candidate.target_position_m == expected
        coarse_x, coarse_y = world.canvas.grid_center_world(
            *candidate.position_grid_key
        )
        assert candidate.target_position_m[:2] != (coarse_x, coarse_y)
        assert candidate.target_position_m[2] == 123.456

    changed = _formal_universe(
        world=world,
        platform_type="HOPPER",
        physical_reachability=_physical_reachability(
            world,
            platform_type="HOPPER",
            offset_xy_m=(0.02, -0.02),
            landing_z_m=123.457,
        ),
    )
    assert {
        candidate.candidate_id for candidate in changed.candidates
    }.isdisjoint(
        candidate.candidate_id for candidate in universe.candidates
    )


def test_candidate_ids_use_exact_hopper_landing_xy_but_ground_grid_key() -> None:
    world = _world_with_frontier()
    physical_mask = np.zeros_like(world.observed_mask)
    physical_mask[122, 127] = True
    first_offset = (0.020, -0.020)
    second_offset = (0.022, -0.020)

    hopper_first = _formal_universe(
        world=world,
        platform_type="HOPPER",
        physical_reachability=_physical_reachability(
            world,
            platform_type="HOPPER",
            mask=physical_mask,
            offset_xy_m=first_offset,
            landing_z_m=123.456,
        ),
    )
    hopper_second = _formal_universe(
        world=world,
        platform_type="HOPPER",
        physical_reachability=_physical_reachability(
            world,
            platform_type="HOPPER",
            mask=physical_mask,
            offset_xy_m=second_offset,
            landing_z_m=123.456,
        ),
    )
    first_by_cell = {
        candidate.position_grid_key: candidate
        for candidate in hopper_first.candidates
    }
    second_by_cell = {
        candidate.position_grid_key: candidate
        for candidate in hopper_second.candidates
    }

    assert len(first_by_cell) == len(hopper_first.candidates) == 1
    assert len(second_by_cell) == len(hopper_second.candidates) == 1
    assert first_by_cell.keys() == second_by_cell.keys()
    for cell, first in first_by_cell.items():
        second = second_by_cell[cell]
        assert second.target_position_m[0] - first.target_position_m[0] == pytest.approx(0.002)
        assert second.target_position_m[1:] == first.target_position_m[1:]
        assert second.target_yaw_bin == first.target_yaw_bin
        assert second.candidate_id != first.candidate_id
    assert hopper_second.universe_sha256 != hopper_first.universe_sha256

    ground_first = _formal_universe(
        world=world,
        physical_reachability=_physical_reachability(
            world,
            mask=physical_mask,
            offset_xy_m=first_offset,
            landing_z_m=123.456,
        ),
    )
    ground_second = _formal_universe(
        world=world,
        physical_reachability=_physical_reachability(
            world,
            mask=physical_mask,
            offset_xy_m=second_offset,
            landing_z_m=123.456,
        ),
    )
    assert tuple(
        candidate.candidate_id for candidate in ground_second.candidates
    ) == tuple(candidate.candidate_id for candidate in ground_first.candidates)
    assert ground_second.universe_sha256 == ground_first.universe_sha256


def test_candidate_identity_rejects_out_of_range_millimetre_key() -> None:
    world = _world_with_frontier()
    physical = _physical_reachability(
        world,
        platform_type="HOPPER",
        landing_z_m=np.finfo(np.float64).max,
    )

    with pytest.raises(ValueError, match="millimetre"):
        _formal_universe(
            world=world,
            platform_type="HOPPER",
            physical_reachability=physical,
        )


def test_physical_mask_false_rejects_candidate_and_counts_unreachable() -> None:
    world = _world_with_frontier()
    baseline = _formal_universe(world=world)
    rejected_cell = baseline.candidates[0].position_grid_key
    mask = world.observed_mask.copy()
    mask[rejected_cell] = False

    rejected = _formal_universe(
        world=world,
        physical_reachability=_physical_reachability(world, mask=mask),
    )

    assert rejected_cell not in {
        candidate.position_grid_key for candidate in rejected.candidates
    }
    assert (
        rejected.diagnostics.physical_unreachable_count
        > baseline.diagnostics.physical_unreachable_count
    )


def test_physical_ids_are_stable_across_frontier_fallback_and_generation_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world_with_frontier()
    builder = CandidateBuilderV2(_RecordingEstimator())
    with monkeypatch.context() as scoped:
        scoped.setattr(builder, "_fallback_observation_poses", lambda *args: [])
        discovery = _formal_universe(builder, world=world)
    rejected_cell = discovery.candidates[0].position_grid_key
    physical_mask = world.observed_mask.copy()
    physical_mask[rejected_cell] = False
    physical = _physical_reachability(world, mask=physical_mask)
    with monkeypatch.context() as scoped:
        scoped.setattr(builder, "_fallback_observation_poses", lambda *args: [])
        frontier_only = _formal_universe(
            builder,
            world=world,
            physical_reachability=physical,
        )
    targets = [
        rejected_cell,
        *(candidate.position_grid_key for candidate in frontier_only.candidates),
    ]
    with monkeypatch.context() as scoped:
        scoped.setattr(
            builder,
            "_fallback_observation_poses",
            lambda *args: targets,
        )
        duplicated = _formal_universe(
            builder,
            world=world,
            physical_reachability=physical,
        )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            builder,
            "_fallback_observation_poses",
            lambda *args: targets[::-1],
        )
        reordered = _formal_universe(
            builder,
            world=world,
            physical_reachability=physical,
        )

    expected_ids = tuple(candidate.candidate_id for candidate in frontier_only.candidates)
    assert tuple(candidate.candidate_id for candidate in duplicated.candidates) == expected_ids
    assert tuple(candidate.candidate_id for candidate in reordered.candidates) == expected_ids
    assert duplicated.universe_sha256 == frontier_only.universe_sha256
    assert reordered.universe_sha256 == frontier_only.universe_sha256
    assert duplicated.diagnostics == frontier_only.diagnostics
    assert reordered.diagnostics == frontier_only.diagnostics


def test_no_reveal_and_primitive_source_changes_preserve_physical_identities() -> None:
    first = _formal_universe()
    second = _formal_universe()
    projection = _projection()
    renamed = PlatformProjection(
        projection.canvas,
        projection.traversable_ratio,
        projection.local_traversable_ratio,
        projection.clearance_margin_norm,
        source="cpp_v3/renamed-and-reordered-primitives",
    )
    primitive_changed = _formal_universe(projection=renamed)

    assert second.physical_snapshot_id == first.physical_snapshot_id
    assert second.universe_sha256 == first.universe_sha256
    assert first.universe_sha256 == sha256(
        "".join(
            candidate.candidate_id for candidate in first.candidates
        ).encode("ascii")
    ).hexdigest()
    assert tuple(candidate.candidate_id for candidate in second.candidates) == tuple(
        candidate.candidate_id for candidate in first.candidates
    )
    assert primitive_changed.physical_snapshot_id == first.physical_snapshot_id
    assert primitive_changed.universe_sha256 == first.universe_sha256


def test_hopper_universe_ignores_history_while_selection_preserves_tiers_and_exact_parent() -> None:
    world = _world_with_frontier()
    positive_cell = (128, 123)
    parent_cell = (128, 124)

    class MixedGainEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            candidate_cells = args[-1]
            self.calls.append(candidate_cells.copy())
            gains = np.zeros((len(candidate_cells), 2), dtype=np.float32)
            positive = np.all(
                candidate_cells == np.asarray(positive_cell), axis=1
            )
            gains[positive] = 1.0
            return gains

    builder = CandidateBuilderV2(MixedGainEstimator())
    physical = _physical_reachability(
        world,
        platform_type="HOPPER",
        offset_xy_m=(0.02, -0.02),
        landing_z_m=123.456,
    )
    physical_positions = {
        tuple(cell): tuple(float(value) for value in position)
        for cell, position in zip(
            zip(*np.nonzero(physical.physical_observation_pose_mask), strict=True),
            physical.observation_positions_m,
            strict=True,
        )
    }
    parent = Pose2(*physical_positions[parent_cell][:2], elevation_m=123.456)
    visited = set(map(tuple, np.column_stack(np.nonzero(world.observed_mask))))

    canonical = _formal_universe(
        builder,
        world=world,
        platform_type="HOPPER",
        physical_reachability=physical,
    )
    history_bound = _formal_universe(
        builder,
        world=world,
        platform_type="HOPPER",
        physical_reachability=physical,
        excluded_cells=visited,
        backtrack_pose=parent,
    )

    assert history_bound.physical_snapshot_id == canonical.physical_snapshot_id
    assert history_bound.universe_sha256 == canonical.universe_sha256
    assert tuple(candidate.candidate_id for candidate in history_bound.candidates) == tuple(
        candidate.candidate_id for candidate in canonical.candidates
    )

    observation = builder.select_available(
        canonical,
        canvas_id=world.canvas.identity,
    )
    assert observation.batch.count > 0
    assert np.all(observation.batch.features[observation.batch.mask, 5] > 0.0)

    transit = builder.select_available(
        canonical,
        canvas_id=world.canvas.identity,
        excluded_cells={positive_cell},
    )
    assert transit.batch.count > 0
    assert np.all(transit.batch.features[transit.batch.mask, 5:7] == 0.0)

    backtrack = builder.select_available(
        canonical,
        canvas_id=world.canvas.identity,
        excluded_cells={
            candidate.position_grid_key for candidate in canonical.candidates
        },
        backtrack_pose=parent,
    )
    assert backtrack.universe.universe_sha256 == canonical.universe_sha256
    assert backtrack.batch.count == 1
    np.testing.assert_array_equal(
        backtrack.batch.target_positions_m[0],
        np.asarray(physical_positions[parent_cell], dtype=np.float64),
    )


def test_hopper_failed_direct_positive_selects_remote_progress_transit() -> None:
    world = _world_with_frontier()
    builder = CandidateBuilderV2(_RecordingEstimator())
    physical = _physical_reachability(world, platform_type="HOPPER")
    universe = _formal_universe(
        builder,
        world=world,
        platform_type="HOPPER",
        physical_reachability=physical,
    )
    direct_positive = universe.candidates[0]
    transit = next(
        candidate
        for candidate in universe.candidates
        if candidate.position_grid_key != direct_positive.position_grid_key
    )
    authority, bridge = _fake_opportunity_authority(
        world.observed_mask.shape,
        direct_positive=direct_positive.position_grid_key,
        transit=transit.position_grid_key,
    )
    remote = next(
        candidate.position_grid_key
        for candidate in universe.candidates
        if candidate.position_grid_key
        not in (direct_positive.position_grid_key, transit.position_grid_key)
    )
    positive_mask = np.zeros_like(world.observed_mask)
    positive_mask[direct_positive.position_grid_key] = True
    positive_mask[remote] = True

    universe = _universe_with_gains(
        universe, {direct_positive.position_grid_key: 1.0}
    )
    by_cell = {
        candidate.position_grid_key: candidate for candidate in universe.candidates
    }
    direct_positive = by_cell[direct_positive.position_grid_key]
    transit = by_cell[transit.position_grid_key]
    progress_universe = replace(
        universe,
        candidates=(direct_positive, transit),
        universe_sha256=sha256(
            f"{direct_positive.candidate_id}{transit.candidate_id}".encode("ascii")
        ).hexdigest(),
        diagnostics=replace(
            universe.diagnostics,
            physical_candidate_universe_count=2,
            selected_policy_candidate_count=1,
            available_candidate_count=2,
            untried_reserve_count=1,
            zero_gain_count=1,
        ),
    )
    selected = builder.select_available(
        progress_universe,
        canvas_id=world.canvas.identity,
        failure_snapshot_id=progress_universe.physical_snapshot_id,
        planner_failed_candidate_ids={direct_positive.candidate_id},
    )

    assert selected.batch.count == 1
    assert tuple(selected.batch.candidate_ids[selected.batch.mask]) == (
        transit.candidate_id,
    )
    del authority, bridge, positive_mask, remote


def test_hopper_all_zero_gain_has_no_transit_candidate() -> None:
    world = _world_with_frontier()
    builder = CandidateBuilderV2(_RecordingEstimator())
    universe = _universe_with_gains(_formal_universe(builder), {})
    direct = universe.candidates[0].position_grid_key
    transit = universe.candidates[1].position_grid_key
    authority, bridge = _fake_opportunity_authority(
        world.observed_mask.shape,
        direct_positive=direct,
        transit=transit,
    )

    assert universe.diagnostics.zero_gain_count == len(universe.candidates)
    assert not np.zeros_like(world.observed_mask).any()
    del authority, bridge


def test_hopper_build_time_authority_admits_only_direct_positive_and_progress_transit() -> None:
    world = _world_with_frontier()
    direct = (128, 123)
    transit = (128, 124)
    remote = (128, 125)
    authority, bridge = _fake_opportunity_authority(
        world.observed_mask.shape,
        direct_positive=direct,
        transit=transit,
    )
    positive_mask = np.zeros_like(world.observed_mask)
    positive_mask[direct] = True
    positive_mask[remote] = True
    physical = replace(
        _physical_reachability(world, platform_type="HOPPER"),
        hopper_opportunity_authority=authority,
    )
    builder = CandidateBuilderV2(_RecordingEstimator())
    builder.hopper_positive_mask = lambda *args: np.ascontiguousarray(positive_mask)

    universe = _formal_universe(
        builder,
        world=world,
        platform_type="HOPPER",
        physical_reachability=physical,
    )

    cells = {candidate.position_grid_key for candidate in universe.candidates}
    assert direct in cells
    assert transit in cells
    assert remote not in cells
    assert not bridge.queries[0][direct]
    assert bridge.queries[0][remote]


def test_hopper_build_time_all_zero_has_empty_canonical_universe() -> None:
    world = _world_with_frontier()
    direct = (128, 123)
    transit = (128, 124)
    authority, bridge = _fake_opportunity_authority(
        world.observed_mask.shape,
        direct_positive=direct,
        transit=transit,
    )
    physical = replace(
        _physical_reachability(world, platform_type="HOPPER"),
        hopper_opportunity_authority=authority,
    )
    builder = CandidateBuilderV2(_RecordingEstimator())
    builder.hopper_positive_mask = lambda *args: np.zeros_like(world.observed_mask)

    universe = _formal_universe(
        builder,
        world=world,
        platform_type="HOPPER",
        physical_reachability=physical,
    )

    assert universe.candidates == ()
    assert universe.diagnostics.available_candidate_count == 0
    assert len(bridge.queries) == 1
    assert not bridge.queries[0].any()


def test_hopper_without_a_direct_first_hop_does_not_query_remote_gain() -> None:
    world = _world_with_frontier()
    certified = np.zeros_like(world.observed_mask, dtype=np.bool_)
    certified[128, 128] = True
    context = SimpleNamespace(
        direct=np.zeros_like(world.observed_mask, dtype=np.bool_),
        algorithm_id="cpp-hopper-opportunity-connectivity/v1",
    )

    class RejectingBridge:
        def query_hopper_opportunity_distance(self, *args, **kwargs):
            raise AssertionError("remote opportunities require a direct first hop")

    canvas = world.canvas
    positions = np.asarray(
        [
            (*canvas.grid_center_world(int(row), int(column)), 0.0)
            for row, column in zip(*np.nonzero(certified), strict=True)
        ],
        dtype=np.float64,
    )
    authority = HopperOpportunityAuthority(
        bridge=RejectingBridge(),
        context=context,
        certified_mask=certified,
        certified_positions_m=positions,
    )
    physical = replace(
        _physical_reachability(world, platform_type="HOPPER"),
        physical_observation_pose_mask=np.zeros_like(world.observed_mask),
        observation_positions_m=np.empty((0, 3), dtype=np.float64),
        physically_reachable_pose_count=0,
        hopper_opportunity_authority=authority,
    )
    positive_mask = np.zeros_like(world.observed_mask)
    positive_mask[128, 128] = True
    builder = CandidateBuilderV2(_RecordingEstimator())
    builder.hopper_positive_mask = lambda *args: np.ascontiguousarray(
        positive_mask
    )

    universe = _formal_universe(
        builder,
        world=world,
        platform_type="HOPPER",
        physical_reachability=physical,
    )

    assert universe.candidates == ()
    assert universe.diagnostics.available_candidate_count == 0


def test_hopper_exact_parent_requires_opportunity_distance_decrease() -> None:
    world = _world_with_frontier()
    builder = CandidateBuilderV2(_RecordingEstimator())
    universe = _formal_universe(builder)
    parent = universe.candidates[0]
    other = universe.candidates[1]
    authority, _ = _fake_opportunity_authority(
        world.observed_mask.shape,
        direct_positive=other.position_grid_key,
        transit=other.position_grid_key,
    )
    positive_mask = np.zeros_like(world.observed_mask)
    positive_mask[other.position_grid_key] = True
    parent_pose = Pose2(*parent.target_position_m[:2], elevation_m=parent.target_position_m[2])

    progress_universe = replace(
        universe,
        candidates=(other,),
        universe_sha256=sha256(other.candidate_id.encode("ascii")).hexdigest(),
        diagnostics=replace(
            universe.diagnostics,
            physical_candidate_universe_count=1,
            selected_policy_candidate_count=1,
            available_candidate_count=1,
            untried_reserve_count=0,
            zero_gain_count=int(float(other.feature[5]) == 0.0),
        ),
    )
    rejected = builder.select_available(
        progress_universe,
        canvas_id=world.canvas.identity,
        excluded_cells={other.position_grid_key},
        backtrack_pose=parent_pose,
    )

    assert rejected.batch.count == 0
    del authority, positive_mask


def test_ground_universe_ignores_visited_cells_but_availability_excludes_them() -> None:
    builder = CandidateBuilderV2(_RecordingEstimator())
    canonical = _formal_universe(builder)
    visited_candidate = canonical.candidates[0]
    history_bound = _formal_universe(
        builder,
        excluded_cells={visited_candidate.position_grid_key},
    )

    assert history_bound.physical_snapshot_id == canonical.physical_snapshot_id
    assert history_bound.universe_sha256 == canonical.universe_sha256
    assert tuple(candidate.candidate_id for candidate in history_bound.candidates) == tuple(
        candidate.candidate_id for candidate in canonical.candidates
    )

    selected = builder.select_available(
        canonical,
        canvas_id=_canvas().identity,
        excluded_cells={visited_candidate.position_grid_key},
    )

    assert selected.universe.universe_sha256 == canonical.universe_sha256
    assert visited_candidate.candidate_id not in set(
        selected.batch.candidate_ids[selected.batch.mask]
    )
    assert selected.batch.diagnostics.visited_excluded_count == 1
    assert selected.batch.diagnostics.available_candidate_count == (
        len(canonical.candidates) - 1
    )


def test_current_snapshot_failures_refill_from_reserve_and_padding_ids_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[110:146, 110:146] = True
    roi = observed.copy()
    roi[109:147, 109:147] = True
    world = _world(observed)
    mission = _mission_for_roi(roi)
    builder = CandidateBuilderV2(_RecordingEstimator())
    reserve_cells = [
        cell
        for cell in zip(*np.nonzero(observed), strict=True)
        if cell != (128, 128)
    ][:80]
    monkeypatch.setattr(
        builder,
        "_fallback_observation_poses",
        lambda *args: reserve_cells,
    )
    universe = _formal_universe(
        builder,
        world=world,
        mission=mission,
        pose=Pose2(*_canvas().grid_center_world(128, 128)),
    )
    assert len(universe.candidates) > 64
    first_ids = tuple(candidate.candidate_id for candidate in universe.candidates[:65])

    selected = builder.select_available(
        universe,
        canvas_id=_canvas().identity,
        failure_snapshot_id=universe.physical_snapshot_id,
        planner_failed_candidate_ids={first_ids[0]},
    )

    assert tuple(selected.batch.candidate_ids[selected.batch.mask]) == first_ids[1:65]
    assert selected.universe.physical_snapshot_id == universe.physical_snapshot_id
    assert selected.universe.universe_sha256 == universe.universe_sha256
    assert np.all(selected.batch.candidate_ids[~selected.batch.mask] == "")
    assert selected.batch.diagnostics.selected_policy_candidate_count == 64
    assert selected.batch.diagnostics.planner_failed_current_snapshot_count == 1
    assert selected.batch.diagnostics.untried_reserve_count == len(universe.candidates) - 65
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in universe.candidates
    }
    for row, candidate_id in enumerate(first_ids[1:65]):
        np.testing.assert_array_equal(
            selected.batch.features[row],
            candidate_by_id[candidate_id].feature,
        )


def test_current_snapshot_unknown_failed_id_fails_closed_but_old_snapshot_is_dropped() -> None:
    builder = CandidateBuilderV2(_RecordingEstimator())
    first = _formal_universe(builder)
    with pytest.raises(ValueError, match="current physical universe"):
        builder.select_available(
            first,
            canvas_id=_canvas().identity,
            failure_snapshot_id=first.physical_snapshot_id,
            planner_failed_candidate_ids={"f" * 64},
        )

    moved = _formal_universe(
        builder,
        pose=Pose2(500.002, 512.0),
    )
    evidence_updated = _formal_universe(
        builder,
        evidence_generation=4,
        physical_evidence_sha256="3" * 64,
    )
    failed_id = first.candidates[0].candidate_id
    assert moved.physical_snapshot_id != first.physical_snapshot_id
    assert evidence_updated.physical_snapshot_id != first.physical_snapshot_id
    assert failed_id in {
        candidate.candidate_id for candidate in evidence_updated.candidates
    }
    selected = builder.select_available(
        evidence_updated,
        canvas_id=_canvas().identity,
        failure_snapshot_id=first.physical_snapshot_id,
        planner_failed_candidate_ids={failed_id},
    )
    assert failed_id in set(selected.batch.candidate_ids[selected.batch.mask])
    assert selected.batch.diagnostics.planner_failed_current_snapshot_count == 0


@pytest.mark.parametrize(
    "malformation",
    (
        "active_blank",
        "duplicate",
        "not_in_universe",
        "target_mismatch",
        "feature_mismatch",
    ),
)
def test_formal_candidate_result_rejects_malformed_active_batch(
    malformation: str,
) -> None:
    builder = CandidateBuilderV2(_RecordingEstimator())
    universe = _formal_universe(builder)
    selected = builder.select_available(
        universe,
        canvas_id=_canvas().identity,
    )
    assert selected.batch.count >= 2
    features = selected.batch.features.copy()
    mask = selected.batch.mask.copy()
    elevations = selected.batch.target_elevation_m.copy()
    positions = selected.batch.target_positions_m.copy()
    yaw = selected.batch.target_yaw_rad.copy()
    candidate_ids = selected.batch.candidate_ids.copy()
    if malformation == "active_blank":
        candidate_ids[0] = ""
    elif malformation == "duplicate":
        candidate_ids[1] = candidate_ids[0]
    elif malformation == "not_in_universe":
        candidate_ids[0] = "f" * 64
    elif malformation == "target_mismatch":
        positions[0, 0] += 0.001
    else:
        assert not np.array_equal(features[0], features[1])
        features[0] = features[1]
    malformed = CandidateBatch(
        features=features,
        mask=mask,
        canvas_id=selected.batch.canvas_id,
        diagnostics=selected.batch.diagnostics,
        target_elevation_m=elevations,
        target_positions_m=positions,
        target_yaw_rad=yaw,
        candidate_ids=candidate_ids,
    )

    with pytest.raises(ValueError, match="formal candidate batch"):
        CandidateBuildResult(selected.universe, malformed)


def test_formal_candidate_result_owns_readonly_arrays_without_freezing_callers() -> None:
    builder = CandidateBuilderV2(_RecordingEstimator())
    selected = builder.select_available(
        _formal_universe(builder),
        canvas_id=_canvas().identity,
    )
    caller_arrays = (
        selected.batch.features.copy(),
        selected.batch.mask.copy(),
        selected.batch.target_elevation_m.copy(),
        selected.batch.target_positions_m.copy(),
        selected.batch.target_yaw_rad.copy(),
        selected.batch.candidate_ids.copy(),
    )
    caller_batch = CandidateBatch(
        features=caller_arrays[0],
        mask=caller_arrays[1],
        canvas_id=selected.batch.canvas_id,
        diagnostics=selected.batch.diagnostics,
        target_elevation_m=caller_arrays[2],
        target_positions_m=caller_arrays[3],
        target_yaw_rad=caller_arrays[4],
        candidate_ids=caller_arrays[5],
    )

    result = CandidateBuildResult(selected.universe, caller_batch)
    result_arrays = (
        result.batch.features,
        result.batch.mask,
        result.batch.target_elevation_m,
        result.batch.target_positions_m,
        result.batch.target_yaw_rad,
        result.batch.candidate_ids,
    )

    assert all(array.flags.writeable for array in caller_arrays)
    assert all(not array.flags.writeable for array in result_arrays)
    assert all(
        not np.shares_memory(caller, owned)
        for caller, owned in zip(caller_arrays, result_arrays, strict=True)
    )
    legacy = _builder().build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0),
        _projection(),
        platform_type="WHEELED",
    )
    assert legacy.count > 0
    assert np.all(legacy.candidate_ids[legacy.mask] == "")


def test_4097_qualified_positions_compress_to_stable_4096_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[80:176, 80:176] = True
    roi = observed.copy()
    roi[79:177, 79:177] = True
    cells = [
        cell
        for cell in zip(*np.nonzero(observed), strict=True)
        if cell != (128, 128)
    ][:4097]

    class WideEstimator(_RecordingEstimator):
        def __init__(self) -> None:
            super().__init__()
            self.sensor = SensorGeometry(2000.0, 2.0 * math.pi)

    builder = CandidateBuilderV2(WideEstimator())
    monkeypatch.setattr(candidate_builder_module, "_spaced_anchors", lambda *args: [])
    monkeypatch.setattr(
        builder,
        "_fallback_observation_poses",
        lambda *args: cells,
    )
    arguments = {
        "builder": builder,
        "world": _world(observed),
        "mission": _mission_for_roi(roi),
        "pose": Pose2(*_canvas().grid_center_world(128, 128)),
    }

    first = _formal_universe(**arguments)
    second = _formal_universe(**arguments)

    assert len(first.candidates) == 4096
    assert len({candidate.candidate_id for candidate in first.candidates}) == 4096
    assert tuple(candidate.candidate_id for candidate in second.candidates) == tuple(
        candidate.candidate_id for candidate in first.candidates
    )
    assert second.universe_sha256 == first.universe_sha256


def test_candidate_builder_is_observed_only_uses_exact_12_fields_and_stable_64_padding() -> None:
    batch = _builder().build(_world_with_frontier(), _mission(), Pose2(500.0, 512.0), _projection(), platform_type="WHEELED")

    assert batch.features.shape == (64, 12)
    assert batch.mask.shape == (64,)
    assert batch.mask.dtype == np.bool_
    assert batch.mask.any()
    assert not batch.mask[batch.count :].any()
    assert np.all(batch.features[~batch.mask] == 0.0)
    valid = batch.features[batch.mask]
    assert np.all(np.isfinite(valid))
    assert np.all(np.abs(valid[:, :2]) <= 1.0)
    assert np.all(valid[:, 10] == 0.3)
    again = _builder().build(_world_with_frontier(), _mission(), Pose2(500.0, 512.0), _projection(), platform_type="WHEELED")
    np.testing.assert_array_equal(batch.features, again.features)
    np.testing.assert_array_equal(batch.mask, again.mask)


def test_candidate_gain_ratios_use_fractional_roi_area_instead_of_cell_count() -> None:
    mission = _mission()
    mission = MissionRaster(_canvas(), mission.priority, mission.roi_ratio * 0.1)
    batch = _builder().build(_world_with_frontier(), mission, Pose2(500.0, 512.0), _projection(), platform_type="WHEELED")
    valid = batch.features[batch.mask]
    assert np.all(valid[:, 5] <= 1.0)
    assert np.all(valid[:, 6] <= 1.0)
    assert np.all(valid[:, 11] <= 1.0)


def test_candidate_gain_features_are_normalized_against_the_reachable_batch() -> None:
    class UnequalGainEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            candidates = args[-1]
            self.calls.append(candidates.copy())
            gains = np.full((len(candidates), 2), (8.0, 16.0), np.float32)
            gains[0] = (2.0, 4.0)
            return gains

    batch = CandidateBuilderV2(UnequalGainEstimator()).build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0),
        _projection(),
        platform_type="WHEELED",
    )

    assert batch.count >= 2
    valid = batch.features[batch.mask]
    np.testing.assert_allclose(np.unique(valid[:, 5]), [0.25, 1.0])
    np.testing.assert_allclose(np.unique(valid[:, 6]), [0.25, 1.0])


def test_candidate_builder_returns_all_false_instead_of_robot_fallback_when_los_has_no_frontier() -> None:
    observed = np.ones((256, 256), dtype=bool)
    canvas = _canvas()
    world = ObservedWorld(canvas, np.zeros((256, 256), dtype=np.float32), observed, _obstacle_layer(canvas), LocalObservation(canvas.identity, (508.8, 508.8, 515.2, 515.2), np.zeros((32, 32), dtype=np.float32), np.ones((32, 32), dtype=bool), np.zeros((32, 32), dtype=np.float32)))
    batch = _builder().build(world, _mission(), Pose2(512.0, 512.0), _projection(), platform_type="WHEELED")
    assert batch.count == 0
    np.testing.assert_array_equal(batch.features, CandidateBatch.empty().features)
    np.testing.assert_array_equal(batch.mask, CandidateBatch.empty().mask)


def test_candidate_builder_excludes_the_robot_cell_from_exploration_targets() -> None:
    canvas = _canvas()
    robot_cell = (128, 128)
    robot_x_m, robot_y_m = canvas.grid_center_world(*robot_cell)
    observed = np.zeros((256, 256), dtype=bool)
    observed[robot_cell] = True
    roi = np.zeros((256, 256), dtype=bool)
    roi[127:130, 127:130] = True

    batch = _builder().build(
        _world(observed),
        _mission_for_roi(roi),
        Pose2(robot_x_m, robot_y_m),
        _projection(),
        platform_type="WHEELED",
    )

    assert batch.count == 0


def test_candidate_builder_excludes_visited_landing_cells_only() -> None:
    batch = _builder().build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0),
        _projection(),
        platform_type="WHEELED",
        excluded_cells={(121, 121)},
    )

    positions = batch.features[batch.mask, :2]
    assert batch.count == 23
    assert not np.any(
        np.all(positions == np.asarray([0.474609375, 0.474609375]), axis=1)
    )
    assert np.any(
        np.all(positions == np.asarray([0.474609375, 0.525390625]), axis=1)
    )
    assert batch.diagnostics.visited_excluded_count == 1


def test_sensor_geometry_and_obstacle_or_zero_traversable_reject_candidates() -> None:
    sensor = SensorGeometry(range_m=24.0, fov_rad=2.0 * np.pi)
    assert sensor.anchor_spacing_m > 0.0
    assert sensor.standoff_m > 0.0
    world = _world_with_frontier()
    blocked = ObservedWorld(world.canvas, world.elevation_m, world.observed_mask, _obstacle_layer(world.canvas, np.where(world.observed_mask, 1.0, 0.0).astype(np.float32)), world.local)
    assert _builder(sensor).build(blocked, _mission(), Pose2(500.0, 512.0), _projection(), platform_type="WHEELED").count == 0
    projection = _projection()
    zero = PlatformProjection(
        projection.canvas,
        traversable_ratio=np.zeros((256, 256), dtype=np.float32), local_traversable_ratio=projection.local_traversable_ratio,
        clearance_margin_norm=projection.clearance_margin_norm, source="test_only/proxy",
    )
    assert _builder(sensor).build(world, _mission(), Pose2(500.0, 512.0), zero, platform_type="WHEELED").count == 0


def test_long_single_contour_uses_farthest_fill_after_its_representative() -> None:
    observed = np.zeros((256, 256), dtype=bool)
    observed[112:145, 112:145] = True
    roi = np.zeros((256, 256), dtype=bool)
    roi[108:149, 108:149] = True

    batch = _builder(SensorGeometry(80.0, 2.0 * math.pi)).build(
        _world(observed), _mission_for_roi(roi), Pose2(512.0, 512.0), _projection(), platform_type="WHEELED"
    )

    assert batch.count > 1
    positions = batch.features[batch.mask, :2]
    assert np.unique(positions, axis=0).shape[0] == batch.count


def test_each_disconnected_frontier_segment_keeps_a_representative() -> None:
    observed = np.zeros((256, 256), dtype=bool)
    observed[116:141, 108:149] = True
    roi = observed.copy()
    holes = ((124, 116), (124, 140))
    for hole in holes:
        observed[hole] = False

    batch = _builder(SensorGeometry(80.0, 2.0 * math.pi)).build(
        _world(observed), _mission_for_roi(roi), Pose2(512.0, 512.0), _projection(), platform_type="WHEELED"
    )

    columns = batch.features[batch.mask, 0] * 256.0 - 0.5
    assert np.any(columns < 120.0)
    assert np.any(columns > 136.0)


def test_more_than_64_segment_representatives_use_farthest_subset() -> None:
    observed = np.zeros((256, 256), dtype=bool)
    observed[104:153, 104:153] = True
    roi = observed.copy()
    for index in range(28):
        angle = 2.0 * math.pi * index / 28.0
        hole = (round(128 + 18 * math.sin(angle)), round(128 + 18 * math.cos(angle)))
        observed[hole] = False

    batch = _builder(SensorGeometry(80.0, 2.0 * math.pi)).build(
        _world(observed), _mission_for_roi(roi), Pose2(512.0, 512.0), _projection(), platform_type="WHEELED"
    )

    assert batch.count == 64
    grid_columns = batch.features[batch.mask, 0] * 256.0 - 0.5
    grid_rows = batch.features[batch.mask, 1] * 256.0 - 0.5
    assert grid_columns.min() < 114.0 and grid_columns.max() > 142.0
    assert grid_rows.min() < 114.0 and grid_rows.max() > 142.0


def test_candidate_result_does_not_depend_on_argwhere_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    world = _world_with_frontier()
    baseline = _builder().build(world, _mission(), Pose2(500.0, 512.0), _projection(), platform_type="WHEELED")
    original_argwhere = np.argwhere
    monkeypatch.setattr(candidate_builder_module.np, "argwhere", lambda values: original_argwhere(values)[::-1])

    perturbed = _builder().build(world, _mission(), Pose2(500.0, 512.0), _projection(), platform_type="WHEELED")

    np.testing.assert_array_equal(perturbed.features, baseline.features)
    np.testing.assert_array_equal(perturbed.mask, baseline.mask)


def test_candidate_builder_batches_all_feasible_anchors_once_and_is_yaw_invariant() -> None:
    class RecordingEstimator:
        def __init__(self) -> None:
            self.sensor = SensorGeometry(30.0, 2.0 * math.pi)
            self.calls: list[np.ndarray] = []

        def estimate_candidate_gains(
            self,
            observed_mask,
            obstacle_ratio,
            roi_ratio,
            priority_weight,
            candidate_cells,
        ) -> np.ndarray:
            self.calls.append(candidate_cells.copy())
            return np.ones((candidate_cells.shape[0], 2), dtype=np.float32)

    estimator = RecordingEstimator()
    builder = CandidateBuilderV2(estimator)
    first = builder.build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0, yaw_rad=-1.2),
        _projection(),
        platform_type="WHEELED",
    )
    second = builder.build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0, yaw_rad=2.4),
        _projection(),
        platform_type="WHEELED",
    )

    assert len(estimator.calls) == 2
    assert estimator.calls[0].shape[0] > 0
    np.testing.assert_array_equal(estimator.calls[0], estimator.calls[1])
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.mask, second.mask)


def test_sparse_primary_candidates_add_platform_checked_observation_reserves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail when one coarse-reachable goal can end the whole episode."""

    class SparsePrimaryEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            candidates = args[-1]
            self.calls.append(candidates.copy())
            gains = np.ones((len(candidates), 2), dtype=np.float32)
            if len(self.calls) == 1:
                gains.fill(0.0)
                gains[0] = (1.0, 1.0)
            return gains

    reachability_calls: list[np.ndarray] = []

    def platform_filter(
        _self,
        candidate_cells: np.ndarray,
        *,
        target_positions_map: np.ndarray | None = None,
    ) -> CandidateReachabilityResult:
        del target_positions_map
        reachability_calls.append(candidate_cells.copy())
        accepted = np.ascontiguousarray(
            candidate_cells[:, 1] % 2 == 0, dtype=np.bool_
        )
        return CandidateReachabilityResult(
            accepted,
            {
                "platform_unreachable_count": int(
                    (~accepted).sum(dtype=np.int64)
                )
            },
        )

    monkeypatch.setattr(
        PlatformCandidateReachability,
        "filter",
        platform_filter,
    )
    reachability = object.__new__(PlatformCandidateReachability)
    estimator = SparsePrimaryEstimator()

    batch = CandidateBuilderV2(estimator).build(
        _world_with_frontier(),
        _mission(),
        Pose2(500.0, 512.0),
        _projection(),
        platform_type="WHEELED",
        platform_reachability=reachability,
    )

    assert len(estimator.calls) == 2
    assert len(reachability_calls) == 2
    assert estimator.calls[0].shape[0] > 0
    assert estimator.calls[1].shape[0] > estimator.calls[0].shape[0]
    assert set(map(tuple, estimator.calls[0])) < set(
        map(tuple, estimator.calls[1])
    )
    assert batch.count > 1
    assert batch.diagnostics.selected_policy_candidate_count == batch.count


def test_ground_platform_keeps_candidate_when_observed_detour_exists() -> None:
    world, mission, projection, pose, target = _detour_fixture()
    estimator = _RecordingEstimator()
    builder = CandidateBuilderV2(estimator)

    batch = builder.build(
        world,
        mission,
        pose,
        projection,
        platform_type="WHEELED",
    )
    legacy = builder.build(
        world,
        mission,
        pose,
        projection,
        platform_type="WHEELED",
        platform_reachability_filter_enabled=False,
    )

    assert target in map(tuple, estimator.calls[0])
    assert len(estimator.calls) == 2
    assert set(map(tuple, estimator.calls[0])) < set(
        map(tuple, estimator.calls[1])
    )
    assert legacy.count == 0
    assert batch.count > legacy.count
    assert batch.diagnostics.selected_policy_candidate_count == batch.count
    assert batch.diagnostics.physical_candidate_universe_count >= batch.count
    assert batch.diagnostics.physical_unreachable_count > 0


def test_hopper_keeps_observed_landing_when_ground_ray_is_blocked() -> None:
    world, mission, projection, pose, target = _detour_fixture()
    estimator = _RecordingEstimator()
    builder = CandidateBuilderV2(estimator)

    batch = builder.build(
        world,
        mission,
        pose,
        projection,
        platform_type="HOPPER",
    )
    legacy = builder.build(
        world,
        mission,
        pose,
        projection,
        platform_type="HOPPER",
        platform_reachability_filter_enabled=False,
    )

    assert target in map(tuple, estimator.calls[0])
    assert len(estimator.calls) == 2
    assert set(map(tuple, estimator.calls[0])) < set(
        map(tuple, estimator.calls[1])
    )
    assert legacy.count == 0
    assert batch.count > legacy.count


def test_hopper_rejects_unobserved_or_projection_infeasible_landings() -> None:
    world, mission, projection, pose, target = _detour_fixture()
    zero_projection = PlatformProjection(
        projection.canvas,
        traversable_ratio=np.zeros((256, 256), dtype=np.float32),
        local_traversable_ratio=projection.local_traversable_ratio,
        clearance_margin_norm=projection.clearance_margin_norm,
        source="test_only/proxy",
    )
    unsafe = CandidateBuilderV2(_RecordingEstimator()).build(
        world,
        mission,
        pose,
        zero_projection,
        platform_type="HOPPER",
    )
    observed = world.observed_mask.copy()
    observed[target] = False
    target_only = np.zeros((256, 256), dtype=np.float32)
    target_only[target] = 1.0
    target_projection = PlatformProjection(
        projection.canvas,
        traversable_ratio=target_only,
        local_traversable_ratio=projection.local_traversable_ratio,
        clearance_margin_norm=projection.clearance_margin_norm,
        source="test_only/proxy",
    )
    unobserved_world = ObservedWorld(
        world.canvas,
        world.elevation_m,
        observed,
        world.physical_obstacle_layer,
        world.local,
    )
    unobserved = CandidateBuilderV2(_RecordingEstimator()).build(
        unobserved_world,
        mission,
        pose,
        target_projection,
        platform_type="HOPPER",
    )

    assert unsafe.count == 0
    assert unsafe.diagnostics.physical_unreachable_count > 0
    assert unobserved.count == 0


def test_hopper_falls_back_to_reachable_positive_gain_observation_pose() -> None:
    canvas = _canvas()
    robot = (128, 128)
    robot_x_m, robot_y_m = canvas.grid_center_world(*robot)
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[127:130, 128:142] = True
    roi = np.zeros((256, 256), dtype=np.bool_)
    roi[127:130, 128:143] = True
    estimator = _RecordingEstimator()

    batch = CandidateBuilderV2(estimator).build(
        _world(observed),
        _mission_for_roi(roi),
        Pose2(robot_x_m, robot_y_m),
        _projection(),
        platform_type="HOPPER",
    )

    assert batch.count > 0
    assert len(estimator.calls) == 1
    frontier_cell = (128, 141)
    assert frontier_cell not in map(tuple, estimator.calls[0])
    candidate_distances_m = np.linalg.norm(
        estimator.calls[0] - np.asarray(robot), axis=1
    ) * canvas.geometry.resolution_m
    assert np.all(candidate_distances_m <= 30.0)
    assert np.any(estimator.calls[0][:, 1] >= 135)


def test_hopper_emits_truthful_zero_gain_transit_when_frontier_is_remote() -> None:
    canvas = _canvas()
    robot = (128, 128)
    robot_x_m, robot_y_m = canvas.grid_center_world(*robot)
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[127:130, 128:142] = True
    roi = np.zeros((256, 256), dtype=np.bool_)
    roi[127:130, 128:143] = True

    class ZeroGainEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            self.calls.append(args[-1].copy())
            return np.zeros((len(args[-1]), 2), dtype=np.float32)

    batch = CandidateBuilderV2(ZeroGainEstimator()).build(
        _world(observed),
        _mission_for_roi(roi),
        Pose2(robot_x_m, robot_y_m),
        _projection(),
        platform_type="HOPPER",
    )

    assert batch.count > 0
    assert np.all(batch.features[batch.mask, 5:7] == 0.0)
    assert batch.diagnostics.zero_gain_count == 0


def test_hopper_emits_zero_gain_transit_when_frontier_anchors_have_no_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world_with_frontier()
    mission = _mission()
    projection = _projection()
    pose = Pose2(500.0, 512.0)

    class ZeroGainEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            self.calls.append(args[-1].copy())
            return np.zeros((len(args[-1]), 2), dtype=np.float32)

    def accept(
        _self,
        candidate_cells: np.ndarray,
        *,
        target_positions_map: np.ndarray | None = None,
    ) -> CandidateReachabilityResult:
        del target_positions_map
        return CandidateReachabilityResult(
            np.ones(len(candidate_cells), dtype=np.bool_),
            {"platform_unreachable_count": 0},
        )

    monkeypatch.setattr(PlatformCandidateReachability, "filter", accept)
    reachability = object.__new__(PlatformCandidateReachability)
    batch = CandidateBuilderV2(ZeroGainEstimator()).build(
        world,
        mission,
        pose,
        projection,
        platform_type="HOPPER",
        platform_reachability=reachability,
    )

    assert batch.count > 0
    assert np.all(batch.features[batch.mask, 5:7] == 0.0)


def test_zero_gain_backtrack_preserves_exact_parent_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canvas = _canvas()
    robot = (128, 128)
    robot_x_m, robot_y_m = canvas.grid_center_world(*robot)
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[127:130, 128:142] = True
    roi = np.zeros((256, 256), dtype=np.bool_)
    roi[127:130, 128:143] = True
    visited = set(map(tuple, np.column_stack(np.nonzero(observed))))

    class ZeroGainEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            self.calls.append(args[-1].copy())
            return np.zeros((len(args[-1]), 2), dtype=np.float32)

    estimator = ZeroGainEstimator()
    parent_cell = (128, 129)
    parent_center_x, parent_center_y = canvas.grid_center_world(*parent_cell)
    parent = Pose2(
        parent_center_x + 0.02,
        parent_center_y - 0.02,
        elevation_m=123.0,
    )
    recorded_positions: list[np.ndarray | None] = []

    def accept(
        _self,
        candidate_cells: np.ndarray,
        target_positions_map: np.ndarray | None = None,
    ) -> CandidateReachabilityResult:
        recorded_positions.append(
            None if target_positions_map is None else target_positions_map.copy()
        )
        return CandidateReachabilityResult(
            np.ones(len(candidate_cells), dtype=np.bool_),
            {"platform_unreachable_count": 0},
        )

    monkeypatch.setattr(PlatformCandidateReachability, "filter", accept)
    reachability = object.__new__(PlatformCandidateReachability)
    batch = CandidateBuilderV2(estimator).build(
        _world(observed),
        _mission_for_roi(roi),
        Pose2(robot_x_m, robot_y_m),
        _projection(),
        platform_type="HOPPER",
        platform_reachability=reachability,
        excluded_cells=visited,
        backtrack_pose=parent,
    )

    assert batch.count == 1
    assert recorded_positions[-1] is not None
    np.testing.assert_array_equal(
        recorded_positions[-1],
        np.asarray([[parent.x_m, parent.y_m, parent.elevation_m]]),
    )
    np.testing.assert_allclose(
        batch.features[0, :2],
        np.asarray(
            [
                (parent.x_m - canvas.bounds_m[0]) / canvas.geometry.size_m,
                (canvas.bounds_m[3] - parent.y_m) / canvas.geometry.size_m,
            ]
        ),
    )
    assert batch.target_elevation_m[0] == 123.0
    assert np.all(batch.features[0, 5:7] == 0.0)


def test_platform_filter_rejects_unknown_platform_and_is_byte_deterministic() -> None:
    world, mission, projection, pose, _ = _detour_fixture()
    builder = CandidateBuilderV2(_RecordingEstimator())

    with pytest.raises(ValueError, match="platform_type"):
        builder.build(world, mission, pose, projection, platform_type="FLYING")

    first = builder.build(world, mission, pose, projection, platform_type="LEGGED")
    second = builder.build(world, mission, pose, projection, platform_type="LEGGED")
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.diagnostics == second.diagnostics


def test_stage_diagnostics_isolate_platform_unreachable_and_zero_gain() -> None:
    world = _world_with_frontier()
    mission = _mission()
    projection = _projection()
    pose = Pose2(500.0, 512.0)

    class RejectingBridge:
        def project_reachability(self, request, maximum_edge_distance_m):
            del request, maximum_edge_distance_m
            return type(
                "Projection",
                (),
                {
                    "platform_type": "WHEELED",
                    "reachable": np.zeros((256, 256), dtype=np.uint8),
                    "algorithm_id": "test/rejecting-ground/v1",
                },
            )()

    reachability = PlatformCandidateReachability(
        platform_type="WHEELED",
        canvas=world.canvas,
        pose_map=pose,
        observed_elevation_m=world.elevation_m,
        bridge=RejectingBridge(),
        request=SimpleNamespace(
            world=SimpleNamespace(
                local_map=SimpleNamespace(
                    frame_id="odom",
                    width=256,
                    height=256,
                    resolution_m=4.0,
                    origin_m=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                ),
                map_from_odom=SimpleNamespace(
                    parent_frame="map",
                    child_frame="odom",
                    translation_m=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            )
            ),
            local_traversability_projection=SimpleNamespace(
                hard_feasible=np.zeros((256, 256), dtype=np.uint8),
                connected_component=np.full((256, 256), -1, dtype=np.int32),
            ),
        )
    unreachable = CandidateBuilderV2(_RecordingEstimator()).build(
        world,
        mission,
        pose,
        projection,
        platform_type="WHEELED",
        platform_reachability=reachability,
    )

    class ZeroGainEstimator(_RecordingEstimator):
        def estimate_candidate_gains(self, *args) -> np.ndarray:
            candidates = args[-1]
            return np.zeros((len(candidates), 2), dtype=np.float32)

    zero_gain = CandidateBuilderV2(ZeroGainEstimator()).build(
        world,
        mission,
        pose,
        projection,
        platform_type="HOPPER",
        platform_reachability_filter_enabled=False,
    )

    assert unreachable.count == 0
    assert unreachable.diagnostics.physical_unreachable_count > 0
    assert unreachable.diagnostics.zero_gain_count == 0
    assert zero_gain.count == 0
    assert zero_gain.diagnostics.physical_unreachable_count == 0
    assert zero_gain.diagnostics.zero_gain_count > 0


def test_ground_reachability_uses_local_detail_authority_inside_window_and_global_outside() -> None:
    canvas = _canvas()
    pose = Pose2(2.0, 2.0)
    local_shape = (50, 50)
    local_hard = np.ones(local_shape, dtype=np.uint8)
    local_components = np.ones(local_shape, dtype=np.int32)
    local_components[10, 30] = 2

    class RecordingBridge:
        def project_reachability(self, request, maximum_edge_distance_m):
            del request, maximum_edge_distance_m
            reachable = np.zeros((256, 256), dtype=np.uint8)
            reachable[0, 3] = 1
            return SimpleNamespace(
                platform_type="LEGGED",
                reachable=reachable,
                algorithm_id="test/ground-local-override/v1",
            )

        def project_traversability(self, request):
            del request
            return SimpleNamespace(
                hard_feasible=local_hard,
                connected_component=local_components,
            )

    request = SimpleNamespace(
        world=SimpleNamespace(
            local_map=SimpleNamespace(
                frame_id="odom",
                width=local_shape[1],
                height=local_shape[0],
                resolution_m=0.2,
                origin_m=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            ),
            map_from_odom=SimpleNamespace(
                parent_frame="map",
                child_frame="odom",
                translation_m=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )
    )
    reachability = PlatformCandidateReachability(
        platform_type="LEGGED",
        canvas=canvas,
        pose_map=pose,
        observed_elevation_m=np.zeros((256, 256), dtype=np.float32),
        bridge=RecordingBridge(),
        request=request,
    )
    candidates = np.asarray(((255, 0), (255, 1), (255, 3)), dtype=np.int32)
    exact_targets = np.asarray(
        ((3.9, 2.0, 0.0), (6.0, 2.0, 0.0), (12.0, 2.0, 0.0)),
        dtype=np.float64,
    )

    result = reachability.filter(
        candidates, target_positions_map=exact_targets
    )

    # The observed detail component is authoritative inside its window even
    # when the partially observed 4 m start cell makes the global component
    # empty. Targets outside the detail window retain the global result.
    assert result.accepted_mask.tolist() == [True, False, True]
    assert dict(result.reason_counts) == {"platform_unreachable_count": 1}


def test_ground_project_physical_returns_row_major_exact_positions_and_local_override() -> None:
    canvas = _canvas()
    start = (128, 128)
    start_x_m, start_y_m = canvas.grid_center_world(*start)
    resolution_m = canvas.geometry.resolution_m
    local_hard = np.zeros((256, 256), dtype=np.uint8)
    local_components = np.full((256, 256), -1, dtype=np.int32)
    local_row_by_grid_row = {128: 127, 129: 126}
    for grid_row, column, component in (
        (128, 128, 7),
        (128, 129, 7),
        (128, 130, 8),
        (129, 128, 7),
    ):
        local_row = local_row_by_grid_row[grid_row]
        local_hard[local_row, column] = 1
        local_components[local_row, column] = component

    class RecordingBridge:
        def __init__(self) -> None:
            self.reachability_calls = 0

        def project_reachability(self, request, maximum_edge_distance_m):
            del request
            self.reachability_calls += 1
            assert maximum_edge_distance_m == 30.0
            return SimpleNamespace(
                platform_type="WHEELED",
                reachable=np.ones((256, 256), dtype=np.uint8),
                algorithm_id="test/ground-connected-component/v1",
            )

    bridge = RecordingBridge()
    request = SimpleNamespace(
        world=SimpleNamespace(
            local_map=SimpleNamespace(
                frame_id="odom",
                width=256,
                height=256,
                resolution_m=resolution_m,
                origin_m=SimpleNamespace(
                    x=canvas.bounds_m[0],
                    y=canvas.bounds_m[1],
                    z=0.0,
                ),
            ),
            map_from_odom=SimpleNamespace(
                parent_frame="map",
                child_frame="odom",
                translation_m=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )
    )
    reachability = PlatformCandidateReachability(
        platform_type="WHEELED",
        canvas=canvas,
        pose_map=Pose2(start_x_m, start_y_m, elevation_m=4.0),
        observed_elevation_m=np.zeros((256, 256), dtype=np.float32),
        bridge=bridge,
        request=request,
        local_traversability_projection=SimpleNamespace(
            hard_feasible=local_hard,
            connected_component=local_components,
        ),
    )
    cells = np.asarray(
        ((129, 128), (128, 130), (128, 129)), dtype=np.int32
    )
    positions = np.asarray(
        [
            (*canvas.grid_center_world(129, 128), 13.0),
            (*canvas.grid_center_world(128, 130), 12.0),
            (*canvas.grid_center_world(128, 129), 11.0),
        ],
        dtype=np.float64,
    )

    result = reachability.project_physical(
        cells, target_positions_map=np.ascontiguousarray(positions)
    )

    expected_mask = np.zeros((256, 256), dtype=np.bool_)
    expected_mask[128, 129] = True
    expected_mask[129, 128] = True
    np.testing.assert_array_equal(
        result.physical_observation_pose_mask, expected_mask
    )
    np.testing.assert_array_equal(
        result.observation_positions_m,
        np.asarray(
            [
                (*canvas.grid_center_world(128, 129), 11.0),
                (*canvas.grid_center_world(129, 128), 13.0),
            ],
            dtype=np.float64,
        ),
    )
    assert result.platform_type == "WHEELED"
    assert result.physical_projection_schema == PHYSICAL_PROJECTION_SCHEMA
    assert (
        result.physical_reachability_algorithm_id
        == "test/ground-connected-component/v1"
    )
    assert (
        result.physical_evidence_algorithm_id
        == "cpp-safe-traversability-projection/v1"
    )
    assert result.physical_safe_pose_count == 3
    assert result.physically_reachable_pose_count == 2
    assert bridge.reachability_calls == 1


def test_hopper_project_physical_binds_certified_aim_and_direct_connectivity() -> None:
    canvas = _canvas()
    start = (128, 128)
    start_x_m, start_y_m = canvas.grid_center_world(*start)
    target_rejected = (128, 130)
    target_accepted = (128, 129)
    accepted_x_m, accepted_y_m = canvas.grid_center_world(*target_accepted)
    certified_aim = np.asarray(
        (accepted_x_m + 0.02, accepted_y_m - 0.02, 321.0),
        dtype=np.float64,
    )

    class RecordingBridge:
        def __init__(self) -> None:
            self.targets: np.ndarray | None = None
            self.evidence = None

        def project_hopper_landing_evidence(self, request, targets):
            del request
            self.targets = targets.copy()
            aims = targets.copy()
            certified = np.zeros(len(targets), dtype=np.bool_)
            for index, target in enumerate(targets):
                cell = canvas.world_to_grid(
                    float(target[0]), float(target[1])
                )
                if cell == start:
                    certified[index] = True
                elif cell == target_accepted:
                    certified[index] = True
                    aims[index] = certified_aim
            return SimpleNamespace(
                certified=np.ascontiguousarray(certified),
                aim_positions_m=np.ascontiguousarray(aims),
                boundary_m=np.zeros((3, 4, 3), dtype=np.float64),
                area_m2=np.ones(3, dtype=np.float64),
                algorithm_id="test/hopper-landing-evidence/v1",
            )

        def project_hopper_opportunity_context(
            self, request, maximum_edge_distance_m, evidence
        ):
            del request
            assert maximum_edge_distance_m == 30.0
            self.evidence = evidence
            direct = np.zeros((256, 256), dtype=np.bool_)
            direct[127, 129:131] = True
            return SimpleNamespace(
                direct=np.ascontiguousarray(direct),
                algorithm_id="test/hopper-opportunity-connectivity/v1",
            )

        def query_hopper_opportunity_distance(self, context, positive):
            del context, positive
            raise AssertionError("opportunity query not expected")

    bridge = RecordingBridge()
    reachability = PlatformCandidateReachability(
        platform_type="HOPPER",
        canvas=canvas,
        pose_map=Pose2(start_x_m, start_y_m, elevation_m=123.0),
        observed_elevation_m=np.zeros((256, 256), dtype=np.float32),
        bridge=bridge,
        request=object(),
    )
    cells = np.asarray(
        (target_rejected, target_accepted), dtype=np.int32
    )
    targets = np.asarray(
        [
            (*canvas.grid_center_world(*target_rejected), 200.0),
            (*canvas.grid_center_world(*target_accepted), 201.0),
        ],
        dtype=np.float64,
    )

    result = reachability.project_physical(
        cells, target_positions_map=np.ascontiguousarray(targets)
    )

    expected_mask = np.zeros((256, 256), dtype=np.bool_)
    expected_mask[target_accepted] = True
    np.testing.assert_array_equal(
        result.physical_observation_pose_mask, expected_mask
    )
    np.testing.assert_array_equal(
        result.observation_positions_m, certified_aim.reshape(1, 3)
    )
    assert result.physical_safe_pose_count == 1
    assert result.physically_reachable_pose_count == 1
    assert (
        result.physical_evidence_algorithm_id
        == "test/hopper-landing-evidence/v1"
    )
    assert (
        result.physical_reachability_algorithm_id
        == "test/hopper-opportunity-connectivity/v1"
    )
    assert bridge.targets is not None
    np.testing.assert_array_equal(
        bridge.targets[0], np.asarray((start_x_m, start_y_m, 123.0))
    )
    assert bridge.evidence.algorithm_id == "test/hopper-landing-evidence/v1"


def test_ground_project_physical_accepts_empty_observed_safe_set() -> None:
    canvas = _canvas()
    start_x_m, start_y_m = canvas.grid_center_world(128, 128)

    class EmptyBridge:
        def project_reachability(self, request, maximum_edge_distance_m):
            del request, maximum_edge_distance_m
            return SimpleNamespace(
                platform_type="LEGGED",
                reachable=np.zeros((256, 256), dtype=np.uint8),
                algorithm_id="test/empty-ground/v1",
            )

    reachability = PlatformCandidateReachability(
        platform_type="LEGGED",
        canvas=canvas,
        pose_map=Pose2(start_x_m, start_y_m),
        observed_elevation_m=np.zeros((256, 256), dtype=np.float32),
        bridge=EmptyBridge(),
        request=SimpleNamespace(
            world=SimpleNamespace(
                local_map=SimpleNamespace(
                    frame_id="odom",
                    width=256,
                    height=256,
                    resolution_m=canvas.geometry.resolution_m,
                    origin_m=SimpleNamespace(
                        x=canvas.bounds_m[0],
                        y=canvas.bounds_m[1],
                        z=0.0,
                    ),
                ),
                map_from_odom=SimpleNamespace(
                    parent_frame="map",
                    child_frame="odom",
                    translation_m=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    rotation=SimpleNamespace(
                        x=0.0, y=0.0, z=0.0, w=1.0
                    ),
                ),
            )
        ),
        local_traversability_projection=SimpleNamespace(
            hard_feasible=np.zeros((256, 256), dtype=np.uint8),
            connected_component=np.full(
                (256, 256), -1, dtype=np.int32
            ),
        ),
    )

    result = reachability.project_physical(
        np.empty((0, 2), dtype=np.int32)
    )

    assert result.physical_observation_pose_mask.shape == (256, 256)
    assert not result.physical_observation_pose_mask.any()
    assert result.observation_positions_m.shape == (0, 3)
    assert result.physical_safe_pose_count == 0
    assert result.physically_reachable_pose_count == 0
    assert result.physical_reachability_algorithm_id == "test/empty-ground/v1"


def test_project_physical_rejects_position_outside_declared_cell() -> None:
    canvas = _canvas()
    start_x_m, start_y_m = canvas.grid_center_world(128, 128)
    reachability = PlatformCandidateReachability(
        platform_type="WHEELED",
        canvas=canvas,
        pose_map=Pose2(start_x_m, start_y_m),
        observed_elevation_m=np.zeros((256, 256), dtype=np.float32),
        bridge=SimpleNamespace(
            project_reachability=lambda *_: pytest.fail(
                "misaligned input reached the bridge"
            )
        ),
        request=object(),
        local_traversability_projection=object(),
    )
    wrong_x_m, wrong_y_m = canvas.grid_center_world(128, 130)

    with pytest.raises(ValueError, match="leaves its cell"):
        reachability.project_physical(
            np.asarray(((128, 129),), dtype=np.int32),
            target_positions_map=np.asarray(
                ((wrong_x_m, wrong_y_m, 0.0),), dtype=np.float64
            ),
        )


@pytest.mark.parametrize(
    ("reachable", "algorithm_id", "message"),
    (
        (np.zeros((2, 2), dtype=np.uint8), "test/ground/v1", "geometry"),
        (np.zeros((256, 256), dtype=np.uint8), "", "algorithm"),
    ),
)
def test_project_physical_rejects_malformed_reachability_projection(
    reachable: np.ndarray,
    algorithm_id: str,
    message: str,
) -> None:
    canvas = _canvas()
    start_x_m, start_y_m = canvas.grid_center_world(128, 128)
    bridge = SimpleNamespace(
        project_reachability=lambda *_: SimpleNamespace(
            platform_type="WHEELED",
            reachable=reachable,
            algorithm_id=algorithm_id,
        )
    )
    reachability = PlatformCandidateReachability(
        platform_type="WHEELED",
        canvas=canvas,
        pose_map=Pose2(start_x_m, start_y_m),
        observed_elevation_m=np.zeros((256, 256), dtype=np.float32),
        bridge=bridge,
        request=object(),
        local_traversability_projection=SimpleNamespace(
            hard_feasible=np.ones((256, 256), dtype=np.uint8),
            connected_component=np.zeros((256, 256), dtype=np.int32),
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        reachability.project_physical(
            np.asarray(((128, 129),), dtype=np.int32)
        )


def test_hopper_project_physical_rejects_certified_aim_outside_cell() -> None:
    canvas = _canvas()
    start = (128, 128)
    target = (128, 129)
    start_x_m, start_y_m = canvas.grid_center_world(*start)
    wrong_x_m, wrong_y_m = canvas.grid_center_world(128, 130)

    class MisalignedLandingBridge:
        def project_hopper_landing_evidence(self, request, targets):
            del request
            aims = targets.copy()
            aims[1] = (wrong_x_m, wrong_y_m, 10.0)
            return SimpleNamespace(
                certified=np.ones(2, dtype=np.bool_),
                aim_positions_m=np.ascontiguousarray(aims),
                boundary_m=np.zeros((2, 4, 3), dtype=np.float64),
                area_m2=np.ones(2, dtype=np.float64),
                algorithm_id="test/misaligned-landing/v1",
            )

        def project_hopper_opportunity_context(self, *args):
            del args
            return SimpleNamespace(
                direct=np.ones((256, 256), dtype=np.bool_),
                algorithm_id="test/opportunity-connectivity/v1",
            )

        def query_hopper_opportunity_distance(self, context, positive):
            del context, positive
            raise AssertionError("opportunity query not expected")

    reachability = PlatformCandidateReachability(
        platform_type="HOPPER",
        canvas=canvas,
        pose_map=Pose2(start_x_m, start_y_m),
        observed_elevation_m=np.zeros((256, 256), dtype=np.float32),
        bridge=MisalignedLandingBridge(),
        request=object(),
    )

    with pytest.raises(RuntimeError, match="leaves its cell"):
        reachability.project_physical(
            np.asarray((target,), dtype=np.int32)
        )


def test_hopper_reachability_uses_certified_pose_height_for_start() -> None:
    canvas = _canvas()
    start = (128, 128)
    target = (128, 129)
    start_x_m, start_y_m = canvas.grid_center_world(*start)
    elevation = np.full((256, 256), 7.0, dtype=np.float32)

    class RecordingBridge:
        def __init__(self) -> None:
            self.targets = None

        def project_hopper_landing_evidence(self, request, targets):
            del request
            self.targets = targets.copy()
            count = len(targets)
            return SimpleNamespace(
                certified=np.ones(count, dtype=np.bool_),
                aim_positions_m=targets.copy(),
                boundary_m=np.zeros((count, 4, 3), dtype=np.float64),
                area_m2=np.ones(count, dtype=np.float64),
                algorithm_id="test/landing-evidence/v1",
            )

        def project_hopper_opportunity_context(
            self, request, maximum_edge_distance_m, evidence
        ):
            del request, maximum_edge_distance_m, evidence
            return SimpleNamespace(
                direct=np.ones((256, 256), dtype=np.bool_),
                algorithm_id="test/opportunity-connectivity/v1",
            )

        def query_hopper_opportunity_distance(self, context, positive):
            del context, positive
            raise AssertionError("opportunity query not expected")

    bridge = RecordingBridge()
    reachability = PlatformCandidateReachability(
        platform_type="HOPPER",
        canvas=canvas,
        pose_map=Pose2(start_x_m, start_y_m, elevation_m=123.0),
        observed_elevation_m=elevation,
        bridge=bridge,
        request=object(),
    )

    target_x_m, target_y_m = canvas.grid_center_world(*target)
    exact_target = np.asarray(
        [[target_x_m + 0.02, target_y_m - 0.02, 321.0]],
        dtype=np.float64,
    )
    result = reachability.filter(
        np.asarray([target], dtype=np.int32),
        target_positions_map=exact_target,
    )

    assert result.accepted_mask.tolist() == [True]
    np.testing.assert_array_equal(
        bridge.targets[0], np.asarray([start_x_m, start_y_m, 123.0])
    )
    np.testing.assert_array_equal(bridge.targets[1], exact_target[0])

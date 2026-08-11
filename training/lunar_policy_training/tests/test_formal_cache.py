from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import pathlib
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import lunar_planner_training_bridge as bridge_api

import lunar_policy_training.polar_data.formal_cache as formal_cache_module

from lunar_policy_training.capability_freeze import (
    FrozenHopperCapability,
    FrozenInterval,
    FrozenLeggedBodyPrimitive,
    FrozenLeggedCapability,
    FrozenObservationCapability,
    FrozenPlatformCapability,
    FrozenPose3,
    FrozenQuaternion,
    FrozenVec2,
    FrozenVec3,
    FrozenWheelMotionPrimitive,
    FrozenWheeledCapability,
)
from lunar_policy_training.environment.coverability import (
    build_coverable_detail_mask,
    IneligibleReason,
    PlatformCoverability,
    mask_sha256,
    pack_detail_mask,
    physical_projection_sha256 as canonical_physical_projection_sha256,
)
from lunar_policy_training.environment.macro_step import PolicyAction
from lunar_policy_training.polar_data.formal_cache import (
    FORMAL_CACHE_SCHEMA,
    FormalCacheError,
    FormalCacheIdentity,
    StaticSceneData,
    _finite_hopper_landing_targets,
    _formal_platform_eligibility_ready,
    _ordered_bounded_process_map,
    _parallel_platform_map,
    load_formal_cache,
    platform_scenario_schedule_id,
    write_formal_cache,
)
from lunar_policy_training.proxy_scenario import _ProxyEpisode
from lunar_policy_training.polar_data.raster import MapCanvas


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _independent_array_metadata(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(values)
    byte_order = array.dtype.byteorder
    if byte_order == "=":
        byte_order = "little" if np.little_endian else "big"
    elif byte_order == "|":
        byte_order = "not-applicable"
    else:
        byte_order = "little" if byte_order == "<" else "big"
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "byte_order": byte_order,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _rewrite_scene_array_and_resign_manifest(
    root: pathlib.Path,
    manifest: dict[str, object],
    array_name: str,
    replacement: np.ndarray,
) -> None:
    scene = manifest["scenes"][0]
    scene_path = root / scene["relative_path"]
    with np.load(scene_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays[array_name] = np.ascontiguousarray(replacement)
    np.savez_compressed(scene_path, **arrays)
    _refresh_scene_integrity_and_resign(root, manifest)


def _resign_manifest(root: pathlib.Path, manifest: dict[str, object]) -> None:
    body = dict(manifest)
    body.pop("cache_manifest_sha256")
    manifest["cache_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    (root / "cache-manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _refresh_scene_integrity_and_resign(
    root: pathlib.Path, manifest: dict[str, object]
) -> None:
    scene = manifest["scenes"][0]
    scene_path = root / scene["relative_path"]
    with np.load(scene_path, allow_pickle=False) as archive:
        scene["arrays"] = {
            name: _independent_array_metadata(archive[name])
            for name in archive.files
        }
    scene["size_bytes"] = scene_path.stat().st_size
    scene["sha256"] = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    inventory = next(
        item
        for item in manifest["inventory"]
        if item["relative_path"] == scene["relative_path"]
    )
    inventory["size_bytes"] = scene["size_bytes"]
    inventory["sha256"] = scene["sha256"]
    _resign_manifest(root, manifest)


def _scenario_document(scene_id: str) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "test-formal-scenario-manifest/v1",
        "scenarios": [{"scene_id": scene_id, "split": "train"}],
    }
    body["scenario_manifest_sha256"] = _sha("scenario-document")
    return body


def _identity(scenario_sha256: str) -> FormalCacheIdentity:
    return FormalCacheIdentity(
        source_lock_file_sha256=_sha("source-lock-file"),
        source_sha256s={
            "NASA_LOLA_87S_DEM": _sha("dem"),
            "NASA_LOLA_87S_COUNT": _sha("count"),
            "JAXA_LUPEX_DATA_S1": _sha("jaxa"),
        },
        split_manifest_file_sha256=_sha("split-file"),
        split_sha256=_sha("split"),
        scenario_manifest_sha256=scenario_sha256,
        generator_sha256=_sha("generator"),
        capability_sha256=_sha("capability"),
        reward_sha256=_sha("reward"),
        training_semantics_sha256=_sha("semantics"),
        v3_source_commit="a" * 40,
        v3_sha256=_sha("v3"),
    )


def _scene(
    scene_id: str,
    *,
    qualified_start_cells: dict[str, tuple[int, int] | None] | None = None,
    ineligible_platforms: frozenset[str] = frozenset(),
) -> StaticSceneData:
    shape = (256, 256)
    hard = {
        platform: np.ones(shape, dtype=np.uint8)
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    }
    clearance = {
        platform: np.full(shape, 0.5, dtype=np.float32)
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    }
    starts = (
        {
            "WHEELED": (127, 127),
            "LEGGED": (127, 127),
            "HOPPER": (127, 127),
        }
        if qualified_start_cells is None
        else qualified_start_cells
    )
    bounds = (0.0, 0.0, 1024.0, 1024.0)
    resolution_m = 4.0
    coverability: dict[str, PlatformCoverability] = {}
    observation_positions: dict[str, np.ndarray] = {}
    evidence_algorithms: dict[str, str] = {}
    for platform in ("WHEELED", "LEGGED", "HOPPER"):
        reachable = np.ones(shape, dtype=np.bool_)
        start = starts[platform]
        if start is None:
            reachable[:] = False
        detail = np.ones(shape, dtype=np.bool_)
        reason = None
        eligible = True
        if start is None:
            detail[:] = False
            reason = IneligibleReason.UNSAFE_START
            eligible = False
        elif platform in ineligible_platforms:
            detail.reshape(-1)[int(detail.size * 0.94) :] = False
            reason = IneligibleReason.MISSION_COVERABLE_BELOW_95
            eligible = False
        count = int(detail.sum(dtype=np.int64))
        algorithm_id = f"test-physical-reachability/{platform.lower()}"
        evidence_algorithm_id = f"test-physical-evidence/{platform.lower()}"
        capability_sha256 = _sha(f"capability-content/{platform}")
        rows, columns = np.nonzero(reachable)
        positions = np.ascontiguousarray(
            np.column_stack(
                (
                    (columns.astype(np.float64) + 0.5) * resolution_m,
                    bounds[3]
                    - (rows.astype(np.float64) + 0.5) * resolution_m,
                    np.zeros(len(rows), dtype=np.float64),
                )
            ),
            dtype=np.float64,
        ).reshape((-1, 3))
        start_position = (
            None
            if start is None
            else (
                (float(start[1]) + 0.5) * resolution_m,
                bounds[3] - (float(start[0]) + 0.5) * resolution_m,
                0.0,
            )
        )
        start_sha256 = formal_cache_module._physical_start_identity_from_position(
            platform_type=platform,
            start_cell=start,
            position_m=start_position,
        )
        observation_positions[platform] = positions
        evidence_algorithms[platform] = evidence_algorithm_id
        coverability[platform] = PlatformCoverability(
            platform_type=platform,
            qualified_start_cell=start,
            physical_observation_pose_mask=reachable,
            physical_projection_schema=(
                "lunar-physical-coverability-projection/v1"
            ),
            physical_reachability_algorithm_id=algorithm_id,
            physical_safe_pose_count=(0 if start is None else reachable.size),
            physically_reachable_pose_count=int(
                reachable.sum(dtype=np.int64)
            ),
            physical_projection_sha256=canonical_physical_projection_sha256(
                platform_type=platform,
                physical_reachability_algorithm_id=algorithm_id,
                physical_evidence_algorithm_id=evidence_algorithm_id,
                physical_observation_pose_mask=reachable,
                physical_observation_positions_m=positions,
                physical_grid_resolution_m=resolution_m,
                physical_grid_origin_m=(bounds[0], bounds[3]),
                physical_grid_world_bounds_m=bounds,
                physical_grid_axis_convention=(
                    "north-up-row-major-row-decreases-y-column-increases-x"
                ),
                capability_content_sha256=capability_sha256,
                start_identity_sha256=start_sha256,
            ),
            mission_target_detail_mask_sha256=mask_sha256(
                np.ones(shape, dtype=np.bool_)
            ),
            coverable_detail_shape=detail.shape,
            coverable_detail_bits=pack_detail_mask(detail),
            coverable_ratio=detail.astype(np.float32),
            mission_target_detail_cell_count=detail.size,
            coverable_detail_cell_count=count,
            mission_coverable_fraction=count / detail.size,
            initial_coverable_fraction=0.1,
            initial_candidate_count=1,
            coverable_detail_mask_sha256=mask_sha256(detail),
            sensor_visibility_algorithm_id="two-dimensional-detail-los/v1",
            capability_content_sha256=capability_sha256,
            start_identity_sha256=start_sha256,
            exact=True,
            eligible=eligible,
            ineligible_reason=reason,
        )
    return StaticSceneData(
        scene_id=scene_id,
        source="NASA_LOLA",
        split="train",
        window_id="nasa-window-000",
        window_sha256="b" * 64,
        world_bounds_m=bounds,
        elevation_m=np.zeros(shape, np.float32),
        valid_mask=np.ones(shape, bool),
        physical_obstacle_ratio=np.zeros(shape, np.float32),
        physical_obstacle_height_m=np.zeros(shape, np.float32),
        forbidden_ratio=np.zeros(shape, np.float32),
        rocks=np.empty((0, 4), np.float64),
        craters=np.empty((0, 4), np.float64),
        no_go_vertices=np.empty((0, 6, 2), np.float64),
        hard_feasible=hard,
        clearance_margin_norm=clearance,
        coverability=coverability,
        physical_evidence_algorithm_ids=evidence_algorithms,
        hopper_physical_observation_positions_um=np.ascontiguousarray(
            np.rint(observation_positions["HOPPER"] * 1_000_000.0),
            dtype="<i8",
        ),
    )


def _write(tmp_path: pathlib.Path):
    scene_id = _sha("scene")
    scenario = _scenario_document(scene_id)
    identity = _identity(str(scenario["scenario_manifest_sha256"]))
    root = tmp_path / "formal-cache"
    manifest = write_formal_cache(
        root,
        identity=identity,
        scenario_manifest=scenario,
        materialization="preflight",
        scenes=(_scene(scene_id),),
        repository_root=REPOSITORY_ROOT,
    )
    return root, identity, manifest


def _frozen_pose(x_m: float, y_m: float, yaw_rad: float = 0.0) -> FrozenPose3:
    return FrozenPose3(
        position_m=FrozenVec3(x_m, y_m, 0.0),
        orientation=FrozenQuaternion(
            math.cos(yaw_rad / 2.0),
            0.0,
            0.0,
            math.sin(yaw_rad / 2.0),
        ),
    )


def _frozen_proxy_capability(
    platform_type: str, *, alternate_motion: bool
) -> FrozenPlatformCapability:
    if platform_type == "WHEELED":
        primitives = (
            (
                FrozenWheelMotionPrimitive(
                    "long-forward",
                    "FORWARD",
                    _frozen_pose(0.4, 0.0),
                ),
                FrozenWheelMotionPrimitive(
                    "stop-after-long-forward",
                    "STOP_AND_SWITCH",
                    _frozen_pose(0.0, 0.0),
                ),
            )
            if alternate_motion
            else (
                FrozenWheelMotionPrimitive(
                    "forward", "FORWARD", _frozen_pose(0.2, 0.0)
                ),
                FrozenWheelMotionPrimitive(
                    "reverse", "REVERSE", _frozen_pose(-0.2, 0.0)
                ),
                FrozenWheelMotionPrimitive(
                    "spin-left",
                    "SPIN_COUNTERCLOCKWISE",
                    _frozen_pose(0.0, 0.0, math.pi / 32.0),
                ),
            )
        )
        typed = FrozenWheeledCapability(
            reference_point="base_link",
            footprint_xy_m=(
                FrozenVec2(-0.591, -0.409),
                FrozenVec2(0.591, -0.409),
                FrozenVec2(0.591, 0.409),
                FrozenVec2(-0.591, 0.409),
            ),
            body_extent_m=FrozenVec3(1.182, 0.818, 1.29996),
            wheel_diameter_m=0.319,
            wheel_width_m=0.148,
            wheelbase_m=0.8175,
            track_width_m=0.67,
            minimum_underbody_clearance_m=0.21,
            maximum_local_obstacle_relief_m=0.2,
            allow_unsupported_gap=False,
            minimum_body_z_m=0.0,
            maximum_body_z_m=1.29996,
            maximum_forward_speed_mps=1.5,
            maximum_reverse_speed_mps=1.5,
            maximum_spin_rate_radps=1.0,
            maximum_acceleration_mps2=0.5,
            maximum_braking_deceleration_mps2=0.5,
            maximum_yaw_acceleration_radps2=0.5,
            maximum_lateral_acceleration_mps2=0.5,
            maximum_curvature_per_m=1.0,
            maximum_slope_rad=0.3490658503988659,
            minimum_clearance_m=0.2,
            motion_primitives=primitives,
        )
    elif platform_type == "LEGGED":
        primitives = (
            (
                FrozenLeggedBodyPrimitive(
                    "long-forward", "FORWARD", FrozenVec3(0.4, 0.0, 0.0), 0.0
                ),
                FrozenLeggedBodyPrimitive(
                    "big-spin", "SPIN", FrozenVec3(0.0, 0.0, 0.0), math.pi / 32.0
                ),
            )
            if alternate_motion
            else (
                FrozenLeggedBodyPrimitive(
                    "forward", "FORWARD", FrozenVec3(0.2, 0.0, 0.0), 0.0
                ),
                FrozenLeggedBodyPrimitive(
                    "backward", "BACKWARD", FrozenVec3(-0.2, 0.0, 0.0), 0.0
                ),
                FrozenLeggedBodyPrimitive(
                    "left", "LATERAL_LEFT", FrozenVec3(0.0, 0.2, 0.0), 0.0
                ),
            )
        )
        typed = FrozenLeggedCapability(
            reference_point="base_link",
            body_extent_m=FrozenVec3(0.68, 0.33, 0.35),
            platform_mass_kg=15.89,
            maximum_payload_kg=10.0,
            maximum_slope_rad=0.5235987755982988,
            maximum_step_height_m=0.5,
            maximum_gap_width_m=0.3,
            minimum_body_clearance_m=0.3,
            step_vertical_rate_mps=0.1,
            body_height_m=FrozenInterval(0.28, 0.38),
            forward_speed_mps=FrozenInterval(-1.5, 1.5),
            lateral_speed_mps=FrozenInterval(-0.8, 0.8),
            yaw_rate_radps=FrozenInterval(-1.0, 1.0),
            maximum_linear_acceleration_mps2=1.0,
            maximum_yaw_acceleration_radps2=1.0,
            motion_primitives=primitives,
        )
    else:
        typed = FrozenHopperCapability(
            specific_impulse_s=301.0,
            reference_total_mass_kg=20.0,
            reference_propellant_mass_kg=0.2,
            landing_support_radius_m=0.45,
            flight_collision_radius_m=0.55,
            maximum_landing_plane_residual_m=0.05,
            landing_lateral_margin_m=0.2,
            flight_map_margin_m=0.2,
            reachability_delta_v_margin_ratio=0.1,
            standard_gravity_mps2=9.80665,
            maximum_landing_slope_rad=0.17453292519943295,
        )
    return FrozenPlatformCapability(
        platform_type=platform_type,
        capability_type=f"test-{platform_type.lower()}-physical/v1",
        capability_version=f"{platform_type.lower()}-capability-v2",
        platform_id=f"proxy-{platform_type.lower()}",
        base_frame_id="base_link",
        platform_document_path=f"{platform_type.lower()}/platform.yaml",
        observation_document_path=f"{platform_type.lower()}/observation.json",
        urdf_path=f"{platform_type.lower()}/platform.urdf",
        mesh_paths=(),
        observation_capability=FrozenObservationCapability(
            sensor_range_m=30.0, sensor_fov_rad=2.0 * math.pi
        ),
        typed_capability=typed,
        content_sha256=_sha(
            f"{platform_type}/{'alternate' if alternate_motion else 'baseline'}"
        ),
        resources=(),
    )


def _proxy_request(platform_type: str, frontier_index: int = 0) -> object:
    episode = _ProxyEpisode(0, platform_type, scenario_index=0)
    identity = episode.observation.observation_identities[0]
    return episode.build_request(
        PolicyAction(frontier_index=frontier_index, theta_rad=0.0), identity
    ).request


def _replace_with_substantive_motion_primitives(
    request: object, platform_type: str
) -> None:
    if platform_type == "WHEELED":
        forward = bridge_api.WheelMotionPrimitive()
        forward.primitive_id = "long-forward"
        forward.kind = bridge_api.WheelPrimitiveKind.FORWARD
        pose = bridge_api.Pose3()
        pose.orientation.w = 1.0
        pose.position_m.x = 0.4
        forward.relative_end_pose = pose
        stop = bridge_api.WheelMotionPrimitive()
        stop.primitive_id = "stop-after-long-forward"
        stop.kind = bridge_api.WheelPrimitiveKind.STOP_AND_SWITCH
        request.capability.motion_primitives = [forward, stop]
    else:
        forward = bridge_api.LeggedBodyPrimitive()
        forward.primitive_id = "long-forward"
        forward.kind = bridge_api.LeggedPrimitiveKind.FORWARD
        displacement = bridge_api.Vec3()
        displacement.x = 0.4
        forward.body_frame_displacement_m = displacement
        spin = bridge_api.LeggedBodyPrimitive()
        spin.primitive_id = "big-spin"
        spin.kind = bridge_api.LeggedPrimitiveKind.SPIN
        spin.yaw_change_rad = math.pi / 32.0
        request.capability.motion_primitives = [forward, spin]


def _bridge_motion_signature(request: object, platform_type: str) -> tuple:
    if platform_type == "WHEELED":
        return tuple(
            (
                item.primitive_id,
                float(item.relative_end_pose.position_m.x),
                float(item.relative_end_pose.position_m.y),
                float(item.relative_end_pose.orientation.z),
            )
            for item in request.capability.motion_primitives
        )
    return tuple(
        (
            item.primitive_id,
            float(item.body_frame_displacement_m.x),
            float(item.body_frame_displacement_m.y),
            float(item.yaw_change_rad),
        )
        for item in request.capability.motion_primitives
    )


def _hopper_evidence(request: object, bridge: object) -> tuple[object, np.ndarray, str]:
    grid = request.world.local_map
    targets = np.asarray(
        [
            (
                float(grid.origin_m.x) + (column + 0.5) * grid.resolution_m,
                float(grid.origin_m.y) + (row + 0.5) * grid.resolution_m,
                0.0,
            )
            for row in range(grid.height)
            for column in range(grid.width)
        ],
        dtype=np.float64,
    )
    result = bridge.project_hopper_landing_evidence(request, targets)
    shape = (grid.height, grid.width)
    certified = np.ascontiguousarray(result.certified.reshape(shape))
    aim = np.ascontiguousarray(result.aim_positions_m.reshape((*shape, 3)))
    boundary = np.ascontiguousarray(
        result.boundary_m.reshape((*shape, 4, 3))
    )
    area = np.ascontiguousarray(result.area_m2.reshape(shape))
    evidence = bridge_api.HopperLandingEvidenceGrid(
        certified, aim, boundary, area, str(result.algorithm_id)
    )
    return evidence, np.ascontiguousarray(np.flipud(aim)), str(result.algorithm_id)


def _real_physical_coverability_product(
    request: object,
    *,
    platform_type: str,
    capability_sha256: str,
    bridge: object,
    hopper_evidence: tuple[object, np.ndarray, str] | None = None,
) -> tuple[np.ndarray, str, np.ndarray]:
    grid = request.world.global_map
    if platform_type == "HOPPER":
        assert hopper_evidence is not None
        evidence, north_up_aim, evidence_algorithm_id = hopper_evidence
        projection = bridge.project_reachability(request, 30.0, evidence)
    else:
        projection = bridge.project_reachability(request, 30.0)
        north_up_aim = None
        evidence_algorithm_id = "cpp-safe-traversability-projection/v1"
    physical = np.ascontiguousarray(
        np.flipud(projection.reachable).astype(np.bool_)
    )
    left = float(grid.origin_m.x)
    bottom = float(grid.origin_m.y)
    right = left + grid.width * grid.resolution_m
    top = bottom + grid.height * grid.resolution_m
    if north_up_aim is None:
        positions = np.asarray(
            [
                (
                    left + (int(column) + 0.5) * grid.resolution_m,
                    top - (int(row) + 0.5) * grid.resolution_m,
                    0.0,
                )
                for row, column in np.argwhere(physical)
            ],
            dtype=np.float64,
        ).reshape((-1, 3))
    else:
        positions = np.ascontiguousarray(north_up_aim[physical], dtype=np.float64)
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    state = request.current_state
    pose = state.body_pose if platform_type == "LEGGED" else state.pose
    start_sha256 = hashlib.sha256(
        json.dumps(
            {
                "platform_type": platform_type,
                "position_m": [
                    float(pose.position_m.x),
                    float(pose.position_m.y),
                    float(pose.position_m.z),
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    projection_sha256 = canonical_physical_projection_sha256(
        platform_type=platform_type,
        physical_reachability_algorithm_id=str(projection.algorithm_id),
        physical_evidence_algorithm_id=evidence_algorithm_id,
        physical_observation_pose_mask=physical,
        physical_observation_positions_m=positions,
        physical_grid_resolution_m=float(grid.resolution_m),
        physical_grid_origin_m=(left, top),
        physical_grid_world_bounds_m=(left, bottom, right, top),
        physical_grid_axis_convention=(
            "north-up-row-major-row-decreases-y-column-increases-x"
        ),
        capability_content_sha256=capability_sha256,
        start_identity_sha256=start_sha256,
    )
    pose_cells = np.asarray(
        [
            (
                int(math.floor((top - float(position[1])) / grid.resolution_m)),
                int(math.floor((float(position[0]) - left) / grid.resolution_m)),
            )
            for position in positions
        ],
        dtype=np.int32,
    )
    pose_cells[:, 0] = np.clip(pose_cells[:, 0], 0, grid.height - 1)
    pose_cells[:, 1] = np.clip(pose_cells[:, 1], 0, grid.width - 1)

    def reveal(_truth: np.ndarray, pose_cell: tuple[int, int]) -> np.ndarray:
        visible = np.zeros(physical.shape, dtype=np.bool_)
        row, column = pose_cell
        visible[
            max(row - 1, 0) : min(row + 2, physical.shape[0]),
            max(column - 1, 0) : min(column + 2, physical.shape[1]),
        ] = True
        return visible

    coverable = build_coverable_detail_mask(
        mission_target_detail_mask=np.ones(physical.shape, dtype=np.bool_),
        truth_obstacle_ratio=np.zeros(physical.shape, dtype=np.float32),
        reachable_pose_mask=physical,
        observation_pose_cells=np.ascontiguousarray(pose_cells),
        reveal_from_pose=reveal,
    )
    return physical, projection_sha256, pack_detail_mask(coverable)


def test_platform_map_executes_independent_work_concurrently_in_fixed_order() -> None:
    barrier = threading.Barrier(3, timeout=2.0)

    def worker(platform: str) -> str:
        barrier.wait()
        return platform.lower()

    result = _parallel_platform_map(worker)

    assert list(result) == ["WHEELED", "LEGGED", "HOPPER"]
    assert result == {
        "WHEELED": "wheeled",
        "LEGGED": "legged",
        "HOPPER": "hopper",
    }


def test_bounded_process_map_yields_source_order() -> None:
    result = list(
        _ordered_bounded_process_map((-3, -1, 2), abs, max_workers=2)
    )

    assert result == [3, 1, 2]


def test_hopper_landing_targets_exclude_nodata_cells_in_row_major_order() -> None:
    class Canvas:
        @staticmethod
        def grid_center_world(row: int, column: int) -> tuple[float, float]:
            return float(column) + 0.5, float(row) + 0.5

    class Projected:
        canvas = Canvas()
        valid_mask = np.asarray(
            [[True, False, True], [False, True, True]], dtype=np.bool_
        )
        elevation_m = np.asarray(
            [[1.0, np.nan, 2.0], [np.nan, 3.0, 4.0]], dtype=np.float32
        )

    cells, targets = _finite_hopper_landing_targets(
        Projected(), row0=0, row1=2, column0=0, column1=3
    )

    assert cells == [(0, 0), (0, 2), (1, 1), (1, 2)]
    np.testing.assert_array_equal(
        targets,
        np.asarray(
            [
                [0.5, 0.5, 1.0],
                [2.5, 0.5, 2.0],
                [1.5, 1.5, 3.0],
                [2.5, 1.5, 4.0],
            ],
            dtype=np.float64,
        ),
    )
    assert targets.flags.c_contiguous


def test_physical_capability_identity_excludes_planner_motion_primitives() -> None:
    @dataclass(frozen=True)
    class PhysicalCapability:
        maximum_slope_rad: float
        minimum_clearance_m: float
        motion_primitives: tuple[str, ...]

    @dataclass(frozen=True)
    class ObservationCapability:
        sensor_range_m: float
        sensor_fov_rad: float

    common = {
        "platform_type": "WHEELED",
        "capability_type": "WHEELED",
        "capability_version": "1.0.0",
        "platform_id": "test-rover",
        "base_frame_id": "base_link",
        "observation_capability": ObservationCapability(30.0, 6.0),
    }
    first = SimpleNamespace(
        **common,
        typed_capability=PhysicalCapability(
            maximum_slope_rad=0.4,
            minimum_clearance_m=0.2,
            motion_primitives=("forward", "reverse"),
        ),
    )
    second = SimpleNamespace(
        **common,
        typed_capability=PhysicalCapability(
            maximum_slope_rad=0.4,
            minimum_clearance_m=0.2,
            motion_primitives=("renamed-reverse", "renamed-forward", "spin"),
        ),
    )

    assert formal_cache_module._physical_capability_content_sha256(
        first
    ) == formal_cache_module._physical_capability_content_sha256(second)


@pytest.mark.parametrize("platform_type", ("WHEELED", "LEGGED", "HOPPER"))
def test_real_physical_product_is_invariant_to_substantive_planner_motion(
    platform_type: str,
) -> None:
    first_request = _proxy_request(platform_type, frontier_index=0)
    second_request = _proxy_request(
        platform_type,
        frontier_index=1 if platform_type == "HOPPER" else 0,
    )
    first_capability = _frozen_proxy_capability(
        platform_type, alternate_motion=False
    )
    second_capability = _frozen_proxy_capability(
        platform_type, alternate_motion=True
    )
    if platform_type == "HOPPER":
        bridge = bridge_api.PlannerBridge()
        first_plan = bridge.plan(first_request)
        second_plan = bridge.plan(second_request)
        assert first_plan.reference is not None
        assert second_plan.reference is not None
        first_landing = first_plan.reference.data.segments[-1].nominal_landing_point_m
        second_landing = second_plan.reference.data.segments[-1].nominal_landing_point_m
        assert (float(first_landing.x), float(first_landing.y)) != (
            float(second_landing.x),
            float(second_landing.y),
        )
        evidence = _hopper_evidence(first_request, bridge)
    else:
        first_signature = _bridge_motion_signature(first_request, platform_type)
        _replace_with_substantive_motion_primitives(
            second_request, platform_type
        )
        second_signature = _bridge_motion_signature(second_request, platform_type)
        assert first_signature != second_signature
        assert len(first_signature) != len(second_signature)
        bridge = bridge_api.PlannerBridge()
        evidence = None

    first_capability_sha256 = (
        formal_cache_module._physical_capability_content_sha256(first_capability)
    )
    second_capability_sha256 = (
        formal_cache_module._physical_capability_content_sha256(second_capability)
    )
    assert first_capability_sha256 == second_capability_sha256

    first = _real_physical_coverability_product(
        first_request,
        platform_type=platform_type,
        capability_sha256=first_capability_sha256,
        bridge=bridge,
        hopper_evidence=evidence,
    )
    second = _real_physical_coverability_product(
        second_request,
        platform_type=platform_type,
        capability_sha256=second_capability_sha256,
        bridge=bridge,
        hopper_evidence=evidence,
    )

    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    np.testing.assert_array_equal(first[2], second[2])


def test_truth_hopper_physical_projection_uses_certified_landing_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = object()
    evidence_grid = object()
    calls: list[tuple[object, float, object]] = []
    reachable = np.zeros((256, 256), dtype=np.uint8)
    reachable[128, 128] = 1
    output = SimpleNamespace(
        platform_type="HOPPER",
        reachable=reachable,
        algorithm_id="cpp-hopper-certified-bidirectional-bfs/v3",
    )
    certified = np.zeros((256, 256), dtype=np.bool_)
    certified[127, 128] = True
    aim = np.zeros((256, 256, 3), dtype=np.float64)
    aim[127, 128] = (514.0, 510.0, 1.0)
    evidence = SimpleNamespace(
        bridge_grid=evidence_grid,
        certified_pose_mask=certified,
        aim_positions_m=aim,
        evidence_algorithm_id="cpp-hopper-exact-landing-evidence/v2",
    )

    class Bridge:
        def project_reachability(
            self, request_value: object, distance: float, evidence_value: object
        ) -> object:
            calls.append((request_value, distance, evidence_value))
            return output

    request_kwargs: list[dict[str, object]] = []
    evidence_kwargs: list[dict[str, object]] = []

    def projection_request(*_args, **kwargs):
        request_kwargs.append(kwargs)
        return request

    def hopper_evidence(**kwargs):
        evidence_kwargs.append(kwargs)
        return evidence

    monkeypatch.setattr(formal_cache_module, "_projection_request", projection_request)
    monkeypatch.setattr(formal_cache_module, "_hopper_landing_evidence", hopper_evidence)
    exact_start = (514.0, 510.0, 1.0)

    result = formal_cache_module._build_truth_physical_reachability(
        platform=SimpleNamespace(
            platform_type="HOPPER", content_sha256=_sha("hopper-capability")
        ),
        scene=object(),
        projected=object(),
        start_cell=(127, 128),
        exact_start_position_m=exact_start,
        bridge=Bridge(),
    )

    assert calls == [(request, 30.0, evidence_grid)]
    assert request_kwargs == [
        {"start_cell": (127, 128), "exact_start_position_m": exact_start}
    ]
    assert evidence_kwargs[0]["exact_start_position_m"] == exact_start
    assert result.physical_reachability_algorithm_id == output.algorithm_id
    assert result.physical_safe_pose_count == 1
    assert result.physically_reachable_pose_count == 1
    assert result.physical_observation_pose_mask[127, 128]
    np.testing.assert_array_equal(
        result.observation_positions_m,
        np.asarray([[514.0, 510.0, 1.0]], dtype=np.float64),
    )


def test_truth_hopper_physical_projection_rejects_exact_start_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_start = (514.0, 510.0, 1.0)
    reachable = np.zeros((256, 256), dtype=np.uint8)
    reachable[128, 128] = 1
    certified = np.zeros((256, 256), dtype=np.bool_)
    certified[127, 128] = True
    aim = np.zeros((256, 256, 3), dtype=np.float64)
    aim[127, 128] = (514.001, 510.0, 1.0)
    evidence = SimpleNamespace(
        bridge_grid=object(),
        certified_pose_mask=certified,
        aim_positions_m=aim,
        evidence_algorithm_id="test/hopper-evidence/v1",
    )
    monkeypatch.setattr(
        formal_cache_module,
        "_projection_request",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        formal_cache_module,
        "_hopper_landing_evidence",
        lambda **_kwargs: evidence,
    )
    bridge = SimpleNamespace(
        project_reachability=lambda *_args: SimpleNamespace(
            platform_type="HOPPER",
            reachable=reachable,
            algorithm_id="test/hopper-reachability/v1",
        )
    )

    with pytest.raises(FormalCacheError, match="exact start.*drift"):
        formal_cache_module._build_truth_physical_reachability(
            platform=SimpleNamespace(platform_type="HOPPER"),
            scene=object(),
            projected=object(),
            start_cell=(127, 128),
            exact_start_position_m=exact_start,
            bridge=bridge,
        )


def test_projection_request_binds_canonical_exact_hopper_start() -> None:
    canvas = MapCanvas.from_roi_bounds(
        "e" * 64, (0.0, 0.0, 1024.0, 1024.0)
    )
    shape = (256, 256)
    projected = SimpleNamespace(
        canvas=canvas,
        elevation_m=np.full(shape, 7.0, dtype=np.float32),
        valid_mask=np.ones(shape, dtype=np.bool_),
        physical_obstacle_ratio=np.zeros(shape, dtype=np.float32),
        physical_obstacle_height_m=np.zeros(shape, dtype=np.float32),
        forbidden_ratio=np.zeros(shape, dtype=np.float32),
    )
    start_cell = (127, 128)
    center_x, center_y = canvas.grid_center_world(*start_cell)
    exact_start = (center_x + 0.02, center_y - 0.02, 8.25)

    request = formal_cache_module._projection_request(
        _frozen_proxy_capability("HOPPER", alternate_motion=False),
        SimpleNamespace(scene_id="f" * 64),
        projected,
        start_cell=start_cell,
        exact_start_position_m=exact_start,
    )

    state_position = request.current_state.pose.position_m
    goal_position = request.goal.target.position_m
    assert (
        float(state_position.x),
        float(state_position.y),
        float(state_position.z),
    ) == exact_start
    assert (
        float(goal_position.x),
        float(goal_position.y),
        float(goal_position.z),
    ) == exact_start
    with pytest.raises(FormalCacheError, match="canonical"):
        formal_cache_module._projection_request(
            _frozen_proxy_capability("HOPPER", alternate_motion=False),
            SimpleNamespace(scene_id="f" * 64),
            projected,
            start_cell=start_cell,
            exact_start_position_m=(
                exact_start[0] + 0.0000001,
                exact_start[1],
                exact_start[2],
            ),
        )


def test_ground_coverability_keeps_physical_positions_outside_mission_roi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class GroundCapability:
        maximum_slope_rad: float
        motion_primitives: tuple[str, ...]

    @dataclass(frozen=True)
    class ObservationCapability:
        sensor_range_m: float
        sensor_fov_rad: float

    platform = SimpleNamespace(
        platform_type="WHEELED",
        capability_type="WHEELED",
        capability_version="1.0.0",
        platform_id="test-rover",
        base_frame_id="base_link",
        typed_capability=GroundCapability(0.4, ("forward",)),
        observation_capability=ObservationCapability(30.0, 6.0),
    )

    class CapabilityBundle:
        @staticmethod
        def for_platform(platform_type: str) -> object:
            assert platform_type == "WHEELED"
            return platform

    class Canvas:
        bounds_m = (0.0, 0.0, 256.0, 256.0)
        geometry = SimpleNamespace(resolution_m=1.0)

        @staticmethod
        def grid_center_world(row: int, column: int) -> tuple[float, float]:
            return float(column) + 0.5, float(row) + 0.5

    projected = SimpleNamespace(
        canvas=Canvas(),
        elevation_m=np.zeros((256, 256), dtype=np.float32),
    )
    physical = np.zeros((256, 256), dtype=np.bool_)
    physical[127, 127] = True  # qualified start inside the mission ROI
    physical[127, 190] = True  # outside ROI but able to observe back into it
    positions = np.asarray(
        [[127.5, 127.5, 0.0], [190.5, 127.5, 0.0]], dtype=np.float64
    )
    projection = SimpleNamespace(
        physical_observation_pose_mask=physical,
        observation_positions_m=positions,
        physical_projection_schema="lunar-physical-coverability-projection/v1",
        physical_reachability_algorithm_id=(
            "cpp-ground-start-connected-component/v1"
        ),
        physical_evidence_algorithm_id=(
            "cpp-safe-traversability-projection/v1"
        ),
        physical_safe_pose_count=2,
        physically_reachable_pose_count=2,
    )
    detail_mask = np.ones((256, 256), dtype=np.bool_)
    detail = SimpleNamespace(
        mission_target_detail_cell_count=detail_mask.size,
        coverable_detail_cell_count=detail_mask.size,
        mission_target_mask_sha256=mask_sha256(detail_mask),
        detail_shape=detail_mask.shape,
        coverable_detail_bits=pack_detail_mask(detail_mask),
        coverable_ratio=np.ones((256, 256), dtype=np.float32),
        coverable_mask_sha256=mask_sha256(detail_mask),
    )
    received_positions: list[np.ndarray | None] = []

    monkeypatch.setattr(
        formal_cache_module,
        "_build_truth_physical_reachability",
        lambda **_kwargs: projection,
    )

    def build_detail(**kwargs):
        received_positions.append(kwargs["physical_observation_positions_m"])
        return detail

    monkeypatch.setattr(
        formal_cache_module, "_build_platform_detail_coverability", build_detail
    )
    monkeypatch.setattr(
        formal_cache_module,
        "_initial_coverable_fraction",
        lambda **_kwargs: 0.1,
    )

    product = formal_cache_module._build_scene_platform_coverability(
        platform_type="WHEELED",
        capability_bundle=CapabilityBundle(),
        qualification=SimpleNamespace(
            cell=(127, 127), initial_candidate_count=1
        ),
        scene=object(),
        projected=projected,
        mission_roi=np.zeros((256, 256), dtype=np.bool_),
        detail_shape=(256, 256),
    )

    assert len(received_positions) == 1
    np.testing.assert_array_equal(received_positions[0], positions)
    np.testing.assert_array_equal(product.observation_positions_m, positions)
    assert product.physical_evidence_algorithm_id == (
        "cpp-safe-traversability-projection/v1"
    )


def test_truth_physical_start_failure_marks_only_that_platform_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class HopperCapability:
        landing_support_radius_m: float

    @dataclass(frozen=True)
    class ObservationCapability:
        sensor_range_m: float
        sensor_fov_rad: float

    class CapabilityBundle:
        @staticmethod
        def for_platform(platform_type: str) -> object:
            assert platform_type == "HOPPER"
            return SimpleNamespace(
                platform_type=platform_type,
                capability_type="HOPPER",
                capability_version="1.0.0",
                platform_id="test-hopper",
                base_frame_id="base_link",
                typed_capability=HopperCapability(0.4),
                observation_capability=ObservationCapability(30.0, 6.0),
            )

    def reject_start(**_kwargs):
        raise RuntimeError("HOPPER_START_LANDING_NOT_CERTIFIED")

    monkeypatch.setattr(
        formal_cache_module,
        "_build_truth_physical_reachability",
        reject_start,
    )
    projected = SimpleNamespace(
        canvas=SimpleNamespace(
            bounds_m=(0.0, 0.0, 256.0, 256.0),
            geometry=SimpleNamespace(resolution_m=1.0),
        )
    )

    result = formal_cache_module._build_scene_platform_coverability(
        platform_type="HOPPER",
        capability_bundle=CapabilityBundle(),
        qualification=SimpleNamespace(
            cell=(3, 4),
            initial_candidate_count=1,
            exact_start_position_m=(4.5, 252.5, 0.0),
        ),
        scene=object(),
        projected=projected,
        mission_roi=np.ones((2, 2), dtype=np.bool_),
        detail_shape=(256, 256),
    )

    assert result.coverability.eligible is False
    assert (
        result.coverability.ineligible_reason is IneligibleReason.UNSAFE_START
    )
    assert result.coverability.qualified_start_cell is None
    assert result.observation_positions_m.shape == (0, 3)


def test_truth_physical_projection_does_not_hide_a_non_start_native_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapabilityBundle:
        @staticmethod
        def for_platform(_platform_type: str) -> object:
            return object()

    def fail_projection(**_kwargs):
        raise RuntimeError("HOPPER_REACHABILITY_CERTIFICATION_INVALID")

    monkeypatch.setattr(
        formal_cache_module,
        "_build_truth_physical_reachability",
        fail_projection,
    )

    with pytest.raises(
        RuntimeError,
        match="HOPPER_REACHABILITY_CERTIFICATION_INVALID",
    ):
        formal_cache_module._build_scene_platform_coverability(
            platform_type="HOPPER",
            capability_bundle=CapabilityBundle(),
            qualification=SimpleNamespace(
                cell=(3, 4),
                initial_candidate_count=1,
                exact_start_position_m=(4.5, 252.5, 0.0),
            ),
            scene=object(),
            projected=object(),
            mission_roi=np.ones((2, 2), dtype=np.bool_),
            detail_shape=(256, 256),
        )


def test_cache_root_must_be_absolute_and_outside_git() -> None:
    identity = _identity(_sha("scenario"))
    with pytest.raises(FormalCacheError, match="absolute"):
        write_formal_cache(
            pathlib.Path("relative-cache"),
            identity=identity,
            scenario_manifest=_scenario_document(_sha("scene")),
            materialization="preflight",
            scenes=(),
            repository_root=REPOSITORY_ROOT,
        )
    with pytest.raises(FormalCacheError, match="outside"):
        write_formal_cache(
            REPOSITORY_ROOT / "forbidden-cache",
            identity=identity,
            scenario_manifest=_scenario_document(_sha("scene")),
            materialization="preflight",
            scenes=(),
            repository_root=REPOSITORY_ROOT,
        )
    assert not (REPOSITORY_ROOT / "forbidden-cache").exists()


def test_preflight_cache_round_trip_is_ineligible_for_formal_use(
    tmp_path: pathlib.Path,
) -> None:
    root, identity, manifest = _write(tmp_path)

    cache = load_formal_cache(
        root / "cache-manifest.json",
        expected_identity=identity,
    )
    arrays = cache.load_scene(_sha("scene"))

    assert FORMAL_CACHE_SCHEMA == "lunar-formal-training-cache/v6"
    assert manifest["schema"] == FORMAL_CACHE_SCHEMA
    assert manifest["materialization"] == "preflight"
    assert manifest["platform_eligibility"]["WHEELED"]["splits"]["train"] == {
        "total_scene_count": 1,
        "eligible_scene_count": 1,
        "feasibility_rate": 1.0,
        "ineligible_reason_counts": {},
    }
    assert manifest["exact_common_evaluation"]["scene_count"] == 1
    platform = manifest["scenes"][0]["platform_coverability"]["WHEELED"]
    assert platform["qualified_start_cell"] == [127, 127]
    assert platform["physical_observation_pose_shape"] == [256, 256]
    assert platform["physical_safe_pose_count"] == 256 * 256
    assert platform["physically_reachable_pose_count"] == 256 * 256
    assert len(platform["physical_projection_sha256"]) == 64
    assert len(platform["capability_content_sha256"]) == 64
    assert len(platform["start_identity_sha256"]) == 64
    assert platform["physical_evidence_algorithm_id"] == (
        "test-physical-evidence/wheeled"
    )
    assert platform["physical_grid_axis_convention"] == (
        "north-up-row-major-row-decreases-y-column-increases-x"
    )
    assert not any("primitive" in field for field in platform)
    assert platform["eligible"] is True
    assert platform["ineligible_reason"] is None
    assert platform["exact"] is True
    assert cache.formal_eligible is False
    assert arrays["elevation_m"].shape == (256, 256)
    assert arrays["wheeled_hard_feasible"].dtype == np.uint8
    assert arrays["wheeled_physical_observation_pose_bits"].dtype == np.uint8
    assert arrays["wheeled_coverable_detail_bits"].dtype == np.uint8
    assert arrays["wheeled_coverable_ratio"].dtype == np.dtype("<f4")
    assert arrays["hopper_physical_observation_positions_um"].dtype.str == (
        "<i8"
    )
    assert arrays["hopper_physical_observation_positions_um"].shape == (
        256 * 256,
        3,
    )
    array_contract = manifest["scenes"][0]["arrays"]
    physical_bits = array_contract["wheeled_physical_observation_pose_bits"]
    assert physical_bits["dtype"] == "|u1"
    assert physical_bits["byte_order"] == "not-applicable"
    detail_bits = array_contract["wheeled_coverable_detail_bits"]
    assert detail_bits["dtype"] == "|u1"
    assert detail_bits["byte_order"] == "not-applicable"
    ratio = array_contract["wheeled_coverable_ratio"]
    assert ratio["dtype"] == "<f4"
    assert ratio["byte_order"] == "little"
    with pytest.raises(FormalCacheError, match="full"):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=identity,
            require_full=True,
        )


@pytest.mark.parametrize(
    ("array_name", "wrong_dtype"),
    (
        ("wheeled_physical_observation_pose_bits", np.dtype("<u2")),
        ("wheeled_coverable_detail_bits", np.dtype("<u2")),
        ("wheeled_coverable_ratio", np.dtype(">f4")),
        ("hopper_physical_observation_positions_um", np.dtype(">i8")),
    ),
)
def test_formal_cache_rejects_noncanonical_coverability_array_dtype(
    tmp_path: pathlib.Path,
    array_name: str,
    wrong_dtype: np.dtype[object],
) -> None:
    root, identity, manifest = _write(tmp_path)
    scene_path = root / manifest["scenes"][0]["relative_path"]
    with np.load(scene_path, allow_pickle=False) as archive:
        original = archive[array_name].copy()
    replacement = original.astype(wrong_dtype)
    _rewrite_scene_array_and_resign_manifest(
        root, manifest, array_name, replacement
    )

    with pytest.raises(FormalCacheError, match="dtype|byte order"):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=identity,
        )


@pytest.mark.parametrize("authority_drift", ("big_endian", "shape"))
def test_formal_cache_rejects_noncanonical_ground_projection_elevation(
    tmp_path: pathlib.Path,
    authority_drift: str,
) -> None:
    root, identity, manifest = _write(tmp_path)
    scene_path = root / manifest["scenes"][0]["relative_path"]
    with np.load(scene_path, allow_pickle=False) as archive:
        elevation = archive["elevation_m"].copy()
    replacement = (
        elevation.astype(">f4")
        if authority_drift == "big_endian"
        else elevation[..., np.newaxis]
    )
    _rewrite_scene_array_and_resign_manifest(
        root, manifest, "elevation_m", replacement
    )

    with pytest.raises(FormalCacheError, match="shape|dtype|byte order"):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=identity,
        )


def test_formal_cache_recomputes_physical_projection_digest(
    tmp_path: pathlib.Path,
) -> None:
    root, identity, manifest = _write(tmp_path)
    manifest["scenes"][0]["platform_coverability"]["WHEELED"][
        "physical_projection_sha256"
    ] = "f" * 64
    _refresh_scene_integrity_and_resign(root, manifest)

    with pytest.raises(FormalCacheError, match="physical projection"):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=identity,
        )


def test_formal_cache_recomputes_hopper_exact_start_identity(
    tmp_path: pathlib.Path,
) -> None:
    root, identity, manifest = _write(tmp_path)
    scene = manifest["scenes"][0]
    hopper = scene["platform_coverability"]["HOPPER"]
    scene_path = root / scene["relative_path"]
    with np.load(scene_path, allow_pickle=False) as archive:
        physical = formal_cache_module.unpack_detail_mask(
            archive["hopper_physical_observation_pose_bits"].copy(),
            (256, 256),
        )
        positions_m = formal_cache_module._positions_m_from_canonical_um(
            np.ascontiguousarray(
                archive["hopper_physical_observation_positions_um"]
            )
        )
    resolution_m, origin_m, bounds_m = (
        formal_cache_module._physical_grid_geometry(scene["world_bounds_m"])
    )
    forged_start_sha256 = _sha("forged-hopper-exact-start")
    hopper["start_identity_sha256"] = forged_start_sha256
    hopper["physical_projection_sha256"] = (
        canonical_physical_projection_sha256(
            platform_type="HOPPER",
            physical_reachability_algorithm_id=(
                hopper["physical_reachability_algorithm_id"]
            ),
            physical_evidence_algorithm_id=(
                hopper["physical_evidence_algorithm_id"]
            ),
            physical_observation_pose_mask=physical,
            physical_observation_positions_m=positions_m,
            physical_grid_resolution_m=resolution_m,
            physical_grid_origin_m=origin_m,
            physical_grid_world_bounds_m=bounds_m,
            physical_grid_axis_convention=(
                hopper["physical_grid_axis_convention"]
            ),
            capability_content_sha256=hopper["capability_content_sha256"],
            start_identity_sha256=forged_start_sha256,
        )
    )
    _resign_manifest(root, manifest)

    with pytest.raises(FormalCacheError, match="start identity"):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=identity,
        )


@pytest.mark.parametrize(
    "authority_tamper",
    (
        "geometry",
        "ground_elevation",
        "positions",
        "position_order",
        "evidence_algorithm",
        "reachability_algorithm",
    ),
)
def test_formal_cache_rejects_physical_projection_authority_tamper(
    tmp_path: pathlib.Path,
    authority_tamper: str,
) -> None:
    root, identity, manifest = _write(tmp_path)
    scene = manifest["scenes"][0]
    if authority_tamper == "geometry":
        scene["world_bounds_m"] = [0.0, 0.0, 512.0, 512.0]
        _refresh_scene_integrity_and_resign(root, manifest)
    elif authority_tamper == "ground_elevation":
        scene_path = root / scene["relative_path"]
        with np.load(scene_path, allow_pickle=False) as archive:
            elevation = archive["elevation_m"].copy()
        elevation[0, 0] += np.float32(0.001)
        _rewrite_scene_array_and_resign_manifest(
            root, manifest, "elevation_m", elevation
        )
    elif authority_tamper in {"positions", "position_order"}:
        scene_path = root / scene["relative_path"]
        with np.load(scene_path, allow_pickle=False) as archive:
            positions = archive[
                "hopper_physical_observation_positions_um"
            ].copy()
        if authority_tamper == "positions":
            positions[0, 2] += 1
        else:
            positions[[0, 1]] = positions[[1, 0]]
        _rewrite_scene_array_and_resign_manifest(
            root,
            manifest,
            "hopper_physical_observation_positions_um",
            positions,
        )
    elif authority_tamper == "evidence_algorithm":
        scene["platform_coverability"]["HOPPER"][
            "physical_evidence_algorithm_id"
        ] = "tampered-hopper-evidence/v9"
        _refresh_scene_integrity_and_resign(root, manifest)
    else:
        scene["platform_coverability"]["HOPPER"][
            "physical_reachability_algorithm_id"
        ] = "tampered-hopper-connectivity/v9"
        _refresh_scene_integrity_and_resign(root, manifest)

    with pytest.raises(
        FormalCacheError,
        match="physical (projection|observation|start identity)",
    ):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=identity,
        )


def test_formal_cache_rejects_hopper_position_shape_drift(
    tmp_path: pathlib.Path,
) -> None:
    root, identity, manifest = _write(tmp_path)
    scene_path = root / manifest["scenes"][0]["relative_path"]
    with np.load(scene_path, allow_pickle=False) as archive:
        positions = archive["hopper_physical_observation_positions_um"].copy()
    _rewrite_scene_array_and_resign_manifest(
        root,
        manifest,
        "hopper_physical_observation_positions_um",
        positions[:-1],
    )

    with pytest.raises(FormalCacheError, match="shape"):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=identity,
        )


@pytest.mark.parametrize(
    "schema_drift",
    ("missing_field", "unknown_field", "missing_array", "unknown_array"),
)
def test_formal_cache_rejects_projection_authority_schema_drift(
    tmp_path: pathlib.Path,
    schema_drift: str,
) -> None:
    root, identity, manifest = _write(tmp_path)
    scene = manifest["scenes"][0]
    hopper = scene["platform_coverability"]["HOPPER"]
    arrays = scene["arrays"]
    if schema_drift == "missing_field":
        hopper.pop("physical_evidence_algorithm_id")
    elif schema_drift == "unknown_field":
        hopper["physical_projection_optional"] = "not-allowed"
    elif schema_drift == "missing_array":
        arrays.pop("hopper_physical_observation_positions_um")
    else:
        arrays["hopper_physical_observation_positions_optional"] = dict(
            arrays["hopper_physical_observation_positions_um"]
        )
    _resign_manifest(root, manifest)

    with pytest.raises(FormalCacheError, match="coverability"):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=identity,
        )


def test_formal_cache_rejects_v5_manifest(
    tmp_path: pathlib.Path,
) -> None:
    root, identity, _ = _write(tmp_path)
    manifest_path = root / "cache-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "lunar-formal-training-cache/v5"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(FormalCacheError, match="schema"):
        load_formal_cache(manifest_path, expected_identity=identity)


@pytest.mark.parametrize(
    "field",
    (
        "primitive_state_count",
        "certified_edge_count",
        "recoverable_state_count",
        "primitive_state_schema",
        "primitive_set_sha256",
        "world_evidence_sha256",
        "reachability_graph_sha256",
    ),
)
def test_formal_cache_rejects_residual_primitive_identity_field(
    tmp_path: pathlib.Path,
    field: str,
) -> None:
    root, identity, _ = _write(tmp_path)
    manifest_path = root / "cache-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    platform = manifest["scenes"][0]["platform_coverability"]["WHEELED"]
    platform[field] = 1 if field.endswith("_count") else _sha(field)
    body = dict(manifest)
    body.pop("cache_manifest_sha256")
    manifest["cache_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FormalCacheError, match="primitive identity"):
        load_formal_cache(manifest_path, expected_identity=identity)


def test_formal_cache_rejects_residual_primitive_identity_outside_payload(
    tmp_path: pathlib.Path,
) -> None:
    root, identity, _ = _write(tmp_path)
    manifest_path = root / "cache-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["primitive_set_sha256"] = _sha("residual-primitive-set")
    body = dict(manifest)
    body.pop("cache_manifest_sha256")
    manifest["cache_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FormalCacheError, match="primitive identity"):
        load_formal_cache(manifest_path, expected_identity=identity)


def test_cache_records_platform_local_eligibility_and_exact_common_intersection(
    tmp_path: pathlib.Path,
) -> None:
    scene_id = _sha("unstartable-scene")
    scenario = _scenario_document(scene_id)
    identity = _identity(str(scenario["scenario_manifest_sha256"]))
    root = tmp_path / "unstartable-cache"

    manifest = write_formal_cache(
        root,
        identity=identity,
        scenario_manifest=scenario,
        materialization="preflight",
        scenes=(
            _scene(
                scene_id,
                ineligible_platforms=frozenset(("HOPPER",)),
            ),
        ),
        repository_root=REPOSITORY_ROOT,
    )

    entry = manifest["scenes"][0]["platform_coverability"]
    assert entry["WHEELED"]["eligible"] is True
    assert entry["HOPPER"]["eligible"] is False
    assert entry["HOPPER"]["ineligible_reason"] == (
        "MISSION_COVERABLE_BELOW_95"
    )
    assert manifest["platform_eligibility"]["WHEELED"]["splits"]["train"][
        "eligible_scene_count"
    ] == 1
    assert manifest["platform_eligibility"]["HOPPER"]["splits"]["train"] == {
        "total_scene_count": 1,
        "eligible_scene_count": 0,
        "feasibility_rate": 0.0,
        "ineligible_reason_counts": {"MISSION_COVERABLE_BELOW_95": 1},
    }
    assert manifest["exact_common_evaluation"]["scene_count"] == 0


def test_formal_platform_eligibility_requires_each_platform_and_split_lane(
) -> None:
    counts = {
        platform: {
            split: {"total_scene_count": 3, "eligible_scene_count": 1}
            for split in ("train", "validation", "test", "holdout")
        }
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    }

    assert _formal_platform_eligibility_ready("full", counts)
    assert not _formal_platform_eligibility_ready("preflight", counts)
    counts["HOPPER"]["holdout"]["eligible_scene_count"] = 0
    assert not _formal_platform_eligibility_ready("full", counts)


def test_platform_schedule_identity_binds_platform_split_and_order() -> None:
    first = platform_scenario_schedule_id(
        "WHEELED", "train", ("a" * 64, "b" * 64)
    )

    assert first == platform_scenario_schedule_id(
        "WHEELED", "train", ("a" * 64, "b" * 64)
    )
    assert first != platform_scenario_schedule_id(
        "HOPPER", "train", ("a" * 64, "b" * 64)
    )
    assert first != platform_scenario_schedule_id(
        "WHEELED", "train", ("b" * 64, "a" * 64)
    )


@pytest.mark.parametrize("drift", ("missing", "extra", "hash"))
def test_cache_inventory_fails_closed_on_file_drift(
    tmp_path: pathlib.Path, drift: str
) -> None:
    root, identity, manifest = _write(tmp_path)
    scene_path = root / manifest["scenes"][0]["relative_path"]
    if drift == "missing":
        scene_path.unlink()
    elif drift == "extra":
        (root / "untracked.bin").write_bytes(b"extra")
    else:
        scene_path.write_bytes(scene_path.read_bytes() + b"tamper")

    with pytest.raises(FormalCacheError, match="inventory|hash|size"):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=identity,
        )


@pytest.mark.parametrize(
    "field",
    ("capability_sha256", "generator_sha256", "v3_sha256", "split_sha256"),
)
def test_cache_identity_rejects_current_authority_drift(
    tmp_path: pathlib.Path, field: str
) -> None:
    root, identity, _ = _write(tmp_path)
    changed = replace(identity, **{field: "f" * 64})

    with pytest.raises(FormalCacheError, match=field):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=changed,
        )


def test_second_write_is_verification_only_and_keeps_file_timestamps(
    tmp_path: pathlib.Path,
) -> None:
    root, identity, first = _write(tmp_path)
    mtimes = {
        path.relative_to(root).as_posix(): path.stat().st_mtime_ns
        for path in root.rglob("*")
        if path.is_file()
    }

    second = write_formal_cache(
        root,
        identity=identity,
        scenario_manifest=_scenario_document(_sha("scene")),
        materialization="preflight",
        scenes=(_scene(_sha("scene")),),
        repository_root=REPOSITORY_ROOT,
    )

    assert second["cache_manifest_sha256"] == first["cache_manifest_sha256"]
    assert {
        path.relative_to(root).as_posix(): path.stat().st_mtime_ns
        for path in root.rglob("*")
        if path.is_file()
    } == mtimes


def test_cache_manifest_is_canonical_utf8_json(tmp_path: pathlib.Path) -> None:
    root, _, manifest = _write(tmp_path)
    raw = (root / "cache-manifest.json").read_bytes()
    decoded = json.loads(raw.decode("utf-8"))

    assert raw.endswith(b"\n")
    assert decoded == manifest

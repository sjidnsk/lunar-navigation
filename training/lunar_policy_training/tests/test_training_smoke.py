from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from zipfile import ZIP_DEFLATED, ZipFile

import lunar_planner_training_bridge as bridge_api
import numpy as np
import pytest
import rasterio
import torch
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely import from_wkb
from lunar_planner_training_bridge import (
    ExecutionDirective,
    PlannerBridge,
    PlanningOutcome,
)


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

from lunar_policy_training.cli import (  # noqa: E402
    ResumablePPOTrainer,
    PreflightError,
    _update_run_manifest,
    run_cuda_interrupt_resume_smoke,
)
import lunar_policy_training.cli as training_cli  # noqa: E402
from lunar_policy_training.capability_freeze import (  # noqa: E402
    load_frozen_capability_bundle,
)
from lunar_policy_training.checkpoint import (  # noqa: E402
    RunIdentity,
    build_training_checkpoint,
    config_sha256,
    load_checkpoint,
    load_checkpoint_for_resume,
    restore_training_state,
    save_checkpoint_atomic,
)
from lunar_policy_training.config import load_training_config  # noqa: E402
from lunar_policy_training.environment.candidate_builder import (  # noqa: E402
    CandidateBuilderV2,
)
from lunar_policy_training.environment.observation_boundary import (  # noqa: E402
    ObservationBoundaryController,
    SensorBoundaryEvidence,
)
from lunar_policy_training.environment.sensor_observation import (  # noqa: E402
    SensorObservationState,
    TrainingObservedGrid,
    TrainingWorldTruth,
)
from lunar_policy_training.environment.visibility import (  # noqa: E402
    NativeVisibilityEstimator,
    SensorGeometry,
)
from lunar_policy_training.environment.macro_step import (  # noqa: E402
    ExecutionEvents,
    PlannerTransition,
)
from lunar_policy_training.environment.observation_builder import (  # noqa: E402
    LocalObservation,
    MissionRaster,
    ObservationBuilderV2,
    ObservedWorld,
    PlatformProjection,
    Pose2,
)
from lunar_policy_training.polar_data.hazards import (  # noqa: E402
    GENERATOR_VERSION,
    generate_hazard_scene,
    physical_obstacle_ratio,
)
from lunar_policy_training.polar_data.raster import (  # noqa: E402
    LOCAL_GEOMETRY,
    MapCanvas,
    load_polar_window,
)
from lunar_policy_training.polar_data.source_lock import (  # noqa: E402
    PolarSourceLock,
    write_aggregate_source_lock,
)
from lunar_policy_training.polar_data.split import (  # noqa: E402
    build_split_manifest,
)
from lunar_policy_training.policy.cross_attention import (  # noqa: E402
    CrossAttentionPolicy,
    sample_action,
)
from lunar_policy_training.policy.action_semantics import apply_goal_theta  # noqa: E402
from lunar_policy_training.policy.observation import PolicyBatch  # noqa: E402
from lunar_policy_training.ppo.rollout import RolloutBatch  # noqa: E402
from lunar_policy_training.ppo.trainer import PPOTrainer  # noqa: E402
from lunar_policy_training.reward import reward_weights_sha256  # noqa: E402
from lunar_policy_training.training_semantics import (  # noqa: E402
    FORMAL_SENSOR_FOV_RAD,
    FORMAL_SENSOR_RANGE_M,
    training_semantics_sha256,
)


def _transition() -> PlannerTransition:
    observation = PolicyBatch(
        prior_channels=torch.zeros((1, 7, 8, 8), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 8, 8, 8), dtype=torch.float32),
        local_crop=torch.zeros((1, 8, 8, 8), dtype=torch.float32),
        frontier_features=torch.zeros((1, 2, 22), dtype=torch.float32),
        pose_features=torch.zeros((1, 6), dtype=torch.float32),
        candidate_mask=torch.tensor([[True, False]], dtype=torch.bool),
        platform_context=torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float32
        ),
    )
    return PlannerTransition(
        next_observation=observation,
        mission_observed_delta=0.5,
        priority_observed_delta=0.25,
        normalized_plan_or_execution_cost=0.1,
        normalized_macro_step_time=0.2,
        executed_without_new_coverage=False,
        success_first_crossing=False,
        episode_ended_without_success=False,
        hard_safety_violation=False,
        cancellation_expected=False,
        cpp_exception=None,
        planning_outcome=PlanningOutcome.INVALID_REQUEST,
        execution_directive=ExecutionDirective.NO_SAFE_REFERENCE,
        reason_code="TEST",
        terminated=False,
        execution_events=ExecutionEvents(),
    )


def test_resumable_trainer_uses_injected_transition_reward() -> None:
    """Would fail if Task 4 rewards required replacing the Task 3 trainer."""
    trainer = ResumablePPOTrainer(
        CrossAttentionPolicy(),
        reward_fn=lambda transition: (
            transition.mission_observed_delta + transition.priority_observed_delta
        ),
        ppo_config=load_training_config(
            REPOSITORY_ROOT / "training/configs/rtx4080_super_smoke.yaml"
        ).ppo,
        device="cpu",
    )

    assert trainer.reward_transition(_transition()) == 0.75


def _capability_fixture_module():
    path = pathlib.Path(__file__).with_name("test_capability_freeze.py")
    spec = importlib.util.spec_from_file_location("task9_capability_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command", ("train", "resume", "evaluate"))
def test_public_formal_commands_reject_external_bundle_argument_before_side_effects(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """An external development bundle cannot override the project authority."""
    artifact_root = tmp_path / "formal-artifacts"
    touched: list[str] = []
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: touched.append("cuda") or True,
    )
    monkeypatch.setattr(
        training_cli,
        "ParallelEnvPool",
        lambda *args, **kwargs: touched.append("workers"),
    )
    arguments = [command]
    if command == "train":
        arguments += [
            "--config",
            str(REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"),
        ]
    elif command == "resume":
        arguments += ["--checkpoint", str(artifact_root / "checkpoints/latest.pt")]
    else:
        arguments += [
            "--checkpoint",
            str(artifact_root / "checkpoints/latest.pt"),
            "--gate",
            str(REPOSITORY_ROOT / "training/configs/candidate_gate_v1.yaml"),
        ]
    arguments += [
        "--artifact-root",
        str(artifact_root),
        "--capability-lock",
        str(tmp_path / "test-only-capabilities.json"),
    ]

    with pytest.raises(SystemExit):
        training_cli.main(arguments)

    assert touched == []
    assert not artifact_root.exists()


def _archive_raster_bytes() -> bytes:
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="float32",
            crs="EPSG:3031",
            transform=from_origin(0.0, 10.0, 1.0, 1.0),
            nodata=-9999.0,
        ) as dataset:
            dataset.write(np.ones((1, 2, 2), dtype=np.float32))
        return memory.read()


def _write_synthetic_sources(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    root.mkdir(parents=True)
    raw = root / "raw"
    locks = root / "locks"
    splits = root / "splits"
    raw.mkdir()
    locks.mkdir()
    splits.mkdir()
    transform = from_origin(0.0, 50_000.0, 256.0, 256.0)
    dem = raw / "synthetic-dem.tif"
    elevation = np.linspace(
        0.0,
        1.0,
        num=192 * 384,
        dtype=np.float32,
    ).reshape(1, 192, 384)
    with rasterio.open(
        dem,
        "w",
        driver="GTiff",
        height=192,
        width=384,
        count=1,
        dtype="float32",
        crs="EPSG:3031",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(elevation)
    count = raw / "synthetic-count.tif"
    with rasterio.open(
        count,
        "w",
        driver="GTiff",
        height=192,
        width=384,
        count=1,
        dtype="uint8",
        crs="EPSG:3031",
        transform=transform,
    ) as dataset:
        dataset.write(np.ones((1, 192, 384), dtype=np.uint8))
    archive = raw / "synthetic-jaxa.zip"
    raster_bytes = _archive_raster_bytes()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
        for site in ("CR1", "GR1", "GR2", "LP1", "MP1", "MP2"):
            for kind in ("DTM", "ORTHO", "UNCERTAINTY"):
                output.writestr(f"{site}/{site}_{kind}.tif", raster_bytes)
    source_locks = (
        PolarSourceLock.from_file(
            "NASA_LOLA_87S_DEM",
            dem,
            citation="synthetic development smoke",
            license="test-only",
            final_url="https://example.invalid/synthetic-dem",
        ),
        PolarSourceLock.from_file(
            "NASA_LOLA_87S_COUNT",
            count,
            citation="synthetic development smoke",
            license="test-only",
            final_url="https://example.invalid/synthetic-count",
        ),
        PolarSourceLock.from_file(
            "JAXA_LUPEX_DATA_S1",
            archive,
            citation="synthetic development smoke",
            license="test-only",
            final_url="https://example.invalid/synthetic-jaxa",
        ),
    )
    aggregate = locks / "synthetic-source-lock.json"
    write_aggregate_source_lock(
        aggregate,
        raw,
        source_locks,
        repository_root=REPOSITORY_ROOT,
    )
    split = splits / "synthetic-split.json"
    build_split_manifest(
        aggregate,
        split,
        seed=17,
        repository_root=REPOSITORY_ROOT,
    )
    return aggregate, split


@dataclass(frozen=True)
class _SyntheticPolarInputs:
    world: ObservedWorld
    mission: MissionRaster
    pose: Pose2
    forbidden_ratio: np.ndarray


def _sample_local(
    canvas: MapCanvas,
    pose: Pose2,
    values: np.ndarray,
) -> np.ndarray:
    resolution = LOCAL_GEOMETRY.resolution_m
    half = LOCAL_GEOMETRY.size_m / 2.0
    left, _, _, top = canvas.bounds_m
    x = pose.x_m - half + (np.arange(LOCAL_GEOMETRY.cells) + 0.5) * resolution
    y = pose.y_m + half - (np.arange(LOCAL_GEOMETRY.cells) + 0.5) * resolution
    columns = np.floor((x - left) / canvas.geometry.resolution_m).astype(np.intp)
    rows = np.floor((top - y) / canvas.geometry.resolution_m).astype(np.intp)
    assert rows.min() >= 0 and rows.max() < canvas.geometry.cells
    assert columns.min() >= 0 and columns.max() < canvas.geometry.cells
    return np.ascontiguousarray(np.asarray(values)[np.ix_(rows, columns)])


def _synthetic_polar_inputs(
    dem_path: pathlib.Path,
    split_document: dict[str, object],
) -> _SyntheticPolarInputs:
    row = next(
        item
        for item in split_document["rows"]
        if item["source"] == "NASA_LOLA" and item["split"] == "train"
    )
    canvas = MapCanvas(row["window_sha256"], tuple(row["world_bounds_m"]))
    loaded = load_polar_window(dem_path, canvas)
    hazards = generate_hazard_scene(
        row["window_sha256"],
        17,
        canvas=canvas,
        rock_count=32,
        crater_count=8,
        no_go_count=8,
    )
    observed = loaded.observed_mask.copy()
    observed[:, 128:] = False
    elevation = loaded.elevation_m + hazards.crater_elevation_delta_m
    elevation[~np.isfinite(elevation)] = 0.0
    robot_x, robot_y = canvas.grid_center_world(128, 120)
    pose = Pose2(robot_x, robot_y, elevation_m=float(elevation[128, 120]))
    local = LocalObservation(
        canvas_id=canvas.identity,
        bounds_m=(
            robot_x - LOCAL_GEOMETRY.size_m / 2.0,
            robot_y - LOCAL_GEOMETRY.size_m / 2.0,
            robot_x + LOCAL_GEOMETRY.size_m / 2.0,
            robot_y + LOCAL_GEOMETRY.size_m / 2.0,
        ),
        elevation_m=_sample_local(canvas, pose, elevation),
        observed_mask=_sample_local(canvas, pose, observed),
        physical_obstacle_ratio=_sample_local(
            canvas,
            pose,
            hazards.physical_obstacle_layer.values,
        ),
    )
    world = ObservedWorld(
        canvas=canvas,
        elevation_m=elevation,
        observed_mask=observed,
        physical_obstacle_layer=hazards.physical_obstacle_layer,
        local=local,
    )
    mission = MissionRaster(
        canvas=canvas,
        priority=np.ones((256, 256), dtype=np.float32),
        roi_ratio=np.ones((256, 256), dtype=np.float32),
        remaining_decision_budget_ratio=1.0,
    )
    forbidden = physical_obstacle_ratio(
        tuple(from_wkb(item) for item in hazards.no_go_polygons_wkb),
        canvas=canvas,
    )
    return _SyntheticPolarInputs(world, mission, pose, forbidden)


def _bridge_vec3(x: float, y: float, z: float) -> bridge_api.Vec3:
    value = bridge_api.Vec3()
    value.x = x
    value.y = y
    value.z = z
    return value


def _bridge_grid_map(
    inputs: _SyntheticPolarInputs,
    *,
    frame_id: str,
    stamp_ns: int,
) -> bridge_api.GridMap:
    world = inputs.world
    canvas = world.canvas

    def layer(values: np.ndarray, dtype) -> bridge_api.GridLayer:
        south_up = np.ascontiguousarray(
            np.flipud(np.asarray(values)).astype(dtype, copy=False).reshape(-1)
        )
        return bridge_api.GridLayer(south_up)

    observed = world.observed_mask
    physical = np.where(
        observed,
        world.physical_obstacle_layer.values,
        0.0,
    ).astype(np.float32)
    forbidden = np.where(observed, inputs.forbidden_ratio, 0.0).astype(np.float32)
    grid = bridge_api.GridMap()
    grid.frame_id = frame_id
    grid.stamp.nanoseconds_since_epoch = stamp_ns
    grid.width = canvas.geometry.cells
    grid.height = canvas.geometry.cells
    grid.resolution_m = canvas.geometry.resolution_m
    grid.origin_m = _bridge_vec3(canvas.bounds_m[0], canvas.bounds_m[1], 0.0)
    grid.layers = {
        "elevation": layer(
            np.where(observed, world.elevation_m, 0.0),
            np.float32,
        ),
        "valid_mask": layer(observed, np.uint8),
        "obstacle": layer(physical > 0.0, np.uint8),
        "obstacle_height": layer(physical * 0.5, np.float32),
        "observation_age_s": layer(np.zeros_like(physical), np.float32),
        "observation_quality": layer(observed, np.float32),
        "elevation_variance": layer(np.zeros_like(physical), np.float32),
        "obstacle_variance": layer(np.zeros_like(physical), np.float32),
        "observation_count": layer(observed, np.uint32),
        "forbidden": layer(forbidden > 0.0, np.uint8),
    }
    return grid


def _synthetic_bridge_request(
    inputs: _SyntheticPolarInputs,
    wheeled,
) -> bridge_api.TrainingPlanRequest:
    canvas_id = inputs.world.canvas.identity
    stamp_ns = 1_000_000_000
    request = bridge_api.TrainingPlanRequest()
    request.request_id = f"polar-smoke/{canvas_id}/{wheeled.content_sha256}"
    request.mission_id = f"polar-smoke/{canvas_id}"
    request.mission_revision = 1
    request.platform_id = wheeled.platform_id
    request.capability_version = wheeled.capability_version
    request.global_map_generation = 1
    request.local_map_generation = 1
    request.map_from_odom_generation = 1
    request.state_time.nanoseconds_since_epoch = stamp_ns
    state = bridge_api.WheeledState()
    state.pose.position_m = _bridge_vec3(
        inputs.pose.x_m,
        inputs.pose.y_m,
        inputs.pose.elevation_m,
    )
    state.pose.orientation.w = 1.0
    request.current_state = state
    request.capability = wheeled.to_bridge_capability()
    goal = bridge_api.PointGoal()
    goal.position_m = _bridge_vec3(
        inputs.pose.x_m,
        inputs.pose.y_m,
        inputs.pose.elevation_m,
    )
    goal.tolerance_m = inputs.world.canvas.geometry.resolution_m
    request.goal.goal_id = f"polar-frontier/{canvas_id}"
    request.goal.target = goal
    apply_goal_theta(request.goal, "WHEELED", 0.0)
    request.world.global_map = _bridge_grid_map(
        inputs,
        frame_id="map",
        stamp_ns=stamp_ns,
    )
    request.world.local_map = _bridge_grid_map(
        inputs,
        frame_id="odom",
        stamp_ns=stamp_ns,
    )
    request.world.map_from_odom.parent_frame = "map"
    request.world.map_from_odom.child_frame = "odom"
    request.world.map_from_odom.stamp.nanoseconds_since_epoch = stamp_ns
    request.config.global_map.base_resolution_m = (
        inputs.world.canvas.geometry.resolution_m
    )
    request.config.wheel.xy_resolution_m = inputs.world.canvas.geometry.resolution_m
    request.config.wheel.yaw_bin_count = 64
    return request


def _platform_projection_from_cpp(
    inputs: _SyntheticPolarInputs,
    cpp_projection,
    *,
    capability_sha256: str,
    sensor_range_m: float,
) -> PlatformProjection:
    known = np.ascontiguousarray(np.flipud(cpp_projection.known).astype(bool))
    hard_feasible = np.ascontiguousarray(
        np.flipud(cpp_projection.hard_feasible).astype(bool)
    )
    clearance_m = np.ascontiguousarray(
        np.flipud(cpp_projection.clearance_m).astype(np.float32)
    )
    traversable = (known & hard_feasible).astype(np.float32)
    clearance_norm = np.where(
        known,
        np.clip(
            np.nan_to_num(
                clearance_m,
                nan=0.0,
                posinf=sensor_range_m,
                neginf=0.0,
            ),
            0.0,
            sensor_range_m,
        )
        / sensor_range_m,
        0.0,
    ).astype(np.float32)
    return PlatformProjection(
        canvas=inputs.world.canvas,
        traversable_ratio=traversable,
        local_traversable_ratio=_sample_local(
            inputs.world.canvas,
            inputs.pose,
            traversable,
        ),
        clearance_margin_norm=clearance_norm,
        source=f"cpp_v3/{capability_sha256}",
    )


def _policy_batch_from_projection(
    inputs: _SyntheticPolarInputs,
    projection: PlatformProjection,
) -> tuple[PolicyBatch, object]:
    candidates = CandidateBuilderV2(
        NativeVisibilityEstimator(
            SensorGeometry(
                FORMAL_SENSOR_RANGE_M,
                FORMAL_SENSOR_FOV_RAD,
            ),
            resolution_m=inputs.world.canvas.geometry.resolution_m,
        )
    ).build(
        inputs.world,
        inputs.mission,
        inputs.pose,
        projection,
    )
    assert candidates.count > 0
    arrays = ObservationBuilderV2().build(
        inputs.world,
        inputs.mission,
        inputs.pose,
        projection,
        candidates,
        "WHEELED",
    )
    return (
        PolicyBatch(**{name: torch.from_numpy(value) for name, value in arrays.items()}),
        candidates,
    )


def _one_row_rollout(batch: PolicyBatch, policy: CrossAttentionPolicy) -> tuple[RolloutBatch, object]:
    with torch.no_grad():
        output = policy(batch)
        action = sample_action(
            output,
            batch.candidate_mask,
            batch.platform_context,
            deterministic=True,
        )
    arrays = {
        name: getattr(batch, name).detach().cpu().numpy().copy()
        for name in batch.input_names
    }
    rollout = RolloutBatch(
        **arrays,
        selected_frontier_indices=action.selected_frontier_index.cpu().numpy().astype(np.int64),
        selected_thetas=action.selected_theta.cpu().numpy().astype(np.float32),
        old_log_prob_total=action.log_prob_total.cpu().numpy().astype(np.float32),
        old_values=action.value.cpu().numpy().astype(np.float32),
        advantages=np.ones((1,), dtype=np.float32),
        returns=(action.value.cpu().numpy() + 1.0).astype(np.float32),
    )
    return rollout, action


def test_cpu_pretraining_smoke_links_data_v3_update_checkpoint_and_resume(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if the bounded development chain drifted across a handoff."""
    aggregate, split = _write_synthetic_sources(tmp_path / "synthetic-polar")
    split_document = json.loads(split.read_text(encoding="utf-8"))
    capability_lock = _capability_fixture_module()._write_bundle(
        tmp_path / "test-only-capability",
        formal_eligible=False,
        test_only=True,
        proxy=True,
    )
    capability_bundle = load_frozen_capability_bundle(
        capability_lock,
        run_kind="development-smoke",
    )
    wheeled = capability_bundle.for_platform("WHEELED")
    inputs = _synthetic_polar_inputs(
        tmp_path / "synthetic-polar/raw/synthetic-dem.tif",
        split_document,
    )
    request = _synthetic_bridge_request(inputs, wheeled)
    canvas = inputs.world.canvas
    assert request.request_id == (
        f"polar-smoke/{canvas.identity}/{wheeled.content_sha256}"
    )
    assert request.world.global_map.frame_id == "map"
    assert request.world.local_map.frame_id == "odom"
    assert (
        request.world.map_from_odom.parent_frame
        == request.world.global_map.frame_id
    )
    assert (
        request.world.map_from_odom.child_frame
        == request.world.local_map.frame_id
    )
    assert request.world.local_map.origin_m.x == canvas.bounds_m[0]
    assert request.world.local_map.origin_m.y == canvas.bounds_m[1]
    assert request.world.local_map.resolution_m == canvas.geometry.resolution_m
    assert request.world.local_map.width == request.world.local_map.height == 256
    assert request.capability.maximum_slope_rad == (
        wheeled.typed_capability.maximum_slope_rad
    )
    assert tuple(
        item.primitive_id for item in request.capability.motion_primitives
    ) == wheeled.source_motion_primitive_ids

    def north_up_request_layer(name: str) -> np.ndarray:
        layer = request.world.local_map.layers[name].values.reshape(256, 256)
        return np.ascontiguousarray(np.flipud(layer))

    expected_elevation = np.where(
        inputs.world.observed_mask,
        inputs.world.elevation_m,
        0.0,
    ).astype(np.float32)
    request_elevation = north_up_request_layer("elevation")
    request_observed = north_up_request_layer("valid_mask").astype(bool)
    request_obstacle = north_up_request_layer("obstacle").astype(bool)
    request_forbidden = north_up_request_layer("forbidden").astype(bool)
    global_elevation = np.ascontiguousarray(
        np.flipud(
            request.world.global_map.layers["elevation"].values.reshape(256, 256)
        )
    )
    np.testing.assert_array_equal(global_elevation, request_elevation)
    np.testing.assert_allclose(request_elevation, expected_elevation)
    np.testing.assert_array_equal(request_observed, inputs.world.observed_mask)
    np.testing.assert_array_equal(
        request_obstacle,
        (inputs.world.physical_obstacle_layer.values > 0.0)
        & inputs.world.observed_mask,
    )
    np.testing.assert_array_equal(
        request_forbidden,
        (inputs.forbidden_ratio > 0.0) & inputs.world.observed_mask,
    )

    bridge = PlannerBridge()
    cpp_projection = bridge.project_traversability(request)
    assert cpp_projection.platform_type == "WHEELED"
    assert cpp_projection.known.dtype == np.uint8
    assert cpp_projection.hard_feasible.dtype == np.uint8
    assert cpp_projection.traversal_cost.dtype == np.float32
    assert cpp_projection.known.shape == (256, 256)
    projection = _platform_projection_from_cpp(
        inputs,
        cpp_projection,
        capability_sha256=wheeled.content_sha256,
        sensor_range_m=wheeled.observation_capability.sensor_range_m,
    )
    np.testing.assert_array_equal(
        projection.traversable_ratio > 0.0,
        np.flipud(cpp_projection.known).astype(bool)
        & np.flipud(cpp_projection.hard_feasible).astype(bool),
    )
    np.testing.assert_array_equal(
        projection.local_traversable_ratio,
        _sample_local(canvas, inputs.pose, projection.traversable_ratio),
    )
    assert projection.source == f"cpp_v3/{wheeled.content_sha256}"
    assert not np.all(projection.traversable_ratio == 1.0)
    assert np.isfinite(projection.clearance_margin_norm).all()
    assert (projection.clearance_margin_norm[~request_observed] == 0.0).all()
    known_obstacles = request_observed & request_obstacle
    assert known_obstacles.any()
    assert not projection.traversable_ratio[known_obstacles].any()

    batch, candidates = _policy_batch_from_projection(inputs, projection)
    assert request_elevation.shape == tuple(batch.prior_channels.shape[-2:])
    np.testing.assert_allclose(
        request_elevation,
        batch.prior_channels[0, 0].numpy(),
    )
    np.testing.assert_array_equal(
        request_obstacle,
        batch.prior_channels[0, 2].numpy() > 0.0,
    )
    np.testing.assert_array_equal(
        projection.traversable_ratio,
        batch.prior_channels[0, 3].numpy(),
    )

    # Close one same-world sensor loop before policy selection.  The initial
    # reveal is deliberately reward-free; the next reveal is driven only by
    # the endpoint of a reference certified by the current C++ v3 planner.
    feasible = projection.traversable_ratio > 0.0
    feasible_cells = [
        (row, column)
        for row in range(2, feasible.shape[0] - 2)
        for column in range(112, 121)
        if feasible[row - 1 : row + 2, column : column + 8].all()
    ]
    assert feasible_cells
    sensor_row, sensor_column = min(
        feasible_cells,
        key=lambda cell: (math.dist(cell, (128, 120)), cell),
    )
    sensor_x, sensor_y = canvas.grid_center_world(sensor_row, sensor_column)
    sensor_pose = Pose2(
        sensor_x,
        sensor_y,
        elevation_m=float(inputs.world.elevation_m[sensor_row, sensor_column]),
    )
    initial_known = inputs.world.observed_mask.copy()
    initial_known[:, sensor_column + 6 :] = False
    zeros = np.zeros(initial_known.shape, dtype=np.float32)
    sensor_state = SensorObservationState(
        truth=TrainingWorldTruth(
            canvas,
            inputs.world.elevation_m,
            inputs.world.physical_obstacle_layer.values,
        ),
        observed=TrainingObservedGrid(
            canvas=canvas,
            elevation_m=np.where(
                initial_known, inputs.world.elevation_m, 0.0
            ).astype(np.float32),
            physical_obstacle_ratio=np.where(
                initial_known,
                inputs.world.physical_obstacle_layer.values,
                0.0,
            ).astype(np.float32),
            valid_mask=initial_known,
            observation_age_s=zeros,
            observation_quality=initial_known.astype(np.float32),
            elevation_variance=zeros,
            obstacle_variance=zeros,
            observation_count=initial_known.astype(np.uint32),
        ),
        mission_roi_ratio=inputs.mission.roi_ratio,
        mission_priority=inputs.mission.priority,
        forbidden_mask=inputs.forbidden_ratio > 0.0,
        visibility_estimator=NativeVisibilityEstimator(
            SensorGeometry(
                FORMAL_SENSOR_RANGE_M,
                FORMAL_SENSOR_FOV_RAD,
            ),
            resolution_m=canvas.geometry.resolution_m,
        ),
    )
    boundary_artifacts: dict[str, object] = {}

    def current_global_map(
        current_inputs: _SyntheticPolarInputs,
    ) -> bridge_api.GridMap:
        scale = 4
        observed = np.repeat(
            np.repeat(current_inputs.world.observed_mask, scale, axis=0),
            scale,
            axis=1,
        )
        elevation = np.repeat(
            np.repeat(current_inputs.world.elevation_m, scale, axis=0),
            scale,
            axis=1,
        )
        physical = np.repeat(
            np.repeat(
                current_inputs.world.physical_obstacle_layer.values,
                scale,
                axis=0,
            ),
            scale,
            axis=1,
        )
        forbidden = np.repeat(
            np.repeat(current_inputs.forbidden_ratio, scale, axis=0),
            scale,
            axis=1,
        )

        def layer(values: np.ndarray, dtype) -> bridge_api.GridLayer:
            south_up = np.ascontiguousarray(
                np.flipud(np.asarray(values)).astype(dtype, copy=False).reshape(-1)
            )
            return bridge_api.GridLayer(south_up)

        grid = bridge_api.GridMap()
        grid.frame_id = "map"
        grid.stamp.nanoseconds_since_epoch = 1_000_000_000
        grid.width = observed.shape[1]
        grid.height = observed.shape[0]
        grid.resolution_m = 1.0
        grid.origin_m = _bridge_vec3(
            canvas.bounds_m[0], canvas.bounds_m[1], 0.0
        )
        zeros_global = np.zeros(observed.shape, dtype=np.float32)
        grid.layers = {
            "elevation": layer(
                np.where(observed, elevation, 0.0), np.float32
            ),
            "valid_mask": layer(observed, np.uint8),
            "obstacle": layer(observed & (physical > 0.0), np.uint8),
            "obstacle_height": layer(
                np.where(observed, physical * 0.5, 0.0), np.float32
            ),
            "observation_age_s": layer(zeros_global, np.float32),
            "observation_quality": layer(observed, np.float32),
            "elevation_variance": layer(zeros_global, np.float32),
            "obstacle_variance": layer(zeros_global, np.float32),
            "observation_count": layer(observed, np.uint32),
            "forbidden": layer(observed & (forbidden > 0.0), np.uint8),
        }
        return grid

    def current_local_map(
        current_inputs: _SyntheticPolarInputs,
    ) -> bridge_api.GridMap:
        local_cells = 300
        local_resolution_m = 0.25
        pose = current_inputs.pose
        origin_x = pose.x_m - (local_cells // 2 + 0.5) * local_resolution_m
        origin_y = pose.y_m - (local_cells // 2 + 0.5) * local_resolution_m
        x = origin_x + (np.arange(local_cells) + 0.5) * local_resolution_m
        y = (
            origin_y
            + local_cells * local_resolution_m
            - (np.arange(local_cells) + 0.5) * local_resolution_m
        )
        left, _, _, top = canvas.bounds_m
        columns = np.floor(
            (x - left) / canvas.geometry.resolution_m
        ).astype(np.intp)
        rows = np.floor(
            (top - y) / canvas.geometry.resolution_m
        ).astype(np.intp)
        assert rows.min() >= 0 and rows.max() < canvas.geometry.cells
        assert columns.min() >= 0 and columns.max() < canvas.geometry.cells
        observed = current_inputs.world.observed_mask[np.ix_(rows, columns)]
        elevation = current_inputs.world.elevation_m[np.ix_(rows, columns)]
        physical = current_inputs.world.physical_obstacle_layer.values[
            np.ix_(rows, columns)
        ]
        forbidden = current_inputs.forbidden_ratio[np.ix_(rows, columns)]

        def layer(values: np.ndarray, dtype) -> bridge_api.GridLayer:
            south_up = np.ascontiguousarray(
                np.flipud(np.asarray(values)).astype(dtype, copy=False).reshape(-1)
            )
            return bridge_api.GridLayer(south_up)

        grid = bridge_api.GridMap()
        grid.frame_id = "odom"
        grid.stamp.nanoseconds_since_epoch = 1_000_000_000
        grid.width = local_cells
        grid.height = local_cells
        grid.resolution_m = local_resolution_m
        grid.origin_m = _bridge_vec3(origin_x, origin_y, 0.0)
        zeros_local = np.zeros(observed.shape, dtype=np.float32)
        grid.layers = {
            "elevation": layer(
                np.where(observed, elevation, 0.0), np.float32
            ),
            "valid_mask": layer(observed, np.uint8),
            "obstacle": layer(observed & (physical > 0.0), np.uint8),
            "obstacle_height": layer(
                np.where(observed, physical * 0.5, 0.0), np.float32
            ),
            "observation_age_s": layer(zeros_local, np.float32),
            "observation_quality": layer(observed, np.float32),
            "elevation_variance": layer(zeros_local, np.float32),
            "obstacle_variance": layer(zeros_local, np.float32),
            "observation_count": layer(observed, np.uint32),
            "forbidden": layer(observed & (forbidden > 0.0), np.uint8),
        }
        return grid

    def add_current_wheel_smoke_primitives(
        current_request: bridge_api.TrainingPlanRequest,
    ) -> None:
        capability = current_request.capability
        capability.maximum_acceleration_mps2 = 0.5
        capability.maximum_braking_deceleration_mps2 = 0.5
        capability.maximum_yaw_acceleration_radps2 = 0.5
        capability.maximum_lateral_acceleration_mps2 = 0.5
        primitives = []
        arc_yaw = np.pi / 16.0
        arc_x = math.sin(arc_yaw)
        arc_y = 1.0 - math.cos(arc_yaw)
        for primitive_id, kind, translation_x, translation_y, yaw in (
            (
                "current-forward",
                bridge_api.WheelPrimitiveKind.FORWARD,
                0.25,
                0.0,
                0.0,
            ),
            (
                "current-reverse",
                bridge_api.WheelPrimitiveKind.REVERSE,
                -0.25,
                0.0,
                0.0,
            ),
            (
                "current-forward-arc-left",
                bridge_api.WheelPrimitiveKind.FORWARD_ARC,
                arc_x,
                arc_y,
                arc_yaw,
            ),
            (
                "current-forward-arc-right",
                bridge_api.WheelPrimitiveKind.FORWARD_ARC,
                arc_x,
                -arc_y,
                -arc_yaw,
            ),
            (
                "current-reverse-arc-left",
                bridge_api.WheelPrimitiveKind.REVERSE_ARC,
                -arc_x,
                -arc_y,
                arc_yaw,
            ),
            (
                "current-reverse-arc-right",
                bridge_api.WheelPrimitiveKind.REVERSE_ARC,
                -arc_x,
                arc_y,
                -arc_yaw,
            ),
            (
                "current-spin-left",
                bridge_api.WheelPrimitiveKind.SPIN_COUNTERCLOCKWISE,
                0.0,
                0.0,
                np.pi / 8.0,
            ),
            (
                "current-spin-right",
                bridge_api.WheelPrimitiveKind.SPIN_CLOCKWISE,
                0.0,
                0.0,
                -np.pi / 8.0,
            ),
            (
                "current-stop-switch",
                bridge_api.WheelPrimitiveKind.STOP_AND_SWITCH,
                0.0,
                0.0,
                0.0,
            ),
        ):
            primitive = bridge_api.WheelMotionPrimitive()
            primitive.primitive_id = primitive_id
            primitive.kind = kind
            pose = bridge_api.Pose3()
            pose.position_m.x = translation_x
            pose.position_m.y = translation_y
            pose.orientation.w = math.cos(yaw / 2.0)
            pose.orientation.z = math.sin(yaw / 2.0)
            primitive.relative_end_pose = pose
            primitives.append(primitive)
        capability.motion_primitives = primitives
        current_request.capability = capability
        current_request.config.wheel.xy_resolution_m = 0.25
        current_request.config.wheel.yaw_bin_count = 64

    def build_boundary_policy(
        observed: TrainingObservedGrid, pose: Pose2
    ) -> PolicyBatch:
        local = LocalObservation(
            canvas_id=canvas.identity,
            bounds_m=(
                pose.x_m - LOCAL_GEOMETRY.size_m / 2.0,
                pose.y_m - LOCAL_GEOMETRY.size_m / 2.0,
                pose.x_m + LOCAL_GEOMETRY.size_m / 2.0,
                pose.y_m + LOCAL_GEOMETRY.size_m / 2.0,
            ),
            elevation_m=_sample_local(canvas, pose, observed.elevation_m),
            observed_mask=_sample_local(canvas, pose, observed.valid_mask),
            physical_obstacle_ratio=_sample_local(
                canvas, pose, observed.physical_obstacle_ratio
            ),
        )
        world = observed.to_observed_world(local=local)
        current_inputs = _SyntheticPolarInputs(
            world=world,
            mission=inputs.mission,
            pose=pose,
            forbidden_ratio=inputs.forbidden_ratio,
        )
        current_request = _synthetic_bridge_request(current_inputs, wheeled)
        add_current_wheel_smoke_primitives(current_request)
        current_cpp_projection = bridge.project_traversability(current_request)
        current_projection = _platform_projection_from_cpp(
            current_inputs,
            current_cpp_projection,
            capability_sha256=wheeled.content_sha256,
            sensor_range_m=wheeled.observation_capability.sensor_range_m,
        )
        current_request.world.global_map = current_global_map(current_inputs)
        current_request.world.local_map = current_local_map(current_inputs)
        current_request.config.global_map.base_resolution_m = 0.25
        current_batch, current_candidates = _policy_batch_from_projection(
            current_inputs, current_projection
        )
        boundary_artifacts.update(
            inputs=current_inputs,
            request=current_request,
            projection=current_projection,
            candidates=current_candidates,
        )
        return current_batch

    controller = ObservationBoundaryController(
        platform_type="WHEELED",
        sensor_state=sensor_state,
        policy_observation_builder=build_boundary_policy,
        episode_id="polar-sensor-smoke",
        mission_revision=1,
    )
    initial_boundary = controller.reset(sensor_pose)
    assert initial_boundary.mission_observed_delta == 0.0
    assert initial_boundary.priority_observed_delta == 0.0
    assert np.count_nonzero(sensor_state.observed.valid_mask) > np.count_nonzero(
        initial_known
    )
    current_inputs = boundary_artifacts["inputs"]
    current_request = boundary_artifacts["request"]
    current_projection = boundary_artifacts["projection"]
    current_candidates = boundary_artifacts["candidates"]
    assert isinstance(current_inputs, _SyntheticPolarInputs)
    assert isinstance(current_request, bridge_api.TrainingPlanRequest)
    assert isinstance(current_projection, PlatformProjection)
    inputs = current_inputs
    request = current_request
    projection = current_projection
    candidates = current_candidates
    batch = initial_boundary.next_observation
    policy = CrossAttentionPolicy()
    rollout, sampled = _one_row_rollout(batch, policy)

    selected = int(sampled.selected_frontier_index.item())
    selected_features = candidates.features[selected]
    left, _, _, top = canvas.bounds_m
    selected_x = left + float(selected_features[0]) * canvas.geometry.size_m
    selected_y = top - float(selected_features[1]) * canvas.geometry.size_m
    request.goal.target.position_m.x = selected_x
    request.goal.target.position_m.y = selected_y
    apply_goal_theta(
        request.goal,
        "WHEELED",
        float(sampled.selected_theta.item()),
    )
    request.goal.goal_id = f"polar-frontier/{canvas.identity}/{selected}"
    assert request.goal.target.position_m.x == pytest.approx(selected_x)
    assert request.goal.target.position_m.y == pytest.approx(selected_y)
    assert request.goal.goal_id.endswith(f"/{selected}")
    planner_output = bridge.plan(request)
    assert planner_output.diagnostics.planner_name == "cpp_v3_hierarchical"
    assert planner_output.outcome != PlanningOutcome.INVALID_REQUEST, (
        planner_output.reason_code
    )
    assert planner_output.outcome in {
        PlanningOutcome.NEW_REFERENCE_AVAILABLE,
        PlanningOutcome.SAFE_FRONTIER_REFERENCE_AVAILABLE,
        PlanningOutcome.GOAL_INFEASIBLE,
        PlanningOutcome.NO_KNOWN_SAFE_ROUTE,
        PlanningOutcome.RESOURCE_EXHAUSTED,
        PlanningOutcome.ACTIVE_REFERENCE_INVALIDATED,
        PlanningOutcome.NUMERICAL_FAILURE,
        PlanningOutcome.CANCELED,
    }
    assert planner_output.reason_code
    assert planner_output.outcome == PlanningOutcome.NEW_REFERENCE_AVAILABLE, (
        planner_output.reason_code
    )
    assert planner_output.reference is not None
    known_before_execution = int(
        np.count_nonzero(sensor_state.observed.valid_mask)
    )
    boundary = controller.after_execution(
        platform_type="WHEELED",
        execution_state="DECISION_BOUNDARY",
        evidence=SensorBoundaryEvidence(
            Pose2(
                selected_x,
                selected_y,
                float(sampled.selected_theta.item()),
                elevation_m=inputs.pose.elevation_m,
            ),
            planner_output.diagnostics.elapsed.total_seconds(),
        ),
    )
    newly_observed = (
        int(np.count_nonzero(sensor_state.observed.valid_mask))
        - known_before_execution
    )
    assert boundary.mission_observed_delta == pytest.approx(
        newly_observed / float(np.sum(inputs.mission.roi_ratio))
    )
    assert boundary.priority_observed_delta == pytest.approx(
        newly_observed
        / float(np.sum(inputs.mission.priority * inputs.mission.roi_ratio))
    )

    config = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_smoke.yaml"
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    digest = lambda value: hashlib.sha256(value).hexdigest()
    identity = RunIdentity(
        run_kind="development-smoke",
        data_sha256=digest(aggregate.read_bytes()),
        split_sha256=split_document["split_sha256"],
        generator_sha256=digest(GENERATOR_VERSION.encode("utf-8")),
        capability_sha256=capability_bundle.bundle_sha256,
        reward_sha256=reward_weights_sha256(),
        v3_sha256=digest(f"lunar-planner-v3-source:{source_commit}".encode("utf-8")),
        training_semantics_sha256=training_semantics_sha256(),
    )
    trainer = PPOTrainer(policy, config=config.ppo, device="cpu")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer,
        lr_lambda=lambda step: 1.0,
    )
    first_metrics = trainer.update(rollout, micro_batch_size=1)
    scheduler.step()
    assert first_metrics.optimizer_steps >= 1

    artifacts = tmp_path / "development-artifacts"
    checkpoints = artifacts / "checkpoints"
    checkpoints.mkdir(parents=True)
    manifest_path = artifacts / "run-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "runtime_calibration": {"development_smoke": True},
                "total_gpu_budget_seconds": 86400,
                "consumed_gpu_seconds": 0.0,
                "budget_extension_blocks": 0,
            }
        ),
        encoding="utf-8",
    )
    frozen_config = config.as_frozen_dict()
    checkpoint_path = checkpoints / "latest.pt"
    first = build_training_checkpoint(
        model=trainer.policy,
        optimizer=trainer.optimizer,
        scheduler=scheduler,
        global_step=1,
        curriculum_phase="development-smoke",
        normalization={"reward_mean": 0.0, "reward_var": 1.0},
        frozen_config=frozen_config,
        run_identity=identity,
        source_commit=source_commit,
        consumed_gpu_seconds=0.0,
        budget_extension_blocks=0,
        total_gpu_budget_seconds=86400,
        worker_allocation={"WHEELED": 18},
        micro_batch_size=1,
        latest_checkpoint_gpu_seconds=0.0,
        candidate_checkpoint_gpu_seconds=0.0,
    )
    save_checkpoint_atomic(checkpoint_path, first)
    _update_run_manifest(
        manifest_path,
        source_commit=source_commit,
        config_hash=config_sha256(frozen_config),
        run_identity=identity,
        global_step=1,
        consumed_gpu_seconds=0.0,
        platform_allocation={"WHEELED": 18},
    )

    resumed = load_checkpoint_for_resume(
        checkpoint_path,
        expected_contract_version=first.contract_version,
        expected_config_hash=first.config_hash,
        expected_source_commit=source_commit,
        expected_run_identity=identity,
        expected_worker_allocation={"WHEELED": 18},
        expected_micro_batch_size=1,
        expected_budget_extension_blocks=0,
        expected_total_gpu_budget_seconds=86400,
    )
    resumed_trainer = PPOTrainer(CrossAttentionPolicy(), config=config.ppo, device="cpu")
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        resumed_trainer.optimizer,
        lr_lambda=lambda step: 1.0,
    )
    normalization = {"reward_mean": 0.0, "reward_var": 1.0}
    restore_training_state(
        resumed,
        resumed_trainer.policy,
        resumed_trainer.optimizer,
        resumed_scheduler,
        normalization_state=normalization,
    )
    second_metrics = resumed_trainer.update(rollout, micro_batch_size=1)
    resumed_scheduler.step()
    assert second_metrics.optimizer_steps >= 1
    second = build_training_checkpoint(
        model=resumed_trainer.policy,
        optimizer=resumed_trainer.optimizer,
        scheduler=resumed_scheduler,
        global_step=2,
        curriculum_phase="development-smoke",
        normalization=normalization,
        frozen_config=frozen_config,
        run_identity=identity,
        source_commit=source_commit,
        consumed_gpu_seconds=0.0,
        budget_extension_blocks=0,
        total_gpu_budget_seconds=86400,
        worker_allocation={"WHEELED": 18},
        micro_batch_size=1,
        latest_checkpoint_gpu_seconds=0.0,
        candidate_checkpoint_gpu_seconds=0.0,
    )
    save_checkpoint_atomic(checkpoint_path, second)
    _update_run_manifest(
        manifest_path,
        source_commit=source_commit,
        config_hash=second.config_hash,
        run_identity=identity,
        global_step=2,
        consumed_gpu_seconds=0.0,
        platform_allocation={"WHEELED": 18},
    )

    final_checkpoint = load_checkpoint(checkpoint_path)
    final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert final_checkpoint.schema_version == "lunar-ppo-checkpoint/v4"
    assert final_checkpoint.global_step == final_manifest["global_step"] == 2
    assert final_checkpoint.run_identity.to_dict() == final_manifest["run_identity"]
    assert final_checkpoint.run_identity == identity
    assert not any(path.is_file() for path in REPOSITORY_ROOT.glob("**/*.pt"))


@pytest.mark.cuda
def test_cuda_interrupt_resume_preserves_step_budget_and_allocation(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a real CUDA pause/resume reset progress or the warmup split."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    artifact_root = tmp_path / "cuda-interrupt-resume"
    monkeypatch.setattr(
        training_cli,
        "_proxy_rollout",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("public train/resume must not use proxy rollout")
        ),
    )

    evidence = run_cuda_interrupt_resume_smoke(
        config_path=(
            REPOSITORY_ROOT / "training/configs/rtx4080_super_smoke.yaml"
        ),
        artifact_root=artifact_root,
        repository_root=REPOSITORY_ROOT,
    )

    assert evidence.device_name == "NVIDIA GeForce RTX 4080 SUPER"
    assert evidence.interrupted_global_step >= 1
    assert evidence.resumed_global_step == evidence.interrupted_global_step + 1
    assert (
        evidence.resumed_consumed_gpu_seconds
        > evidence.interrupted_consumed_gpu_seconds
        > 0.0
    )
    assert evidence.signal_observed_at_update_boundary is True
    assert (artifact_root / "checkpoints/latest.pt").is_file()
    manifest = json.loads(
        (artifact_root / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["global_step"] == evidence.resumed_global_step
    assert manifest["consumed_gpu_seconds"] == (
        evidence.resumed_consumed_gpu_seconds
    )
    assert manifest["frozen_config"]["parallel"]["joint_workers"] == {
        "WHEELED": 8,
        "LEGGED": 8,
        "HOPPER": 8,
    }
    selected_workers = manifest["runtime_calibration"]["selected_workers"]
    selected_micro_batch = manifest["runtime_calibration"][
        "selected_micro_batch"
    ]
    epochs_per_update = manifest["frozen_config"]["ppo"]["epochs_per_update"]
    assert all(
        measurement["optimizer_steps"] == epochs_per_update
        and measurement["ipc_failures"] == 0
        for measurement in manifest["runtime_calibration"]["measurements"]
    )
    assert evidence.platform_allocation == {"WHEELED": selected_workers}
    checkpoint = load_checkpoint(artifact_root / "checkpoints/latest.pt")
    assert checkpoint.schema_version == "lunar-ppo-checkpoint/v4"
    assert checkpoint.run_identity.run_kind == "development-smoke"
    assert manifest["run_identity"] == checkpoint.run_identity.to_dict()
    assert checkpoint.worker_allocation == evidence.platform_allocation
    assert checkpoint.curriculum_phase == "warmup_wheeled"
    assert checkpoint.micro_batch_size == selected_micro_batch
    assert checkpoint.latest_checkpoint_gpu_seconds == (
        checkpoint.consumed_gpu_seconds
    )

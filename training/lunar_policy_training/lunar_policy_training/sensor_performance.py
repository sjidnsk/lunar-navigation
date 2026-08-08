"""Strict Release evidence for the formal sensor-observation performance gate."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import socket
import subprocess
import tempfile
import time

from .training_semantics import training_semantics_sha256


SCHEMA_VERSION = "sensor-observation-performance/v1"
NATIVE_SCHEMA_VERSION = "native-visibility-benchmark/v1"
SCENARIO_SEED = 4080
REQUIRED_WORKERS = 24
NATIVE_WARMUP_COUNT = 50
NATIVE_SAMPLE_COUNT = 200
THROUGHPUT_WARMUP_STEPS = 2
THROUGHPUT_SAMPLE_STEPS = 10
CANDIDATE_P95_LIMIT_MS = 5.0
REVEAL_P95_LIMIT_MS = 2.0
THROUGHPUT_DROP_LIMIT_RATIO = 0.10

_CANDIDATE_FIXTURE = {
    "height": 256,
    "width": 256,
    "resolution_m": 4.0,
    "range_m": 30.0,
    "candidate_count": 64,
}
_REVEAL_FIXTURE = {
    "height": 320,
    "width": 320,
    "resolution_m": 0.2,
    "range_m": 30.0,
}
_PLANNER_WORKLOAD = "cpp-v3-current-capability-v2"
_SENSOR_SOURCE_PATHS = (
    "ros2_ws/src/lunar_planner_training_bridge",
    "training/lunar_policy_training/lunar_policy_training",
    "training/tools/benchmark_sensor_observation.py",
)


class SensorPerformanceError(ValueError):
    """Sensor benchmark evidence cannot authorize formal training."""


_THROUGHPUT_BARRIER = None
_THROUGHPUT_CAPABILITIES = None


def benchmark_observation_throughput(
    *,
    workers: int = REQUIRED_WORKERS,
    capability_bundle: object | None = None,
    diagnostic_only: bool = False,
) -> dict[str, float]:
    """Measure paired end-to-end sampling on the current planner/capability."""
    if type(workers) is not int or workers != REQUIRED_WORKERS:
        raise SensorPerformanceError("sensor throughput benchmark requires 24 workers")
    if diagnostic_only:
        if capability_bundle is not None:
            raise SensorPerformanceError(
                "diagnostic throughput must not consume a formal capability"
            )
        capabilities = None
    else:
        from .capability_freeze import FrozenCapabilityBundle

        if (
            not isinstance(capability_bundle, FrozenCapabilityBundle)
            or not capability_bundle.formal_eligible
            or len(capability_bundle.platforms) != 3
        ):
            raise SensorPerformanceError(
                "formal throughput requires the current formal capability bundle"
            )
        capabilities = capability_bundle.platforms
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(workers, timeout=180.0)
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_throughput_worker,
            initargs=(barrier, capabilities),
        ) as executor:
            results = tuple(executor.map(_throughput_worker, range(workers)))
    except Exception as error:
        raise SensorPerformanceError(
            "24-worker sensor throughput benchmark failed"
        ) from error
    disabled_elapsed = max(result[0] for result in results)
    enabled_elapsed = max(result[1] for result in results)
    if any(result[2] <= 0 for result in results):
        raise SensorPerformanceError("enabled sensor workload produced no observations")
    if capabilities is not None and any(result[3] != 2 for result in results):
        raise SensorPerformanceError(
            "current C++ v3 planner did not complete both throughput phases"
        )
    sample_count = workers * THROUGHPUT_SAMPLE_STEPS
    return {
        "observation_disabled": sample_count / disabled_elapsed,
        "observation_enabled": sample_count / enabled_elapsed,
    }


def _initialize_throughput_worker(barrier, capabilities) -> None:
    global _THROUGHPUT_BARRIER, _THROUGHPUT_CAPABILITIES
    _THROUGHPUT_BARRIER = barrier
    _THROUGHPUT_CAPABILITIES = capabilities
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _throughput_worker(worker_index: int) -> tuple[float, float, int, int]:
    if _THROUGHPUT_BARRIER is None:
        raise RuntimeError("sensor throughput worker barrier is missing")
    platform_type = ("WHEELED", "LEGGED", "HOPPER")[worker_index % 3]
    disabled = _performance_boundary_controller(
        worker_index=worker_index,
        observation_enabled=False,
        platform_type=platform_type,
    )
    enabled = _performance_boundary_controller(
        worker_index=worker_index,
        observation_enabled=True,
        platform_type=platform_type,
    )
    poses = _performance_poses(disabled.sensor_state.truth.canvas)
    planners = _performance_planner_pair(worker_index)
    disabled.reset(poses[0])
    enabled.reset(poses[0])
    for step in range(THROUGHPUT_WARMUP_STEPS):
        pose = poses[step + 1]
        if planners is not None:
            _performance_plan_step(planners[0], step)
            _performance_plan_step(planners[1], step)
        _performance_boundary_step(disabled, pose)
        _performance_boundary_step(enabled, pose)

    _THROUGHPUT_BARRIER.wait()
    start = time.perf_counter()
    for step in range(THROUGHPUT_SAMPLE_STEPS):
        if planners is not None:
            _performance_plan_step(planners[0], step)
        _performance_boundary_step(
            disabled, poses[THROUGHPUT_WARMUP_STEPS + 1 + step]
        )
    disabled_elapsed = time.perf_counter() - start

    _THROUGHPUT_BARRIER.wait()
    start = time.perf_counter()
    for step in range(THROUGHPUT_SAMPLE_STEPS):
        if planners is not None:
            _performance_plan_step(planners[1], step)
        _performance_boundary_step(
            enabled, poses[THROUGHPUT_WARMUP_STEPS + 1 + step]
        )
    enabled_elapsed = time.perf_counter() - start
    observed_cells = int(enabled.sensor_state.observed.valid_mask.sum())
    return (
        disabled_elapsed,
        enabled_elapsed,
        observed_cells,
        0 if planners is None else 2,
    )


def _performance_planner_pair(worker_index: int):
    if _THROUGHPUT_CAPABILITIES is None:
        return None
    platform = ("WHEELED", "LEGGED", "HOPPER")[worker_index % 3]
    capability = _THROUGHPUT_CAPABILITIES[worker_index % 3]
    if capability.platform_type != platform:
        raise RuntimeError("formal capability platform order is invalid")
    return (
        _performance_planner_workload(capability, f"disabled-{worker_index}"),
        _performance_planner_workload(capability, f"enabled-{worker_index}"),
    )


def _performance_plan_step(workload, step: int) -> None:
    import lunar_planner_training_bridge as bridge_api

    planner, request = workload
    request.request_id = f"sensor-performance-{step}"
    output = planner.plan(request)
    if output.outcome != bridge_api.PlanningOutcome.NEW_REFERENCE_AVAILABLE:
        raise RuntimeError(
            f"current planner workload failed: {output.reason_code}"
        )


def _performance_planner_workload(capability, suffix: str):
    import numpy as np
    import lunar_planner_training_bridge as bridge_api

    from .policy.action_semantics import apply_goal_theta

    platform = capability.platform_type
    request = bridge_api.TrainingPlanRequest()
    request.request_id = f"sensor-performance-{suffix}"
    request.mission_id = "sensor-performance"
    request.mission_revision = 1
    request.platform_id = capability.platform_id
    request.capability_version = capability.capability_version
    request.global_map_generation = 1
    request.local_map_generation = 1
    request.map_from_odom_generation = 1
    request.state_time.nanoseconds_since_epoch = 1_000_000_000

    if platform == "HOPPER":
        start = (32.1, 32.1, 0.5)
        target = (40.1, 32.1, 0.0)
        state = bridge_api.HopperState()
        state.pose = _performance_pose(bridge_api, *start)
        request.current_state = state
    else:
        start = (32.1, 32.1, 0.0)
        target = (40.1, 32.1, 0.0)
        if platform == "WHEELED":
            state = bridge_api.WheeledState()
            state.pose = _performance_pose(bridge_api, *start)
        else:
            state = bridge_api.LeggedState()
            state.body_pose = _performance_pose(bridge_api, 32.1, 32.1, 0.33)
        request.current_state = state
    request.capability = capability.to_bridge_capability()
    point = bridge_api.PointGoal()
    point.position_m = _performance_vec3(bridge_api, *target)
    point.tolerance_m = 0.0 if platform == "HOPPER" else 0.2
    request.goal.goal_id = f"sensor-performance-{platform.lower()}"
    request.goal.target = point
    apply_goal_theta(request.goal, platform, 0.0)
    request.world.global_map = _performance_grid_map(
        bridge_api, np, "map", 640, 640, 1.6, origin_m=-0.7
    )
    request.world.local_map = _performance_grid_map(
        bridge_api, np, "odom", 320, 320, 0.2
    )
    request.world.map_from_odom.parent_frame = "map"
    request.world.map_from_odom.child_frame = "odom"
    request.world.map_from_odom.stamp.nanoseconds_since_epoch = 1_000_000_000
    request.config.global_map.base_resolution_m = 0.2
    request.config.wheel.xy_resolution_m = 0.2
    request.config.wheel.yaw_bin_count = 64
    request.config.legged.xy_resolution_m = 0.2
    request.config.legged.yaw_bin_count = 64
    return bridge_api.PlannerBridge(), request


def _performance_vec3(bridge_api, x: float, y: float, z: float):
    value = bridge_api.Vec3()
    value.x = x
    value.y = y
    value.z = z
    return value


def _performance_pose(bridge_api, x: float, y: float, z: float):
    value = bridge_api.Pose3()
    value.position_m = _performance_vec3(bridge_api, x, y, z)
    orientation = bridge_api.Quaternion()
    orientation.w = 1.0
    value.orientation = orientation
    return value


def _performance_grid_map(
    bridge_api,
    np,
    frame_id: str,
    width: int,
    height: int,
    resolution: float,
    *,
    origin_m: float = 0.0,
):
    count = width * height
    zeros_f32 = np.zeros(count, dtype=np.float32)
    value = bridge_api.GridMap()
    value.frame_id = frame_id
    value.stamp.nanoseconds_since_epoch = 1_000_000_000
    value.width = width
    value.height = height
    value.resolution_m = resolution
    origin = bridge_api.Vec3()
    origin.x = origin_m
    origin.y = origin_m
    value.origin_m = origin
    value.layers = {
        "elevation": bridge_api.GridLayer(zeros_f32),
        "valid_mask": bridge_api.GridLayer(np.ones(count, dtype=np.uint8)),
        "obstacle": bridge_api.GridLayer(np.zeros(count, dtype=np.uint8)),
        "obstacle_height": bridge_api.GridLayer(zeros_f32),
        "observation_age_s": bridge_api.GridLayer(zeros_f32),
        "observation_quality": bridge_api.GridLayer(
            np.ones(count, dtype=np.float32)
        ),
        "elevation_variance": bridge_api.GridLayer(zeros_f32),
        "obstacle_variance": bridge_api.GridLayer(zeros_f32),
        "observation_count": bridge_api.GridLayer(
            np.ones(count, dtype=np.uint32)
        ),
        "forbidden": bridge_api.GridLayer(np.zeros(count, dtype=np.uint8)),
    }
    return value


def _performance_boundary_step(controller, pose) -> None:
    from .environment.observation_boundary import SensorBoundaryEvidence

    controller.after_execution(
        platform_type=controller.platform_type,
        execution_state=(
            "LANDED_HOLD"
            if controller.platform_type == "HOPPER"
            else "DECISION_BOUNDARY"
        ),
        evidence=SensorBoundaryEvidence(pose, 1.0),
    )


def _performance_poses(canvas) -> tuple[object, ...]:
    from .environment.observation_builder import Pose2

    cells = (
        (154, 154),
        (154, 158),
        (154, 162),
        (154, 166),
        (158, 166),
        (162, 166),
        (166, 166),
        (166, 162),
        (166, 158),
        (166, 154),
        (162, 154),
        (158, 154),
        (158, 158),
    )
    return tuple(
        Pose2(*canvas.grid_center_world(row, column)) for row, column in cells
    )


def _performance_boundary_controller(
    *, worker_index: int, observation_enabled: bool, platform_type: str
):
    import numpy as np
    import torch

    from .environment.observation_boundary import ObservationBoundaryController
    from .environment.sensor_observation import (
        SensorObservationState,
        TrainingObservedGrid,
        TrainingWorldTruth,
    )
    from .environment.visibility import NativeVisibilityEstimator, SensorGeometry
    from .polar_data.raster import GridGeometry, MapCanvas
    from .policy.observation import PolicyBatch

    geometry = GridGeometry(size_m=64.0, resolution_m=0.2, cells=320)
    canvas = MapCanvas("4" * 64, (0.0, 0.0, 64.0, 64.0), geometry)
    random = np.random.default_rng(SCENARIO_SEED)
    obstacle = np.ascontiguousarray(
        (random.random((320, 320)) < (1.0 / 127.0)).astype(np.float32)
    )
    truth = TrainingWorldTruth(
        canvas,
        np.zeros((320, 320), dtype=np.float32),
        obstacle,
    )
    sensor = SensorGeometry(30.0, 2.0 * math.pi)

    if observation_enabled:
        estimator = NativeVisibilityEstimator(sensor, resolution_m=0.2)
    else:
        class NoRevealEstimator:
            def __init__(self) -> None:
                self.sensor = sensor
                self.resolution_m = 0.2

            def reveal_from_pose(self, truth_obstacle_ratio, pose_cell):
                return np.zeros(truth_obstacle_ratio.shape, dtype=np.bool_)

        estimator = NoRevealEstimator()

    sensor_state = SensorObservationState(
        truth=truth,
        observed=TrainingObservedGrid.empty(canvas),
        mission_roi_ratio=np.ones((320, 320), dtype=np.float32),
        mission_priority=np.full((320, 320), 0.5, dtype=np.float32),
        forbidden_mask=np.zeros((320, 320), dtype=np.bool_),
        visibility_estimator=estimator,
    )
    prior = torch.zeros((1, 4, 256, 256), dtype=torch.float32)
    coverage = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    local = torch.zeros((1, 4, 32, 32), dtype=torch.float32)
    frontier = torch.zeros((1, 64, 12), dtype=torch.float32)
    candidate_mask = torch.zeros((1, 64), dtype=torch.bool)
    candidate_mask[0, 0] = True
    platform_context = torch.zeros((1, 3), dtype=torch.float32)
    platform_context[
        0, {"WHEELED": 0, "LEGGED": 1, "HOPPER": 2}[platform_type]
    ] = 1.0

    def build_policy_observation(observed, pose_map):
        coverage_ratio = float(observed.valid_mask.mean(dtype=np.float64))
        coverage[0, 0].fill_(coverage_ratio)
        pose = torch.tensor(
            [[
                pose_map.x_m / 64.0,
                pose_map.y_m / 64.0,
                0.0,
                1.0,
                coverage_ratio,
                1.0,
            ]],
            dtype=torch.float32,
        )
        return PolicyBatch(
            prior,
            coverage,
            local,
            frontier,
            pose,
            candidate_mask,
            platform_context,
        )

    return ObservationBoundaryController(
        platform_type=platform_type,
        sensor_state=sensor_state,
        policy_observation_builder=build_policy_observation,
        episode_id=f"sensor-performance-{worker_index}",
        mission_revision=1,
    )


def current_host_identity() -> dict[str, str]:
    """Return the exact host identity used by the formal benchmark gate."""
    return {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def sensor_source_commit(repository_root: str | Path) -> str:
    """Return the newest commit touching sensor/training implementation sources."""
    root = Path(repository_root).resolve()
    try:
        completed = subprocess.run(
            [
                "git",
                "log",
                "-1",
                "--format=%H",
                "--",
                *_SENSOR_SOURCE_PATHS,
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SensorPerformanceError("sensor source commit is unavailable") from error
    value = completed.stdout.strip()
    if not _is_hex(value, 40):
        raise SensorPerformanceError("sensor source commit is invalid")
    return value


def require_clean_sensor_source(repository_root: str | Path) -> None:
    """Reject reports generated from uncommitted implementation bytes."""
    root = Path(repository_root).resolve()
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--", *_SENSOR_SOURCE_PATHS],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SensorPerformanceError("sensor source cleanliness is unavailable") from error
    if completed.stdout:
        raise SensorPerformanceError(
            "sensor source has uncommitted changes; commit before benchmarking"
        )


def build_sensor_performance_report(
    *,
    native_benchmark: Mapping[str, object],
    observation_disabled_steps_per_second: float,
    observation_enabled_steps_per_second: float,
    capability_sha256: str,
    source_commit: str,
    host: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Combine native latency and 24-worker throughput into one sealed report."""
    native = _validated_native_benchmark(native_benchmark)
    disabled = _positive_finite(
        observation_disabled_steps_per_second,
        "disabled observation throughput",
    )
    enabled = _positive_finite(
        observation_enabled_steps_per_second,
        "enabled observation throughput",
    )
    drop_ratio = max(0.0, 1.0 - enabled / disabled)
    candidate_p95 = float(native["latency_ms"]["candidate_gains"]["p95"])
    reveal_p95 = float(native["latency_ms"]["reveal"]["p95"])
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "host": dict(current_host_identity() if host is None else host),
        "build": {
            "type": native["build_type"],
            "compiler_id": native["compiler_id"],
            "compiler_version": native["compiler_version"],
        },
        "source_commit": source_commit,
        "capability_sha256": capability_sha256,
        "training_semantics_sha256": training_semantics_sha256(),
        "fixture": {
            "scenario_seed": SCENARIO_SEED,
            "workers": REQUIRED_WORKERS,
            "planner_workload": _PLANNER_WORKLOAD,
            "native_warmup_count": NATIVE_WARMUP_COUNT,
            "native_sample_count": NATIVE_SAMPLE_COUNT,
            "throughput_warmup_steps": THROUGHPUT_WARMUP_STEPS,
            "throughput_sample_steps": THROUGHPUT_SAMPLE_STEPS,
            "candidate": dict(_CANDIDATE_FIXTURE),
            "reveal": dict(_REVEAL_FIXTURE),
        },
        "latency_ms": {
            "candidate_gains": dict(native["latency_ms"]["candidate_gains"]),
            "reveal": dict(native["latency_ms"]["reveal"]),
        },
        "throughput_steps_per_second": {
            "observation_disabled": disabled,
            "observation_enabled": enabled,
        },
        "throughput_drop_ratio": drop_ratio,
        "thresholds": {
            "candidate_p95_ms_max": CANDIDATE_P95_LIMIT_MS,
            "reveal_p95_ms_max": REVEAL_P95_LIMIT_MS,
            "throughput_drop_ratio_max": THROUGHPUT_DROP_LIMIT_RATIO,
        },
        "passed": (
            native["build_type"] == "Release"
            and candidate_p95 <= CANDIDATE_P95_LIMIT_MS
            and reveal_p95 <= REVEAL_P95_LIMIT_MS
            and drop_ratio <= THROUGHPUT_DROP_LIMIT_RATIO
        ),
    }
    report["report_sha256"] = sensor_performance_sha256(report)
    return report


def sensor_performance_sha256(report: Mapping[str, object]) -> str:
    """Hash canonical report content while excluding its self-hash field."""
    if not isinstance(report, Mapping):
        raise SensorPerformanceError("sensor performance report must be an object")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    try:
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SensorPerformanceError(
            "sensor performance report must contain finite JSON values"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def validate_sensor_performance_report(
    report: Mapping[str, object],
    *,
    expected_host: Mapping[str, object],
    expected_source_commit: str,
    expected_capability_sha256: str,
    expected_training_semantics_sha256: str,
) -> str:
    """Validate exact identity, fixture and passing thresholds for formal use."""
    root = _exact_mapping(
        report,
        {
            "schema_version",
            "host",
            "build",
            "source_commit",
            "capability_sha256",
            "training_semantics_sha256",
            "fixture",
            "latency_ms",
            "throughput_steps_per_second",
            "throughput_drop_ratio",
            "thresholds",
            "passed",
            "report_sha256",
        },
        "sensor performance report fields",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise SensorPerformanceError("sensor performance report schema is invalid")
    host = _exact_mapping(
        root["host"],
        {"hostname", "system", "release", "machine"},
        "sensor performance host fields",
    )
    expected_host_clean = _exact_mapping(
        expected_host,
        {"hostname", "system", "release", "machine"},
        "expected sensor performance host fields",
    )
    if host != expected_host_clean or host["system"] != "Linux" or host["machine"] != "x86_64":
        raise SensorPerformanceError("sensor performance report host is stale")
    if any(not isinstance(value, str) or not value for value in host.values()):
        raise SensorPerformanceError("sensor performance report host is invalid")
    build = _exact_mapping(
        root["build"],
        {"type", "compiler_id", "compiler_version"},
        "sensor performance build fields",
    )
    if build["type"] != "Release":
        raise SensorPerformanceError("sensor performance report requires Release")
    if any(
        not isinstance(build[name], str) or not build[name]
        for name in ("compiler_id", "compiler_version")
    ):
        raise SensorPerformanceError("sensor performance compiler identity is invalid")
    _require_expected_hash(
        root["source_commit"], expected_source_commit, 40, "source commit"
    )
    _require_expected_hash(
        root["capability_sha256"],
        expected_capability_sha256,
        64,
        "capability hash",
    )
    _require_expected_hash(
        root["training_semantics_sha256"],
        expected_training_semantics_sha256,
        64,
        "training semantics hash",
    )
    _validate_fixture(root["fixture"])
    latency = _validate_latency(root["latency_ms"])
    throughput = _exact_mapping(
        root["throughput_steps_per_second"],
        {"observation_disabled", "observation_enabled"},
        "sensor throughput fields",
    )
    disabled = _positive_finite(
        throughput["observation_disabled"], "disabled observation throughput"
    )
    enabled = _positive_finite(
        throughput["observation_enabled"], "enabled observation throughput"
    )
    reported_drop = _finite_nonnegative(
        root["throughput_drop_ratio"], "throughput drop ratio"
    )
    calculated_drop = max(0.0, 1.0 - enabled / disabled)
    if not math.isclose(reported_drop, calculated_drop, rel_tol=0.0, abs_tol=1.0e-12):
        raise SensorPerformanceError("throughput drop ratio is inconsistent")
    thresholds = _exact_mapping(
        root["thresholds"],
        {
            "candidate_p95_ms_max",
            "reveal_p95_ms_max",
            "throughput_drop_ratio_max",
        },
        "sensor performance threshold fields",
    )
    expected_thresholds = {
        "candidate_p95_ms_max": CANDIDATE_P95_LIMIT_MS,
        "reveal_p95_ms_max": REVEAL_P95_LIMIT_MS,
        "throughput_drop_ratio_max": THROUGHPUT_DROP_LIMIT_RATIO,
    }
    if thresholds != expected_thresholds:
        raise SensorPerformanceError("sensor performance thresholds are stale")
    if latency["candidate_gains"]["p95"] > CANDIDATE_P95_LIMIT_MS:
        raise SensorPerformanceError("candidate visibility p95 exceeds 5 ms")
    if latency["reveal"]["p95"] > REVEAL_P95_LIMIT_MS:
        raise SensorPerformanceError("reveal p95 exceeds 2 ms")
    if reported_drop > THROUGHPUT_DROP_LIMIT_RATIO:
        raise SensorPerformanceError("throughput drop exceeds ten percent")
    if root["passed"] is not True:
        raise SensorPerformanceError("sensor performance report did not pass")
    reported_hash = root["report_sha256"]
    if not _is_hex(reported_hash, 64) or reported_hash != sensor_performance_sha256(root):
        raise SensorPerformanceError("sensor performance report hash is invalid")
    return reported_hash


def load_sensor_performance_report(path: str | Path) -> dict[str, object]:
    """Load one regular UTF-8 JSON report without accepting NaN constants."""
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise SensorPerformanceError("sensor performance report file is missing or unsafe")

    def reject_constant(value: str) -> object:
        raise ValueError(value)

    try:
        value = json.loads(
            target.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise SensorPerformanceError("sensor performance report JSON is invalid") from error
    if not isinstance(value, dict):
        raise SensorPerformanceError("sensor performance report must be an object")
    return value


def write_sensor_performance_report_atomic(
    path: str | Path, report: Mapping[str, object]
) -> str:
    """Atomically replace one outside-repository performance report."""
    target = Path(path)
    if target.is_symlink():
        raise SensorPerformanceError("sensor performance output must not be a symlink")
    if not target.parent.is_dir():
        raise SensorPerformanceError("sensor performance output parent is missing")
    digest = sensor_performance_sha256(report)
    if report.get("report_sha256") != digest:
        raise SensorPerformanceError("sensor performance report hash is invalid")
    try:
        contents = json.dumps(
            dict(report),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise SensorPerformanceError("sensor performance report JSON is invalid") from error
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        if isinstance(error, SensorPerformanceError):
            raise
        raise SensorPerformanceError(
            "sensor performance report could not be written atomically"
        ) from error
    return digest


def _validated_native_benchmark(
    value: Mapping[str, object],
) -> dict[str, object]:
    native = _exact_mapping(
        value,
        {
            "schema_version",
            "build_type",
            "compiler_id",
            "compiler_version",
            "warmup_count",
            "sample_count",
            "timing_unit",
            "candidate_fixture",
            "reveal_fixture",
            "latency_ms",
            "checksum",
        },
        "native visibility benchmark fields",
    )
    if (
        native["schema_version"] != NATIVE_SCHEMA_VERSION
        or native["build_type"] != "Release"
        or native["warmup_count"] != NATIVE_WARMUP_COUNT
        or native["sample_count"] != NATIVE_SAMPLE_COUNT
        or native["timing_unit"] != "ms"
        or native["candidate_fixture"] != _CANDIDATE_FIXTURE
        or native["reveal_fixture"] != _REVEAL_FIXTURE
        or type(native["checksum"]) is not int
        or native["checksum"] < 0
    ):
        raise SensorPerformanceError("native visibility benchmark fixture is invalid")
    if any(
        not isinstance(native[name], str) or not native[name]
        for name in ("compiler_id", "compiler_version")
    ):
        raise SensorPerformanceError("native visibility compiler identity is invalid")
    _validate_latency(native["latency_ms"])
    return native


def _validate_fixture(value: object) -> None:
    fixture = _exact_mapping(
        value,
        {
            "scenario_seed",
            "workers",
            "planner_workload",
            "native_warmup_count",
            "native_sample_count",
            "throughput_warmup_steps",
            "throughput_sample_steps",
            "candidate",
            "reveal",
        },
        "sensor performance fixture fields",
    )
    expected_scalars = {
        "scenario_seed": SCENARIO_SEED,
        "workers": REQUIRED_WORKERS,
        "planner_workload": _PLANNER_WORKLOAD,
        "native_warmup_count": NATIVE_WARMUP_COUNT,
        "native_sample_count": NATIVE_SAMPLE_COUNT,
        "throughput_warmup_steps": THROUGHPUT_WARMUP_STEPS,
        "throughput_sample_steps": THROUGHPUT_SAMPLE_STEPS,
    }
    if any(fixture[name] != expected for name, expected in expected_scalars.items()):
        raise SensorPerformanceError("sensor performance fixture is stale")
    if fixture["candidate"] != _CANDIDATE_FIXTURE or fixture["reveal"] != _REVEAL_FIXTURE:
        raise SensorPerformanceError("sensor performance fixture geometry is stale")


def _validate_latency(value: object) -> dict[str, dict[str, float]]:
    latency = _exact_mapping(
        value,
        {"candidate_gains", "reveal"},
        "sensor latency fields",
    )
    result: dict[str, dict[str, float]] = {}
    for name in ("candidate_gains", "reveal"):
        timing = _exact_mapping(
            latency[name], {"p50", "p95"}, f"{name} latency fields"
        )
        p50 = _finite_nonnegative(timing["p50"], f"{name} p50")
        p95 = _finite_nonnegative(timing["p95"], f"{name} p95")
        if p50 > p95:
            raise SensorPerformanceError(f"{name} p50 exceeds p95")
        result[name] = {"p50": p50, "p95": p95}
    return result


def _exact_mapping(
    value: object, fields: set[str], label: str
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields or len(value) != len(fields):
        raise SensorPerformanceError(f"{label} must match exactly")
    return dict(value)


def _finite_nonnegative(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise SensorPerformanceError(f"{label} must be finite and non-negative")
    return float(value)


def _positive_finite(value: object, label: str) -> float:
    parsed = _finite_nonnegative(value, label)
    if parsed <= 0.0:
        raise SensorPerformanceError(f"{label} must be positive")
    return parsed


def _require_expected_hash(
    value: object, expected: object, length: int, label: str
) -> None:
    if not _is_hex(value, length) or not _is_hex(expected, length):
        raise SensorPerformanceError(f"sensor performance {label} is invalid")
    if value != expected:
        raise SensorPerformanceError(f"sensor performance {label} is stale")


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "CANDIDATE_P95_LIMIT_MS",
    "NATIVE_SAMPLE_COUNT",
    "NATIVE_WARMUP_COUNT",
    "REQUIRED_WORKERS",
    "REVEAL_P95_LIMIT_MS",
    "SCHEMA_VERSION",
    "SensorPerformanceError",
    "THROUGHPUT_DROP_LIMIT_RATIO",
    "THROUGHPUT_SAMPLE_STEPS",
    "THROUGHPUT_WARMUP_STEPS",
    "benchmark_observation_throughput",
    "build_sensor_performance_report",
    "current_host_identity",
    "load_sensor_performance_report",
    "require_clean_sensor_source",
    "sensor_performance_sha256",
    "sensor_source_commit",
    "validate_sensor_performance_report",
    "write_sensor_performance_report_atomic",
]

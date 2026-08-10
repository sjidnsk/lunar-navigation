from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import math
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lunar_policy_training.capability_freeze import ScenarioIdentity
from lunar_policy_training.environment import formal_builder as formal_builder_module
from lunar_policy_training.environment import candidate_builder as candidate_builder_module
from lunar_policy_training.environment import formal_start_qualification as qualification_module
from lunar_policy_training.environment.candidate_builder import CandidateBuilderV2
from lunar_policy_training.environment.coverability import (
    IneligibleReason,
    PlatformCoverability,
    QualifiedStartState,
    mask_sha256,
    pack_detail_mask,
)
from lunar_policy_training.environment.formal_builder import (
    FormalEpisode,
    FormalEnvironmentBuilder,
    FormalWorkerBuilder,
    _aggregate_primitive_detail,
    _formal_schedule_index,
)
from lunar_policy_training.environment.formal_episode_state import (
    FormalWorkerState,
    policy_batch_sha256,
)
from lunar_policy_training.environment.formal_start_qualification import (
    qualify_initial_start_cell,
)
from lunar_policy_training.environment.macro_step import PolicyAction
from lunar_policy_training.environment.observation_boundary import (
    SensorBoundaryEvidence,
)
from lunar_policy_training.environment.multires_observation import (
    DetailObservedWindow,
    MultiresSensorObservationState,
)
from lunar_policy_training.environment.observation_builder import Pose2
from lunar_policy_training.environment.parallel_pool import (
    ParallelActions,
    ParallelEnvPool,
)
from lunar_policy_training.environment.visibility import (
    NativeVisibilityEstimator,
    SensorGeometry,
)
from lunar_policy_training.evaluation import report as report_module
from lunar_policy_training.evaluation.report import FormalEvaluationBatch
from lunar_policy_training.formal_preflight import _request_signature
from lunar_policy_training.policy.cross_attention import CrossAttentionPolicy
from lunar_policy_training.reward import compute_transition_reward
from lunar_policy_training.polar_data.formal_cache import (
    FormalCacheIdentity,
    StaticSceneData,
    write_formal_cache,
)
from lunar_policy_training.polar_data.raster import (
    GridGeometry,
    MapCanvas,
)
from lunar_policy_training.polar_data.hazards import (
    FORMAL_GENERATOR_VERSION,
    scene_seed,
)
from lunar_policy_training.project_capability import load_project_formal_capability


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_formal_episode_maps_visited_detail_poses_to_current_graph_state_ids() -> None:
    episode = object.__new__(FormalEpisode)
    episode._visited_candidate_filter_enabled = True
    episode._visited_primitive_pose_keys = {(5, 10), (15, 20)}
    snapshot = SimpleNamespace(
        state_ids=np.asarray([11, 22, 33], dtype=np.uint64),
        positions_m=np.asarray(
            [[1.01, 2.01, 0.0], [3.01, 4.01, 0.0], [9.0, 9.0, 0.0]],
            dtype=np.float64,
        ),
    )

    assert episode._visited_primitive_state_ids(snapshot) == {11, 22}

    episode._visited_candidate_filter_enabled = False
    assert episode._visited_primitive_state_ids(snapshot) == set()


def test_primitive_detail_aggregation_requires_complete_point_two_metre_evidence() -> None:
    detail_canvas = MapCanvas(
        "e" * 64,
        (0.0, 0.0, 64.0, 64.0),
        GridGeometry(64.0, 0.2, 320),
    )
    shape = (320, 320)
    valid = np.ones(shape, dtype=np.bool_)
    valid[0, 0] = False
    elevation = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
    obstacle = np.zeros(shape, dtype=np.float32)
    obstacle[10:20, 10:20] = 0.4
    detail = DetailObservedWindow(
        canvas=detail_canvas,
        elevation_m=elevation,
        physical_obstacle_ratio=obstacle,
        physical_obstacle_height_m=obstacle * 2.0,
        forbidden_ratio=np.zeros(shape, dtype=np.float32),
        valid_mask=valid,
        observation_age_s=np.zeros(shape, dtype=np.float32),
        observation_quality=np.ones(shape, dtype=np.float32),
        observation_count=np.ones(shape, dtype=np.uint32),
    )

    aggregated = _aggregate_primitive_detail(detail)

    assert aggregated.canvas.geometry == GridGeometry(64.0, 2.0, 32)
    assert not bool(aggregated.valid_mask[0, 0])
    assert bool(aggregated.valid_mask[1, 1])
    assert aggregated.physical_obstacle_ratio[1, 1] == pytest.approx(0.4)
    assert aggregated.elevation_m[1, 1] == pytest.approx(
        float(elevation[10:20, 10:20].mean(dtype=np.float64))
    )


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _coverability(
    starts: dict[str, tuple[int, int] | None],
) -> dict[str, PlatformCoverability]:
    shape = (256, 256)
    detail_shape = (5120, 5120)
    eligible_detail = np.ones(detail_shape, dtype=np.bool_)
    eligible_bits = pack_detail_mask(eligible_detail)
    eligible_hash = mask_sha256(eligible_detail)
    empty_detail = np.zeros(detail_shape, dtype=np.bool_)
    empty_bits = pack_detail_mask(empty_detail)
    empty_hash = mask_sha256(empty_detail)
    output: dict[str, PlatformCoverability] = {}
    for platform, start in starts.items():
        reachable = np.full(shape, start is not None, dtype=np.bool_)
        count = int(eligible_detail.size) if start is not None else 0
        output[platform] = PlatformCoverability(
            platform_type=platform,
            qualified_start_cell=start,
            qualified_start_state=(
                None
                if start is None
                else QualifiedStartState(
                    position_m=(
                        (float(start[1]) + 0.5) * 4.0,
                        1024.0 - (float(start[0]) + 0.5) * 4.0,
                        7.33 if platform == "LEGGED" else 7.0,
                    ),
                    yaw_rad=0.0,
                    motion_mode=1 if platform == "LEGGED" else 0,
                    body_z_m=(
                        (7.28, 7.38)
                        if platform == "LEGGED"
                        else (0.0, 0.0)
                    ),
                )
            ),
            reachable_pose_mask=reachable,
            coverable_detail_shape=detail_shape,
            coverable_detail_bits=(eligible_bits if start is not None else empty_bits),
            coverable_ratio=np.full(shape, start is not None, dtype=np.float32),
            mission_target_detail_cell_count=count,
            coverable_detail_cell_count=count,
            mission_coverable_fraction=1.0 if count else 0.0,
            initial_coverable_fraction=0.1 if count else 0.0,
            initial_candidate_count=1 if count else 0,
            primitive_state_count=int(reachable.sum(dtype=np.int64)),
            certified_edge_count=1 if count else 0,
            recoverable_state_count=int(reachable.sum(dtype=np.int64)),
            reachability_algorithm_id=f"test-reachability/{platform.lower()}",
            primitive_state_schema=f"test-state/{platform.lower()}",
            primitive_set_sha256=_sha(f"primitive-set/{platform}"),
            world_evidence_sha256=_sha(f"world-evidence/{platform}"),
            reachability_graph_sha256=_sha(f"primitive-graph/{platform}"),
            visibility_algorithm_id="two-dimensional-detail-los/v1",
            reachable_mask_sha256=mask_sha256(reachable),
            coverable_mask_sha256=(eligible_hash if start is not None else empty_hash),
            exact=True,
            eligible=start is not None,
            ineligible_reason=(
                None if start is not None else IneligibleReason.UNSAFE_START
            ),
        )
    return output


def _cache(tmp_path: pathlib.Path):
    bundle = load_project_formal_capability(REPOSITORY_ROOT)
    scene_id = _sha("formal-builder-scene")
    window_sha = "d" * 64
    scenario_sha = _sha("formal-builder-scenario-manifest")
    scenario_manifest = {
        "schema": "test-formal-scenario-manifest/v1",
        "scenario_manifest_sha256": scenario_sha,
        "scenarios": [
            {
                "scene_id": scene_id,
                "source": "NASA_LOLA",
                "split": "train",
                "window_id": "flat-window",
                "window_sha256": window_sha,
                "scenario_seed": 408000,
                "scene_seed": scene_seed(
                    window_sha,
                    408000,
                    generator_version=FORMAL_GENERATOR_VERSION,
                ),
                "procedural_overlay": True,
                "world_bounds_m": [0.0, 0.0, 1024.0, 1024.0],
            }
        ],
    }
    identity = FormalCacheIdentity(
        source_lock_file_sha256=_sha("source-lock"),
        source_sha256s={
            "NASA_LOLA_87S_DEM": _sha("dem"),
            "NASA_LOLA_87S_COUNT": _sha("count"),
            "JAXA_LUPEX_DATA_S1": _sha("jaxa"),
        },
        split_manifest_file_sha256=_sha("split-file"),
        split_sha256=_sha("split"),
        scenario_manifest_sha256=scenario_sha,
        generator_sha256=_sha("generator"),
        capability_sha256=bundle.bundle_sha256,
        reward_sha256=_sha("reward"),
        training_semantics_sha256=_sha("semantics"),
        v3_source_commit="a" * 40,
        v3_sha256=_sha("v3"),
    )
    shape = (256, 256)
    hard = {
        platform: np.ones(shape, np.uint8)
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    }
    clearance = {
        platform: np.ones(shape, np.float32)
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    }
    scene = StaticSceneData(
        scene_id=scene_id,
        source="NASA_LOLA",
        split="train",
        window_id="flat-window",
        window_sha256=window_sha,
        world_bounds_m=(0.0, 0.0, 1024.0, 1024.0),
        elevation_m=np.full(shape, 7.0, np.float32),
        valid_mask=np.ones(shape, bool),
        physical_obstacle_ratio=np.zeros(shape, np.float32),
        physical_obstacle_height_m=np.zeros(shape, np.float32),
        forbidden_ratio=np.zeros(shape, np.float32),
        rocks=np.empty((0, 4), np.float64),
        craters=np.empty((0, 4), np.float64),
        no_go_vertices=np.empty((0, 6, 2), np.float64),
        hard_feasible=hard,
        clearance_margin_norm=clearance,
        coverability=_coverability(
            {
                "WHEELED": (127, 127),
                "LEGGED": (127, 127),
                "HOPPER": (127, 127),
            }
        ),
    )
    root = tmp_path / "cache"
    write_formal_cache(
        root,
        identity=identity,
        scenario_manifest=scenario_manifest,
        materialization="preflight",
        scenes=(scene,),
        repository_root=REPOSITORY_ROOT,
    )
    return root / "cache-manifest.json", bundle, scene_id


def _cache_with_common_unstartable_scene(
    tmp_path: pathlib.Path,
    *,
    include_eligible: bool = True,
):
    manifest_path, bundle, eligible_scene_id = _cache(tmp_path)
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario_path = manifest_path.parent / "scenario-manifest.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    ineligible_scene_id = _sha("formal-builder-common-unstartable")
    ineligible_scenario = {
        **scenario["scenarios"][0],
        "scene_id": ineligible_scene_id,
        "scenario_seed": 408001,
        "scene_seed": scene_seed(
            "d" * 64,
            408001,
            generator_version=FORMAL_GENERATOR_VERSION,
        ),
    }
    scenarios = [ineligible_scenario]
    if include_eligible:
        scenarios.append(scenario["scenarios"][0])
    scenario_sha = _sha(
        "formal-builder-two-scene-manifest"
        if include_eligible
        else "formal-builder-unstartable-only-manifest"
    )
    scenario_manifest = {
        **scenario,
        "scenario_manifest_sha256": scenario_sha,
        "scenarios": scenarios,
    }
    identity = FormalCacheIdentity(
        **{
            **original["identity"],
            "scenario_manifest_sha256": scenario_sha,
        }
    )
    shape = (256, 256)
    hard = {
        platform: np.ones(shape, np.uint8)
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    }
    clearance = {
        platform: np.ones(shape, np.float32)
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    }

    def record(
        scene_id: str,
        starts: dict[str, tuple[int, int] | None],
    ) -> StaticSceneData:
        return StaticSceneData(
            scene_id=scene_id,
            source="NASA_LOLA",
            split="train",
            window_id="flat-window",
            window_sha256="d" * 64,
            world_bounds_m=(0.0, 0.0, 1024.0, 1024.0),
            elevation_m=np.full(shape, 7.0, np.float32),
            valid_mask=np.ones(shape, bool),
            physical_obstacle_ratio=np.zeros(shape, np.float32),
            physical_obstacle_height_m=np.zeros(shape, np.float32),
            forbidden_ratio=np.zeros(shape, np.float32),
            rocks=np.empty((0, 4), np.float64),
            craters=np.empty((0, 4), np.float64),
            no_go_vertices=np.empty((0, 6, 2), np.float64),
            hard_feasible=hard,
            clearance_margin_norm=clearance,
            coverability=_coverability(starts),
        )

    root = tmp_path / "qualified-cache"
    records = [
        record(
            ineligible_scene_id,
            {
                "WHEELED": (127, 127),
                "LEGGED": (127, 127),
                "HOPPER": None,
            },
        )
    ]
    if include_eligible:
        records.append(
            record(
                eligible_scene_id,
                {
                    "WHEELED": (127, 127),
                    "LEGGED": (127, 127),
                    "HOPPER": (127, 127),
                },
            )
        )
    write_formal_cache(
        root,
        identity=identity,
        scenario_manifest=scenario_manifest,
        materialization="preflight",
        scenes=records,
        repository_root=REPOSITORY_ROOT,
    )
    return root / "cache-manifest.json", bundle, eligible_scene_id


def _assembly(tmp_path: pathlib.Path):
    manifest, bundle, scene_id = _cache(tmp_path)
    assembly = FormalEnvironmentBuilder(
        cache_manifest_path=manifest,
        capability_bundle=bundle,
        split="train",
        allow_preflight=True,
    ).build()
    return assembly, bundle, scene_id


@pytest.mark.parametrize("platform", ("WHEELED", "LEGGED", "HOPPER"))
def test_formal_worker_uses_one_scene_current_capability_and_safe_start(
    tmp_path: pathlib.Path, platform: str
) -> None:
    assembly, bundle, scene_id = _assembly(tmp_path)

    first = assembly.factory(0, platform)
    second = assembly.factory(0, platform)

    assert first.episode.scene_id == second.episode.scene_id == scene_id
    assert first.episode.current_pose == second.episode.current_pose
    assert first.episode.start_cell == second.episode.start_cell == (127, 127)
    assert first.episode.start_is_safe
    assert first.environment.sensor_closed_loop
    assert first.episode.capability is bundle.for_platform(platform)
    assert first.initial_observation.candidate_mask.any()
    assert first.initial_observation.observation_identities[0].episode_id.startswith(
        scene_id
    )


def test_formal_episode_passes_exact_platform_and_exposes_candidate_diagnostics(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    original_build = CandidateBuilderV2.build_from_primitive_graph
    platform_calls: list[str] = []

    def record_platform(self, *args, **kwargs):
        platform_calls.append(kwargs["platform_type"])
        return original_build(self, *args, **kwargs)

    monkeypatch.setattr(
        CandidateBuilderV2, "build_from_primitive_graph", record_platform
    )
    worker = assembly.factory(0, "HOPPER")

    assert platform_calls
    assert set(platform_calls) == {"HOPPER"}
    diagnostics = worker.episode.current_candidate_diagnostics()
    assert worker.current_candidate_diagnostics() == diagnostics
    assert diagnostics.emitted_count == int(
        worker.environment.current_observation.candidate_mask.sum()
    )


def test_cache_start_qualification_matches_the_episode_candidate_semantics(
    tmp_path: pathlib.Path,
) -> None:
    assembly, bundle, _ = _assembly(tmp_path)
    worker = assembly.factory(0, "WHEELED")
    loaded = worker.episode.loaded

    qualified = qualify_initial_start_cell(
        scene=loaded.scene,
        arrays=loaded.arrays,
        capability=bundle.for_platform("WHEELED"),
    )
    no_roi_arrays = {
        **loaded.arrays,
        "forbidden_ratio": np.ones((256, 256), np.float32),
    }

    assert qualified is not None
    qualified_episode = FormalEpisode(
        worker_index=worker.episode.worker_index,
        platform_type="WHEELED",
        capability=bundle.for_platform("WHEELED"),
        scenario_identity=worker.episode.scenario_identity,
        loaded=loaded,
        start_cell=qualified,
    )
    assert bool(qualified_episode.initial_observation.candidate_mask.any())
    assert qualify_initial_start_cell(
        scene=loaded.scene,
        arrays=no_roi_arrays,
        capability=bundle.for_platform("WHEELED"),
    ) is None


def test_start_qualification_skips_a_native_unsafe_start_candidate(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembly, bundle, _ = _assembly(tmp_path)
    worker = assembly.factory(0, "WHEELED")
    original_update = qualification_module.ObservedPrimitiveReachability.update
    calls = 0

    def reject_first_start(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("WHEEL_START_NOT_SAFE")
        return original_update(self, *args, **kwargs)

    monkeypatch.setattr(
        qualification_module.ObservedPrimitiveReachability,
        "update",
        reject_first_start,
    )

    qualified = qualify_initial_start_cell(
        scene=worker.episode.loaded.scene,
        arrays=worker.episode.loaded.arrays,
        capability=bundle.for_platform("WHEELED"),
    )

    assert calls >= 2
    assert qualified is not None


def test_start_qualification_does_not_hide_other_native_failures(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembly, bundle, _ = _assembly(tmp_path)
    worker = assembly.factory(0, "LEGGED")

    def fail_graph(*_args, **_kwargs):
        raise RuntimeError("PRIMITIVE_GRAPH_FAILED")

    monkeypatch.setattr(
        qualification_module.ObservedPrimitiveReachability,
        "update",
        fail_graph,
    )

    with pytest.raises(RuntimeError, match="PRIMITIVE_GRAPH_FAILED"):
        qualify_initial_start_cell(
            scene=worker.episode.loaded.scene,
            arrays=worker.episode.loaded.arrays,
            capability=bundle.for_platform("LEGGED"),
        )


def test_three_platforms_share_physical_scene_but_keep_distinct_projection(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, scene_id = _assembly(tmp_path)
    workers = {
        platform: assembly.factory(0, platform)
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    }

    assert {worker.episode.scene_id for worker in workers.values()} == {scene_id}
    assert len({worker.episode.episode_seed for worker in workers.values()}) == 1
    assert len({worker.episode.start_seed for worker in workers.values()}) == 3
    assert {
        tuple(worker.initial_observation.platform_context[0].tolist())
        for worker in workers.values()
    } == {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)}


def test_training_uses_each_platforms_own_eligible_scene_lane(
    tmp_path: pathlib.Path,
) -> None:
    manifest, bundle, eligible_scene_id = _cache_with_common_unstartable_scene(
        tmp_path
    )
    assembly = FormalEnvironmentBuilder(
        cache_manifest_path=manifest,
        capability_bundle=bundle,
        split="train",
        allow_preflight=True,
    ).build()

    workers = {
        platform: assembly.factory(0, platform)
        for platform in ("WHEELED", "LEGGED", "HOPPER")
    }

    assert workers["HOPPER"].episode.scene_id == eligible_scene_id
    assert workers["WHEELED"].episode.loaded.entry["platform_coverability"][
        "WHEELED"
    ]["eligible"]
    assert workers["LEGGED"].episode.loaded.entry["platform_coverability"][
        "LEGGED"
    ]["eligible"]
    assert all(
        bool(worker.initial_observation.candidate_mask.any())
        for worker in workers.values()
    )


def test_only_the_empty_platform_lane_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    manifest, bundle, _ = _cache_with_common_unstartable_scene(
        tmp_path,
        include_eligible=False,
    )

    assembly = FormalEnvironmentBuilder(
        cache_manifest_path=manifest,
        capability_bundle=bundle,
        split="train",
        allow_preflight=True,
    ).build()

    assert assembly.factory(0, "WHEELED").episode.start_cell == (127, 127)
    with pytest.raises(ValueError, match="platform-eligible"):
        assembly.factory(0, "HOPPER")


def test_formal_episode_cursor_is_deterministic_and_resume_exact(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, scene_id = _assembly(tmp_path)

    first = assembly.factory.create_for_episode(0, "WHEELED", 5)
    resumed = assembly.factory.create_for_episode(0, "WHEELED", 5)
    next_episode = assembly.factory.create_for_episode(0, "WHEELED", 6)

    assert first.episode.scene_id == resumed.episode.scene_id == scene_id
    assert first.episode.current_pose == resumed.episode.current_pose
    assert first.episode.start_seed == resumed.episode.start_seed
    assert first.episode.episode_seed == resumed.episode.episode_seed
    assert (
        first.initial_observation.observation_identities
        == resumed.initial_observation.observation_identities
    )
    assert first.episode.scenario_identity.episode_cursor == 5
    assert "/episode-5" in first.initial_observation.observation_identities[0].episode_id
    assert (
        next_episode.initial_observation.observation_identities[0].episode_id
        != first.initial_observation.observation_identities[0].episode_id
    )
    assert next_episode.episode.start_seed != first.episode.start_seed
    assert next_episode.episode.episode_seed != first.episode.episode_seed


def test_formal_episode_excludes_a_recorded_landing_from_future_candidates(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory.create_for_episode(
        0,
        "HOPPER",
        4,
        platform_worker_index=0,
        platform_worker_count=1,
    )
    episode = worker.episode
    initial = worker.environment.current_observation
    candidate = int(initial.candidate_mask[0].nonzero()[0])
    visited_position = initial.frontier_features[0, candidate, :2].clone()
    canvas = episode.loaded.scene.base_canvas
    target_x = canvas.bounds_m[0] + float(visited_position[0]) * canvas.geometry.size_m
    target_y = canvas.bounds_m[3] - float(visited_position[1]) * canvas.geometry.size_m
    target_cell = canvas.world_to_grid(target_x, target_y)
    episode._record_reveal(
        SensorBoundaryEvidence(
            Pose2(
                target_x,
                target_y,
                0.0,
                "map",
                float(episode.loaded.arrays["elevation_m"][target_cell]),
            ),
            1.0,
        ),
        "LANDED_HOLD",
    )
    rebuilt = episode.build_policy_observation(
        episode.sensor_state.observed,
        episode.current_pose,
    )
    rebuilt_positions = rebuilt.frontier_features[0, rebuilt.candidate_mask[0], :2]

    assert int(rebuilt.candidate_mask.sum()) == int(initial.candidate_mask.sum())
    assert episode.current_candidate_diagnostics().visited_excluded_count > 0
    assert not torch.any(torch.all(rebuilt_positions == visited_position, dim=1))


def test_formal_episode_navigation_stack_preserves_exact_parent_pose(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    episode = assembly.factory.create_for_episode(
        0,
        "HOPPER",
        4,
        platform_worker_index=0,
        platform_worker_count=1,
    ).episode
    canvas = episode.loaded.scene.base_canvas
    target_cell = (episode.start_cell[0], episode.start_cell[1] + 1)
    target_x, target_y = canvas.grid_center_world(*target_cell)
    target = Pose2(
        target_x + 0.02,
        target_y - 0.02,
        elevation_m=123.0,
    )
    episode._record_reveal(
        SensorBoundaryEvidence(target, 1.0), "LANDED_HOLD"
    )

    assert episode._navigation_stack[-1] == target

    episode._record_reveal(
        SensorBoundaryEvidence(episode._navigation_stack[-2], 1.0),
        "LANDED_HOLD",
    )

    assert episode._navigation_stack == [episode.current_pose]


def test_formal_ground_reference_interpolates_one_metre_sensor_samples(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    episode = assembly.factory.create_for_episode(
        0,
        "WHEELED",
        4,
        platform_worker_index=0,
        platform_worker_count=1,
    ).episode
    start = episode.current_pose
    direction = (
        1.0
        if start.x_m + 4.0 < episode.loaded.scene.base_canvas.bounds_m[2]
        else -1.0
    )
    end = Pose2(
        start.x_m + 4.0 * direction,
        start.y_m,
        yaw_rad=math.pi / 2.0,
        elevation_m=start.elevation_m,
    )
    bridge = formal_builder_module.bridge_api
    trajectory = bridge.TrajectoryReference()
    trajectory.semantics = bridge.TrajectorySemantics.WHEELED_BASE
    points = []
    for pose, elapsed_s in ((start, 0.0), (end, 4.0)):
        point = bridge.TrajectoryPoint()
        point.pose = formal_builder_module._pose3(pose)
        point.time_from_start = timedelta(seconds=elapsed_s)
        points.append(point)
    trajectory.points = points
    reference = bridge.MotionReference()
    reference.plan_id = "path-observation-test"
    reference.platform_type = "WHEELED"
    reference.data = trajectory
    before_revision = episode._revision

    execution = episode.execute_reference(reference)

    evidence = execution.sensor_boundary_evidence
    assert evidence is not None
    assert evidence.path_samples[-1].pose_map == evidence.pose_map
    assert evidence.pose_map.x_m == end.x_m
    assert evidence.pose_map.y_m == end.y_m
    assert evidence.pose_map.yaw_rad == pytest.approx(end.yaw_rad)
    assert sum(sample.elapsed_s for sample in evidence.path_samples) == pytest.approx(
        4.0
    )
    path = (start,) + tuple(sample.pose_map for sample in evidence.path_samples)
    assert all(
        math.dist((left.x_m, left.y_m), (right.x_m, right.y_m)) <= 1.0
        for left, right in zip(path, path[1:])
    )
    assert episode._reveal_history[-1].path_samples

    episode.controller.after_execution(
        platform_type="WHEELED",
        execution_state="DECISION_BOUNDARY",
        evidence=evidence,
    )
    assert episode._revision == before_revision + 1


def test_formal_episode_estimates_candidate_gain_from_detail_observation(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory.create_for_episode(
        0,
        "WHEELED",
        4,
        platform_worker_index=0,
        platform_worker_count=1,
    )

    estimator = worker.episode._candidate_builder._visibility_estimator

    assert estimator is worker.episode.sensor_state
    assert estimator.resolution_m == 0.2
    assert worker.snapshot_episode_state()["candidate_gain_resolution_m"] == 0.2


def test_restore_rejects_a_legacy_state_without_detail_gain_identity(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory.create_for_episode(
        0,
        "WHEELED",
        4,
        platform_worker_index=0,
        platform_worker_count=1,
    )
    legacy_state = worker.snapshot_episode_state()
    legacy_state.pop("candidate_gain_resolution_m")

    with pytest.raises(ValueError, match="structure"):
        assembly.factory.restore_for_episode(
            worker_index=0,
            platform_type="WHEELED",
            episode_cursor=4,
            platform_worker_index=0,
            platform_worker_count=1,
            state=legacy_state,
        )


def test_formal_candidate_path_never_calls_the_legacy_frontier_builder(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory.create_for_episode(
        0,
        "WHEELED",
        4,
        platform_worker_index=0,
        platform_worker_count=1,
    )

    def reject_legacy(*_args, **_kwargs):
        raise AssertionError("formal v10 must not call legacy frontier generation")

    with monkeypatch.context() as patch:
        patch.setattr(CandidateBuilderV2, "build", reject_legacy)
        worker.episode.build_policy_observation(
            worker.episode.sensor_state.observed,
            worker.episode.current_pose,
        )


@pytest.mark.parametrize("platform", ("WHEELED", "LEGGED", "HOPPER"))
def test_active_formal_episode_replays_to_exact_observation_and_request(
    tmp_path: pathlib.Path, platform: str,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory.create_for_episode(
        0,
        platform,
        4,
        platform_worker_index=0,
        platform_worker_count=1,
    )
    initial = worker.environment.current_observation
    candidate = int(initial.candidate_mask[0].nonzero()[0])
    result = worker.environment.advance_prepared_action(
        PolicyAction(candidate, 0.0),
        expected_identity=initial.observation_identities[0],
    )
    assert not result.transition.terminated
    state = FormalWorkerState.from_dict(worker.snapshot_episode_state())

    restored = assembly.factory.restore_for_episode(
        worker_index=0,
        platform_type=platform,
        episode_cursor=4,
        platform_worker_index=0,
        platform_worker_count=1,
        state=state.to_dict(),
    )

    uninterrupted_observation = worker.environment.current_observation
    restored_observation = restored.environment.current_observation
    assert restored_observation.observation_identities == (
        uninterrupted_observation.observation_identities
    )
    assert policy_batch_sha256(restored_observation) == policy_batch_sha256(
        uninterrupted_observation
    )
    next_candidate = int(uninterrupted_observation.candidate_mask[0].nonzero()[0])
    action = PolicyAction(next_candidate, 0.25)
    uninterrupted_request = worker.episode.build_request(
        action, uninterrupted_observation.observation_identities[0]
    ).request
    restored_request = restored.episode.build_request(
        action, restored_observation.observation_identities[0]
    ).request
    assert _request_signature(restored_request) == _request_signature(
        uninterrupted_request
    )

    uninterrupted_result = worker.environment.advance_prepared_action(
        action,
        expected_identity=uninterrupted_observation.observation_identities[0],
    )
    restored_result = restored.environment.advance_prepared_action(
        action,
        expected_identity=restored_observation.observation_identities[0],
    )
    assert policy_batch_sha256(
        restored_result.transition.next_observation
    ) == policy_batch_sha256(uninterrupted_result.transition.next_observation)
    assert compute_transition_reward(restored_result.transition) == (
        compute_transition_reward(uninterrupted_result.transition)
    )


def test_rejected_candidate_mask_survives_active_episode_replay(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory(0, "WHEELED")
    candidate = int(worker.initial_observation.candidate_mask[0].nonzero()[0])
    worker.environment._mask_rejected_candidate(candidate)
    state = worker.snapshot_episode_state()

    restored = assembly.factory.restore_for_episode(
        worker_index=0,
        platform_type="WHEELED",
        episode_cursor=0,
        platform_worker_index=0,
        platform_worker_count=1,
        state=state,
    )

    assert not bool(
        restored.environment.current_observation.candidate_mask[0, candidate]
    )
    assert restored.snapshot_episode_state() == state


def test_active_episode_rejects_a_different_coverability_mask(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory(0, "WHEELED")
    state = worker.snapshot_episode_state()
    state["coverability_mask_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="replay identity"):
        assembly.factory.restore_for_episode(
            worker_index=0,
            platform_type="WHEELED",
            episode_cursor=0,
            platform_worker_index=0,
            platform_worker_count=1,
            state=state,
        )


def test_parallel_pool_snapshots_and_restores_the_active_formal_episode(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    with ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=assembly.observation_template,
        environment_factory=assembly.factory,
        reward_fn=compute_transition_reward,
        worker_timeout_seconds=120.0,
        worker_startup_timeout_seconds=60.0,
        initial_episode_cursors=(4,),
    ) as uninterrupted:
        initial = uninterrupted.reset()
        candidate = int(initial.observations.candidate_mask[0].nonzero()[0])
        stepped = uninterrupted.step(
            ParallelActions(
                candidate_indices=torch.tensor([candidate], dtype=torch.int64),
                thetas=torch.tensor([0.0], dtype=torch.float32),
            ),
            policy_version=9,
        )
        states = uninterrupted.snapshot_episode_states(policy_version=9)

    with ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=assembly.observation_template,
        environment_factory=assembly.factory,
        reward_fn=compute_transition_reward,
        worker_timeout_seconds=120.0,
        worker_startup_timeout_seconds=600.0,
        initial_episode_states=states,
    ) as resumed:
        restored = resumed.reset()

    assert resumed.episode_cursors == (4,)
    assert restored.observations.observation_identities == (
        stepped.observations.observation_identities
    )
    assert policy_batch_sha256(restored.observations) == policy_batch_sha256(
        stepped.observations
    )


def test_formal_schedule_uses_platform_local_scene_lanes_at_any_allocation() -> None:
    assert [
        _formal_schedule_index(
            index,
            0,
            192,
            platform_worker_index=index,
            platform_worker_count=24,
        )
        for index in range(24)
    ] == list(range(24))
    assert [
        _formal_schedule_index(
            global_index,
            1,
            192,
            platform_worker_index=local_index,
            platform_worker_count=3,
        )
        for global_index, local_index in ((0, 0), (3, 0), (6, 0))
    ] == [3, 3, 3]
    assert _formal_schedule_index(
        5,
        2,
        192,
        platform_worker_index=2,
        platform_worker_count=3,
    ) == 8


def test_formal_scene_schedule_is_one_seeded_permutation_without_replacement() -> None:
    """Would fail if a per-episode hash offset duplicated or skipped scenes."""
    entries = tuple(
        {
            "scene_id": character * 64,
            "split": "validation",
            "platform_coverability": {
                platform: {"eligible": True}
                for platform in ("WHEELED", "LEGGED", "HOPPER")
            },
        }
        for character in "abcde"
    )

    scheduled = formal_builder_module._formal_scheduled_entries(
        tuple(reversed(entries)),
        scenario_schedule_id="formal-cache/validation/v6",
    )
    repeated = formal_builder_module._formal_scheduled_entries(
        entries,
        scenario_schedule_id="formal-cache/validation/v6",
    )

    assert tuple(entry["scene_id"] for entry in scheduled) == (
        "b" * 64,
        "d" * 64,
        "e" * 64,
        "c" * 64,
        "a" * 64,
    )
    assert tuple(entry["scene_id"] for entry in repeated) == tuple(
        entry["scene_id"] for entry in scheduled
    )
    assert {entry["scene_id"] for entry in scheduled} == {
        entry["scene_id"] for entry in entries
    }


def test_formal_worker_traverses_the_seeded_scene_permutation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if the worker added a cursor-varying offset to the permutation."""
    schedule_id = "formal-cache/validation/v6"
    entries = tuple(
        {
            "scene_id": character * 64,
            "split": "validation",
            "platform_coverability": {
                platform: {"eligible": True}
                for platform in ("WHEELED", "LEGGED", "HOPPER")
            },
        }
        for character in reversed("abcde")
    )
    cache = type("Cache", (), {"manifest": {"scenes": entries}})()
    monkeypatch.setattr(
        formal_builder_module,
        "load_formal_cache",
        lambda *_args, **_kwargs: cache,
    )
    monkeypatch.setattr(
        formal_builder_module,
        "_load_multires_scene",
        lambda _cache, scene_id: scene_id,
    )
    builder = FormalWorkerBuilder(
        cache_manifest_path="/unused/cache-manifest.json",
        split="validation",
        allow_preflight=True,
        scenario_schedule_id=schedule_id,
    )

    loaded = []
    for cursor in range(5):
        identity = ScenarioIdentity(
            platform_type="WHEELED",
            scenario_schedule_id=schedule_id,
            worker_index=0,
            episode_cursor=cursor,
            platform_worker_index=0,
            platform_worker_count=1,
            capability_version="test/v1",
            capability_sha256="f" * 64,
        )
        loaded.append(builder._load_scheduled_scene(0, identity))

    assert tuple(loaded) == (
        "b" * 64,
        "d" * 64,
        "e" * 64,
        "c" * 64,
        "a" * 64,
    )


def test_policy_input_and_request_are_observed_only_identity_bound_and_multires(
    tmp_path: pathlib.Path,
) -> None:
    assembly, bundle, _ = _assembly(tmp_path)
    worker = assembly.factory(0, "WHEELED")
    observation = worker.initial_observation
    identity = observation.observation_identities[0]
    candidate = int(observation.candidate_mask[0].nonzero()[0])

    prepared = worker.episode.build_request(
        PolicyAction(candidate, 0.0), identity
    )
    request = prepared.request

    assert prepared.identity == identity
    assert identity.map_snapshot_id in request.request_id
    assert request.capability_version == bundle.for_platform("WHEELED").capability_version
    assert observation.prior_channels.shape[-2:] == (256, 256)
    assert request.world.global_map.resolution_m == 4.0
    assert request.world.global_map.width == 256
    assert request.world.local_map.resolution_m == 0.2
    assert request.world.local_map.width == 320
    global_valid = request.world.global_map.layers["valid_mask"].values
    global_elevation = request.world.global_map.layers["elevation"].values
    assert np.all(global_elevation[global_valid == 0] == 0.0)
    assert not hasattr(request, "hopper_propellant")


def test_ground_option_freezes_target_when_candidate_arrays_refresh(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory(0, "WHEELED")
    episode = worker.episode
    observation = worker.initial_observation
    identity = observation.observation_identities[0]
    candidate = int(observation.candidate_mask[0].nonzero()[0])
    action = PolicyAction(candidate, 0.4)

    first = episode.begin_ground_option(action, identity).request
    first_goal = first.goal.target.position_m
    frozen = (first_goal.x, first_goal.y, first_goal.z, first.goal.goal_id)
    episode._snapshot.candidates.features[candidate, :2] = (0.01, 0.99)

    continued = episode.continue_ground_option(identity).request
    continued_goal = continued.goal.target.position_m

    assert (
        continued_goal.x,
        continued_goal.y,
        continued_goal.z,
        continued.goal.goal_id,
    ) == frozen
    with pytest.raises(ValueError, match="active ground option"):
        worker.snapshot_episode_state()
    episode.clear_ground_option()
    worker.snapshot_episode_state()
    with pytest.raises(RuntimeError, match="no active ground option"):
        episode.ground_option_distance_m()


@pytest.mark.parametrize("platform", ("WHEELED", "LEGGED"))
def test_ground_reference_executes_to_certified_endpoint(
    tmp_path: pathlib.Path, platform: str
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory(0, platform)
    observation = worker.initial_observation
    identity = observation.observation_identities[0]
    candidate = int(observation.candidate_mask[0].nonzero()[0])
    before = worker.episode.current_pose

    result = worker.environment.advance_prepared_action(
        PolicyAction(candidate, 0.0), expected_identity=identity
    )

    assert not result.transition.hard_safety_violation
    assert worker.episode.current_pose != before
    assert result.transition.execution_events.reference_samples_consumed >= 2


def test_ground_option_rebuilds_candidates_only_at_final_policy_boundary(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory(0, "WHEELED")
    episode = worker.episode
    observation = worker.initial_observation
    identity = observation.observation_identities[0]
    candidate = int(observation.candidate_mask[0].nonzero()[2])
    original_builder = episode._candidate_builder
    candidate_builds = 0
    reference_count = 0

    class CountingCandidateBuilder:
        def build_from_primitive_graph(self, *args, **kwargs):
            nonlocal candidate_builds
            candidate_builds += 1
            return original_builder.build_from_primitive_graph(*args, **kwargs)

    original_executor = worker.environment._reference_executor

    def counting_executor(reference):
        nonlocal reference_count
        reference_count += 1
        return original_executor(reference)

    episode._candidate_builder = CountingCandidateBuilder()
    worker.environment._reference_executor = counting_executor

    result = worker.environment.advance_prepared_action(
        PolicyAction(candidate, math.pi), expected_identity=identity
    )

    assert not result.transition.hard_safety_violation
    assert reference_count > 1
    assert candidate_builds == 1


def test_hopper_executes_one_nominal_landing_without_cumulative_fuel(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)

    available_delta_v = []
    for worker_index in (0, 1):
        worker = assembly.factory(worker_index, "HOPPER")
        observation = worker.initial_observation
        identity = observation.observation_identities[0]
        candidate = int(observation.candidate_mask[0].nonzero()[0])
        result = worker.environment.advance_prepared_action(
            PolicyAction(candidate, 0.0), expected_identity=identity
        )
        assert not result.transition.hard_safety_violation
        assert result.transition.execution_events.hopper_commitment_states == (
            "JUMP_COMMITTED",
            "IN_FLIGHT",
            "LANDED_HOLD",
        )
        available_delta_v.append(worker.episode.last_hop_available_delta_v_mps)

    assert available_delta_v[0] > 0.0
    assert available_delta_v[0] == available_delta_v[1]


def test_formal_builder_rejects_preflight_cache_by_default(
    tmp_path: pathlib.Path,
) -> None:
    manifest, bundle, _ = _cache(tmp_path)

    with pytest.raises(ValueError, match="full"):
        FormalEnvironmentBuilder(
            cache_manifest_path=manifest,
            capability_bundle=bundle,
            split="train",
        ).build()


def test_formal_evaluation_watchdog_classifies_real_unfinished_workers(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    batch = FormalEvaluationBatch(
        split="validation",
        factory=assembly.factory,
        observation_template=assembly.observation_template,
        scenario_seeds=(409000,),
    )

    with pytest.raises(
        report_module.FormalEvaluationIncomplete,
        match="EVALUATION_INCOMPLETE",
    ):
        report_module._evaluate_formal_chunk(
            CrossAttentionPolicy(),
            method="nearest_frontier",
            device=torch.device("cpu"),
            batch=batch,
            scenario_offset=0,
            scenario_seeds=(409000,),
            watchdog_max_steps=1,
            watchdog_seconds=60.0,
        )

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import hashlib
import inspect
import json
import math
import pathlib

import numpy as np
import pytest
import torch
from lunar_planner_training_bridge import CandidateDisposition

from lunar_policy_training.capability_freeze import ScenarioIdentity
from lunar_policy_training.environment import formal_builder as formal_builder_module
from lunar_policy_training.environment import candidate_builder as candidate_builder_module
from lunar_policy_training.environment import formal_start_qualification as qualification_module
from lunar_policy_training.environment import parallel_pool as parallel_pool_module
from lunar_policy_training.environment.candidate_builder import (
    CandidateBatch,
    CandidateBuilderV2,
    CandidateDiagnostics,
    PhysicalCandidateUniverse,
)
from lunar_policy_training.environment.coverability import (
    IneligibleReason,
    PHYSICAL_GRID_AXIS_CONVENTION,
    PHYSICAL_PROJECTION_SCHEMA,
    PlatformCoverability,
    canonical_physical_positions_um,
    mask_sha256,
    pack_detail_mask,
    physical_projection_sha256,
)
from lunar_policy_training.environment.formal_builder import (
    FormalEpisode,
    FormalEnvironmentBuilder,
    FormalWorkerBuilder,
    _formal_schedule_index,
)
from lunar_policy_training.environment.formal_start_qualification import (
    qualify_initial_start_cell,
)
from lunar_policy_training.environment.macro_step import PolicyAction
from lunar_policy_training.environment.observation_boundary import (
    SensorBoundaryEvidence,
)
from lunar_policy_training.environment.multires_observation import (
    MultiresSensorObservationState,
)
from lunar_policy_training.environment.observation_builder import Pose2
from lunar_policy_training.environment.platform_reachability import (
    PhysicalReachabilityResult,
)
from lunar_policy_training.environment.primitive_reachability import (
    ObservedPrimitiveReachability,
)
from lunar_policy_training.environment.visibility import (
    NativeVisibilityEstimator,
    SensorGeometry,
)
from lunar_policy_training.polar_data.formal_cache import (
    FormalCacheIdentity,
    StaticSceneData,
    load_formal_cache,
    write_formal_cache,
)
from lunar_policy_training.polar_data import formal_cache as formal_cache_module
from lunar_policy_training.polar_data.hazards import (
    FORMAL_GENERATOR_VERSION,
    scene_seed,
)
from lunar_policy_training.project_capability import load_project_formal_capability


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _coverability(
    starts: dict[str, tuple[int, int] | None],
    bundle,
) -> tuple[dict[str, PlatformCoverability], dict[str, str], np.ndarray]:
    shape = (256, 256)
    detail_shape = (5120, 5120)
    eligible_detail = np.ones(detail_shape, dtype=np.bool_)
    eligible_bits = pack_detail_mask(eligible_detail)
    eligible_hash = mask_sha256(eligible_detail)
    empty_detail = np.zeros(detail_shape, dtype=np.bool_)
    empty_bits = pack_detail_mask(empty_detail)
    empty_hash = mask_sha256(empty_detail)
    output: dict[str, PlatformCoverability] = {}
    evidence_algorithms: dict[str, str] = {}
    hopper_positions_m = np.empty((0, 3), dtype=np.float64)
    for platform, start in starts.items():
        reachable = np.zeros(shape, dtype=np.bool_)
        if start is not None:
            reachable[start] = True
        count = int(eligible_detail.size) if start is not None else 0
        positions_m = np.empty((0, 3), dtype=np.float64)
        if start is not None:
            row, column = start
            positions_m = np.asarray(
                (
                    (
                        (float(column) + 0.5) * 4.0,
                        1024.0 - (float(row) + 0.5) * 4.0,
                        7.0,
                    ),
                ),
                dtype=np.float64,
            )
        positions_m = np.ascontiguousarray(positions_m, dtype=np.float64)
        if platform == "HOPPER":
            hopper_positions_m = positions_m
        capability_sha256 = formal_cache_module._physical_capability_content_sha256(
            bundle.for_platform(platform)
        )
        start_sha256 = formal_cache_module._physical_start_identity_from_position(
            platform_type=platform,
            start_cell=start,
            position_m=(
                None
                if start is None
                else tuple(float(value) for value in positions_m[0])
            ),
        )
        reachability_algorithm_id = f"test-physical-reachability/{platform.lower()}"
        evidence_algorithm_id = f"test-physical-evidence/{platform.lower()}"
        evidence_algorithms[platform] = evidence_algorithm_id
        output[platform] = PlatformCoverability(
            platform_type=platform,
            qualified_start_cell=start,
            physical_observation_pose_mask=reachable,
            physical_projection_schema=PHYSICAL_PROJECTION_SCHEMA,
            physical_reachability_algorithm_id=reachability_algorithm_id,
            physical_safe_pose_count=int(reachable.sum(dtype=np.int64)),
            physically_reachable_pose_count=int(
                reachable.sum(dtype=np.int64)
            ),
            physical_projection_sha256=physical_projection_sha256(
                platform_type=platform,
                physical_reachability_algorithm_id=reachability_algorithm_id,
                physical_evidence_algorithm_id=evidence_algorithm_id,
                physical_observation_pose_mask=reachable,
                physical_observation_positions_m=positions_m,
                physical_grid_resolution_m=4.0,
                physical_grid_origin_m=(0.0, 1024.0),
                physical_grid_world_bounds_m=(0.0, 0.0, 1024.0, 1024.0),
                physical_grid_axis_convention=PHYSICAL_GRID_AXIS_CONVENTION,
                capability_content_sha256=capability_sha256,
                start_identity_sha256=start_sha256,
            ),
            mission_target_detail_mask_sha256=(
                eligible_hash if start is not None else empty_hash
            ),
            coverable_detail_shape=detail_shape,
            coverable_detail_bits=(eligible_bits if start is not None else empty_bits),
            coverable_ratio=np.full(shape, start is not None, dtype=np.float32),
            mission_target_detail_cell_count=count,
            coverable_detail_cell_count=count,
            mission_coverable_fraction=1.0 if count else 0.0,
            initial_coverable_fraction=0.1 if count else 0.0,
            initial_candidate_count=1 if count else 0,
            coverable_detail_mask_sha256=(
                eligible_hash if start is not None else empty_hash
            ),
            sensor_visibility_algorithm_id="two-dimensional-detail-los/v1",
            capability_content_sha256=capability_sha256,
            start_identity_sha256=start_sha256,
            exact=True,
            eligible=start is not None,
            ineligible_reason=(
                None if start is not None else IneligibleReason.UNSAFE_START
            ),
        )
    return (
        output,
        evidence_algorithms,
        canonical_physical_positions_um(hopper_positions_m),
    )


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
    coverability, evidence_algorithms, hopper_positions_um = _coverability(
        {
            "WHEELED": (127, 127),
            "LEGGED": (127, 127),
            "HOPPER": (127, 127),
        },
        bundle,
    )
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
        coverability=coverability,
        physical_evidence_algorithm_ids=evidence_algorithms,
        hopper_physical_observation_positions_um=hopper_positions_um,
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
        coverability, evidence_algorithms, hopper_positions_um = _coverability(
            starts,
            bundle,
        )
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
            coverability=coverability,
            physical_evidence_algorithm_ids=evidence_algorithms,
            hopper_physical_observation_positions_um=hopper_positions_um,
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


def _primitive_changed_capability(capability):
    typed = capability.typed_capability
    primitives = getattr(typed, "motion_primitives", None)
    if primitives is not None:
        assert len(primitives) > 1
        changed_first = replace(
            primitives[0],
            primitive_id=f"{primitives[0].primitive_id}/task7-isolation",
        )
        typed = replace(
            typed,
            motion_primitives=(changed_first, *tuple(reversed(primitives[1:]))),
        )
    else:
        assert capability.platform_type == "HOPPER"
        assert not hasattr(typed, "motion_primitives")
    return replace(
        capability,
        typed_capability=typed,
        content_sha256=_sha(
            f"primitive-only-change/{capability.platform_type}"
        ),
    )


def _forbid_primitive_formal_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("formal boundary called primitive reachability")

    monkeypatch.setattr(
        ObservedPrimitiveReachability,
        "update",
        forbidden,
    )
    monkeypatch.setattr(
        CandidateBuilderV2,
        "build_from_primitive_graph",
        forbidden,
    )


@pytest.mark.parametrize("platform", ("WHEELED", "LEGGED", "HOPPER"))
def test_start_qualification_never_calls_primitive_reachability_or_builder(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    manifest, bundle, scene_id = _cache(tmp_path)
    loaded = formal_builder_module._load_multires_scene(
        load_formal_cache(manifest), scene_id
    )
    _forbid_primitive_formal_paths(monkeypatch)
    monkeypatch.setattr(
        formal_cache_module,
        "_global_composed_capability",
        lambda capability, **_kwargs: capability.to_bridge_capability(),
        raising=False,
    )

    qualified = qualification_module.qualify_initial_start(
        scene=loaded.scene,
        arrays=loaded.arrays,
        capability=bundle.for_platform(platform),
    )

    assert qualified is not None
    assert qualified.initial_candidate_count > 0


@pytest.mark.parametrize("platform", ("WHEELED", "LEGGED", "HOPPER"))
def test_formal_runtime_never_calls_primitive_reachability_or_builder(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    _forbid_primitive_formal_paths(monkeypatch)
    monkeypatch.setattr(
        formal_builder_module,
        "_global_composed_capability",
        lambda capability, **_kwargs: capability.to_bridge_capability(),
        raising=False,
    )

    worker = assembly.factory(0, platform)

    assert bool(worker.initial_observation.candidate_mask.any())


@pytest.mark.parametrize("platform", ("WHEELED", "LEGGED", "HOPPER"))
def test_physical_universe_and_oracle_are_repeatable_and_primitive_independent(
    tmp_path: pathlib.Path,
    platform: str,
) -> None:
    assembly, bundle, _ = _assembly(tmp_path)
    first = assembly.factory(0, platform).episode
    repeated = assembly.factory(0, platform).episode
    changed_capability = _primitive_changed_capability(
        bundle.for_platform(platform)
    )
    changed = FormalEpisode(
        worker_index=first.worker_index,
        platform_type=platform,
        capability=changed_capability,
        scenario_identity=first.scenario_identity,
        loaded=first.loaded,
        start_cell=first.start_cell,
    )

    snapshots = (first._snapshot, repeated._snapshot, changed._snapshot)
    assert all(snapshot is not None for snapshot in snapshots)
    assert len(
        {
            snapshot.candidates.diagnostics.physical_candidate_universe_count
            for snapshot in snapshots
        }
    ) == 1
    assert len(
        {snapshot.candidate_universe_sha256 for snapshot in snapshots}
    ) == 1
    assert len(
        {snapshot.frontier_oracle.oracle_opportunity_count for snapshot in snapshots}
    ) == 1
    assert len(
        {
            snapshot.frontier_oracle.oracle_opportunity_set_sha256
            for snapshot in snapshots
        }
    ) == 1


def test_hopper_runtime_preserves_row_major_certified_exact_landing_xyz(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    episode = assembly.factory(0, "HOPPER").episode
    snapshot = episode._snapshot

    assert snapshot is not None
    physical = snapshot.physical_reachability
    cells = tuple(zip(*np.nonzero(physical.physical_observation_pose_mask), strict=True))
    exact_by_cell = dict(
        zip(cells, physical.observation_positions_m, strict=True)
    )
    selected_positions = snapshot.candidates.target_positions_m[
        snapshot.candidates.mask
    ]
    assert len(selected_positions) > 0
    for position in selected_positions:
        cell = episode.loaded.scene.base_canvas.world_to_grid(
            float(position[0]), float(position[1])
        )
        np.testing.assert_array_equal(position, exact_by_cell[cell])


def test_hopper_start_qualification_reprojects_from_certified_exact_start(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, bundle, scene_id = _cache(tmp_path)
    loaded = formal_builder_module._load_multires_scene(
        load_formal_cache(manifest), scene_id
    )
    original_project = (
        qualification_module.PlatformCandidateReachability.project_physical
    )
    calls: list[tuple[Pose2, tuple[float, float, float]]] = []
    certified_start: tuple[float, float, float] | None = None

    def certify_offset_start(self, *args, **kwargs):
        nonlocal certified_start
        state_position = self._request.current_state.pose.position_m
        calls.append(
            (
                self._pose,
                (
                    float(state_position.x),
                    float(state_position.y),
                    float(state_position.z),
                ),
            )
        )
        result = original_project(self, *args, **kwargs)
        authority = result.hopper_opportunity_authority
        assert authority is not None
        start_cell = self._canvas.world_to_grid(
            self._pose.x_m, self._pose.y_m
        )
        cells = tuple(
            zip(*np.nonzero(authority.certified_mask), strict=True)
        )
        assert start_cell in cells
        positions = authority.certified_positions_m.copy()
        index = cells.index(start_cell)
        if len(calls) != 1:
            if certified_start is None:
                return result
            positions[index] = certified_start
            authority.certified_positions_m = np.ascontiguousarray(positions)
            return result
        positions[index] += np.asarray((0.02, -0.02, 1.25))
        certified_start = tuple(float(value) for value in positions[index])
        authority.certified_positions_m = np.ascontiguousarray(positions)
        return result

    monkeypatch.setattr(
        qualification_module.PlatformCandidateReachability,
        "project_physical",
        certify_offset_start,
    )

    qualified = qualification_module.qualify_initial_start(
        scene=loaded.scene,
        arrays=loaded.arrays,
        capability=bundle.for_platform("HOPPER"),
    )

    assert qualified is not None
    assert certified_start is not None
    assert len(calls) >= 2
    assert (
        calls[1][0].x_m,
        calls[1][0].y_m,
        calls[1][0].elevation_m,
    ) == certified_start
    assert calls[1][1] == certified_start


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
    original_build = CandidateBuilderV2.build_physical_universe
    platform_calls: list[str] = []

    def record_platform(self, *args, **kwargs):
        platform_calls.append(kwargs["platform_type"])
        return original_build(self, *args, **kwargs)

    monkeypatch.setattr(
        CandidateBuilderV2, "build_physical_universe", record_platform
    )
    worker = assembly.factory(0, "HOPPER")

    assert platform_calls
    assert set(platform_calls) == {"HOPPER"}
    diagnostics = worker.episode.current_candidate_diagnostics()
    assert worker.current_candidate_diagnostics() == diagnostics
    assert diagnostics.selected_policy_candidate_count == int(
        worker.environment.current_observation.candidate_mask.sum()
    )
    assert len(diagnostics.physical_snapshot_id) == 64
    assert diagnostics.physical_candidate_universe_count >= (
        diagnostics.selected_policy_candidate_count
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
    original_project = qualification_module.PlatformCandidateReachability.project_physical
    calls = 0

    def reject_first_start(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("WHEEL_START_NOT_SAFE")
        return original_project(self, *args, **kwargs)

    monkeypatch.setattr(
        qualification_module.PlatformCandidateReachability,
        "project_physical",
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

    def fail_projection(*_args, **_kwargs):
        raise RuntimeError("PHYSICAL_PROJECTION_FAILED")

    monkeypatch.setattr(
        qualification_module.PlatformCandidateReachability,
        "project_physical",
        fail_projection,
    )

    with pytest.raises(RuntimeError, match="PHYSICAL_PROJECTION_FAILED"):
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
    assert worker.episode._current_candidate_gain_resolution_m == 0.2


def test_formal_v6_snapshot_records_physical_identity(
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
    state = worker.snapshot_episode_state()
    snapshot = worker.episode._snapshot

    assert snapshot is not None
    assert state["physical_snapshot_id"] == (
        snapshot.candidate_universe.physical_snapshot_id
    )
    assert state["physical_candidate_universe_sha256"] == (
        snapshot.candidate_universe_sha256
    )
    assert state["candidate_gain_resolution_m"] == 0.2


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
def test_formal_v5_restore_fails_closed_for_all_platforms(
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
    legacy_state = worker.snapshot_episode_state()
    legacy_state.pop("physical_snapshot_id")

    with pytest.raises(ValueError, match="worker state structure"):
        assembly.factory.restore_for_episode(
            worker_index=0,
            platform_type=platform,
            episode_cursor=4,
            platform_worker_index=0,
            platform_worker_count=1,
            state=legacy_state,
        )


def test_formal_restore_rejects_malformed_state_before_episode_replay(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)

    with pytest.raises(ValueError, match="worker state structure"):
        assembly.factory.restore_for_episode(
            worker_index=0,
            platform_type="WHEELED",
            episode_cursor=4,
            platform_worker_index=0,
            platform_worker_count=1,
            state={},
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
    assert prepared.candidate_id == str(
        worker.episode._snapshot.candidates.candidate_ids[candidate]
    )
    assert prepared.physical_snapshot_id == (
        worker.episode._snapshot.candidate_universe.physical_snapshot_id
    )
    assert identity.map_snapshot_id in request.request_id
    assert request.capability_version == bundle.for_platform("WHEELED").capability_version
    assert observation.prior_channels.shape[-2:] == (256, 256)
    assert request.world.global_map.resolution_m == 4.0
    assert request.world.global_map.width == 256
    assert request.world.local_map.resolution_m == 0.2
    assert request.world.local_map.width == 320
    assert request.config.local_frontier.additional_corridor_margin_m == 2.0
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

    first_prepared = episode.begin_ground_option(action, identity)
    first = first_prepared.request
    first_goal = first.goal.target.position_m
    frozen = (first_goal.x, first_goal.y, first_goal.z, first.goal.goal_id)
    snapshot = episode._snapshot
    assert snapshot is not None
    refreshed_features = snapshot.candidates.features.copy()
    refreshed_features[candidate, :2] = (0.01, 0.99)
    episode._snapshot = replace(
        snapshot,
        candidates=CandidateBatch(
            features=refreshed_features,
            mask=snapshot.candidates.mask.copy(),
            canvas_id=snapshot.candidates.canvas_id,
            diagnostics=snapshot.candidates.diagnostics,
            target_elevation_m=(
                snapshot.candidates.target_elevation_m.copy()
            ),
            target_positions_m=snapshot.candidates.target_positions_m.copy(),
            target_yaw_rad=snapshot.candidates.target_yaw_rad.copy(),
            candidate_ids=snapshot.candidates.candidate_ids.copy(),
        ),
    )

    continued_prepared = episode.continue_ground_option(identity)
    continued = continued_prepared.request
    continued_goal = continued.goal.target.position_m

    assert (
        continued_goal.x,
        continued_goal.y,
        continued_goal.z,
        continued.goal.goal_id,
    ) == frozen
    assert continued_prepared.candidate_id == first_prepared.candidate_id
    with pytest.raises(ValueError, match="active ground option"):
        worker.snapshot_episode_state()
    episode.clear_ground_option()
    state = worker.snapshot_episode_state()
    assert state["physical_snapshot_id"] == (
        episode._snapshot.candidate_universe.physical_snapshot_id
    )
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
        def build_physical_universe(self, *args, **kwargs):
            nonlocal candidate_builds
            candidate_builds += 1
            return original_builder.build_physical_universe(*args, **kwargs)

        def select_available(self, *args, **kwargs):
            return original_builder.select_available(*args, **kwargs)

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


def test_formal_worker_exposes_physical_candidate_diagnostics(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    diagnostics = assembly.factory(
        0, "WHEELED"
    ).episode.current_candidate_diagnostics()

    assert diagnostics.physical_candidate_universe_count > 0
    assert diagnostics.selected_policy_candidate_count > 0
    assert diagnostics.planner_failed_current_snapshot_count == 0
    assert not hasattr(diagnostics, "planner_rejected_count")


def test_planning_failure_refresh_suppresses_stable_id_without_new_evidence(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if suppression mutated the universe or failed to promote reserve."""
    assembly, _, _ = _assembly(tmp_path)
    worker = assembly.factory(0, "WHEELED")
    episode = worker.episode
    before = episode.controller.current_observation
    before_identity = before.observation_identities[0]
    snapshot = episode._snapshot
    assert snapshot is not None
    before_ids = tuple(
        str(snapshot.candidates.candidate_ids[index])
        for index in np.flatnonzero(snapshot.candidates.mask)
    )
    failed_id = before_ids[0]
    before_universe_sha256 = snapshot.candidate_universe_sha256
    before_physical_snapshot_id = (
        snapshot.candidate_universe.physical_snapshot_id
    )
    before_evidence_generation = episode.sensor_state.evidence_generation
    before_evidence_sha256 = episode.sensor_state.physical_evidence_sha256()
    before_pose = episode.current_pose
    before_ratio = float(before.pose_features[0, 4].item())
    candidate_index = int(np.flatnonzero(snapshot.candidates.mask)[0])
    episode.begin_ground_option(
        PolicyAction(candidate_index, 0.0), before_identity
    )
    episode._defer_candidate_rebuild = True

    refreshed = episode.refresh_after_planning_failure(
        failed_id,
        CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT,
        physical_snapshot_id=before_physical_snapshot_id,
    )

    after_snapshot = episode._snapshot
    assert after_snapshot is not None
    after_ids = tuple(
        str(after_snapshot.candidates.candidate_ids[index])
        for index in np.flatnonzero(after_snapshot.candidates.mask)
    )
    after_identity = refreshed.next_observation.observation_identities[0]
    assert episode._active_ground_option is None
    assert episode._defer_candidate_rebuild is False
    assert episode.current_pose == before_pose
    assert episode.sensor_state.evidence_generation == before_evidence_generation
    assert episode.sensor_state.physical_evidence_sha256() == before_evidence_sha256
    assert after_identity.state_time_ns == before_identity.state_time_ns
    assert float(refreshed.next_observation.pose_features[0, 4]) == before_ratio
    assert refreshed.mission_observed_delta == 0.0
    assert refreshed.priority_observed_delta == 0.0
    assert after_snapshot.candidate_universe_sha256 == before_universe_sha256
    assert (
        after_snapshot.candidate_universe.physical_snapshot_id
        == before_physical_snapshot_id
    )
    assert failed_id not in after_ids
    assert len(after_ids) == len(before_ids)
    assert set(after_ids) - set(before_ids)
    assert after_snapshot.candidates.diagnostics.planner_failed_current_snapshot_count == 1
    assert after_identity.map_snapshot_id != before_identity.map_snapshot_id
    assert after_identity.candidate_set_id != before_identity.candidate_set_id


def test_boundary_window_refills_one_reserve_before_exhaustion(
    tmp_path: pathlib.Path,
) -> None:
    assembly, _, _ = _assembly(tmp_path)
    episode = assembly.factory(0, "WHEELED").episode
    snapshot = episode._snapshot
    assert snapshot is not None
    assert snapshot.frontier_oracle.oracle_opportunity_count == 146
    candidates = snapshot.candidate_universe.candidates[:26]
    assert len(candidates) == 26
    diagnostics = replace(
        snapshot.candidate_universe.diagnostics,
        physical_candidate_universe_count=26,
        selected_policy_candidate_count=26,
        available_candidate_count=26,
        untried_reserve_count=0,
    )
    universe = PhysicalCandidateUniverse(
        physical_snapshot_id=(
            snapshot.candidate_universe.physical_snapshot_id
        ),
        physical_reachability_algorithm_id=(
            snapshot.candidate_universe.physical_reachability_algorithm_id
        ),
        candidates=candidates,
        universe_sha256=hashlib.sha256(
            "".join(candidate.candidate_id for candidate in candidates).encode(
                "ascii"
            )
        ).hexdigest(),
        diagnostics=diagnostics,
    )
    failed_ids = {candidate.candidate_id for candidate in candidates[:25]}
    assert len(candidates) - len(failed_ids) == 1
    assert snapshot.candidates.canvas_id is not None

    refilled = episode._candidate_builder.select_available(
        universe,
        canvas_id=snapshot.candidates.canvas_id,
        failure_snapshot_id=universe.physical_snapshot_id,
        planner_failed_candidate_ids=failed_ids,
    )

    assert refilled.batch.count == 1
    assert tuple(refilled.batch.candidate_ids[refilled.batch.mask]) == (
        candidates[-1].candidate_id,
    )
    assert refilled.batch.diagnostics.physical_candidate_universe_count == 26
    assert refilled.batch.diagnostics.planner_failed_current_snapshot_count == 25
    assert refilled.batch.diagnostics.available_candidate_count == 1
    assert refilled.batch.diagnostics.untried_reserve_count == 0

    exhausted = episode._candidate_builder.select_available(
        universe,
        canvas_id=snapshot.candidates.canvas_id,
        failure_snapshot_id=universe.physical_snapshot_id,
        planner_failed_candidate_ids={
            candidate.candidate_id for candidate in candidates
        },
    )

    assert exhausted.batch.count == 0
    assert exhausted.batch.diagnostics.physical_candidate_universe_count == 26
    assert exhausted.batch.diagnostics.planner_failed_current_snapshot_count == 26
    assert exhausted.batch.diagnostics.available_candidate_count == 0
    assert exhausted.batch.diagnostics.untried_reserve_count == 0


def test_planning_failure_snapshot_mismatch_fails_before_refresh_state_changes(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if a stale request could suppress the current snapshot."""
    assembly, _, _ = _assembly(tmp_path)
    episode = assembly.factory(0, "WHEELED").episode
    snapshot = episode._snapshot
    assert snapshot is not None
    candidate_index = int(np.flatnonzero(snapshot.candidates.mask)[0])
    candidate_id = str(snapshot.candidates.candidate_ids[candidate_index])
    before = episode.controller.current_observation
    before_failures = set(episode._planner_failed_candidate_ids)
    before_pending = episode._pending_planning_failure
    before_generation = episode.sensor_state.evidence_generation
    before_evidence = episode.sensor_state.physical_evidence_sha256()

    with pytest.raises(ValueError, match="physical snapshot"):
        episode.refresh_after_planning_failure(
            candidate_id,
            CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT,
            physical_snapshot_id="0" * 64,
        )

    after = episode.controller.current_observation
    assert after.observation_identities == before.observation_identities
    assert all(
        torch.equal(getattr(after, name), getattr(before, name))
        for name in before.input_names
    )
    assert episode._planner_failed_candidate_ids == before_failures
    assert episode._pending_planning_failure == before_pending
    assert episode.sensor_state.evidence_generation == before_generation
    assert episode.sensor_state.physical_evidence_sha256() == before_evidence


def test_rolling_continuation_snapshot_tracks_latest_pose_and_evidence(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if continuation attested the initial physical snapshot."""
    assembly, _, _ = _assembly(tmp_path)
    episode = assembly.factory(0, "WHEELED").episode
    initial_observation = episode.controller.current_observation
    initial_identity = initial_observation.observation_identities[0]
    snapshot = episode._snapshot
    assert snapshot is not None
    candidate_index = int(np.flatnonzero(snapshot.candidates.mask)[2])
    initial_prepared = episode.begin_ground_option(
        PolicyAction(candidate_index, 0.0), initial_identity
    )
    target = initial_prepared.request.goal.target.position_m
    start = episode.current_pose
    moved = Pose2(
        start.x_m + 0.1 * (float(target.x) - start.x_m),
        start.y_m + 0.1 * (float(target.y) - start.y_m),
        start.yaw_rad,
        "map",
        start.elevation_m
        + 0.1 * (float(target.z) - start.elevation_m),
    )
    evidence = SensorBoundaryEvidence(moved, 1.0)
    episode.current_pose = moved
    episode._record_reveal(evidence, "DECISION_BOUNDARY")
    boundary = episode.controller.after_execution(
        platform_type="WHEELED",
        execution_state="DECISION_BOUNDARY",
        evidence=evidence,
    )
    continued = episode.continue_ground_option(
        boundary.next_observation.observation_identities[0]
    )

    assert continued.candidate_id == initial_prepared.candidate_id
    assert (
        continued.physical_snapshot_id
        != initial_prepared.physical_snapshot_id
    )
    refreshed = episode.refresh_after_planning_failure(
        continued.candidate_id,
        CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT,
        physical_snapshot_id=continued.physical_snapshot_id,
    )
    final_snapshot = episode._snapshot
    assert final_snapshot is not None
    assert (
        final_snapshot.candidate_universe.physical_snapshot_id
        == continued.physical_snapshot_id
    )
    assert refreshed.next_observation.observation_identities[0].state_time_ns == (
        boundary.next_observation.observation_identities[0].state_time_ns
    )


def test_parallel_consumer_has_no_legacy_planner_rejection_counter() -> None:
    consumer_source = inspect.getsource(parallel_pool_module._worker_main)

    assert "planner_rejected_count" not in consumer_source
    assert "rejected_candidate_indices" not in consumer_source
    with pytest.raises(TypeError, match="planner_rejected_count"):
        replace(CandidateDiagnostics(), planner_rejected_count=0)

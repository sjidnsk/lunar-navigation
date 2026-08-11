from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import pathlib
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import lunar_policy_training.polar_data.formal_cache as formal_cache_module

from lunar_policy_training.environment.coverability import (
    IneligibleReason,
    PlatformCoverability,
    mask_sha256,
    pack_detail_mask,
)
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


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _physical_projection_sha256(
    platform_type: str,
    mask: np.ndarray,
    algorithm_id: str,
    capability_content_sha256: str,
    start_identity_sha256: str,
) -> str:
    body = {
        "platform_type": platform_type,
        "physical_reachability_algorithm_id": algorithm_id,
        "physical_observation_pose_shape": list(mask.shape),
        "physical_observation_pose_mask_sha256": mask_sha256(mask),
        "capability_content_sha256": capability_content_sha256,
        "start_identity_sha256": start_identity_sha256,
    }
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


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
    coverability: dict[str, PlatformCoverability] = {}
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
        capability_sha256 = _sha(f"capability-content/{platform}")
        start_sha256 = _sha(
            f"physical-start/{platform}/{start if start is not None else 'unsafe'}"
        )
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
            physical_projection_sha256=_physical_projection_sha256(
                platform,
                reachable,
                algorithm_id,
                capability_sha256,
                start_sha256,
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
        world_bounds_m=(0.0, 0.0, 1024.0, 1024.0),
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
    )

    class Bridge:
        def project_reachability(
            self, request_value: object, distance: float, evidence_value: object
        ) -> object:
            calls.append((request_value, distance, evidence_value))
            return output

    monkeypatch.setattr(
        formal_cache_module,
        "_projection_request",
        lambda *_args, **_kwargs: request,
    )
    monkeypatch.setattr(
        formal_cache_module,
        "_hopper_landing_evidence",
        lambda **_kwargs: evidence,
    )

    result = formal_cache_module._build_truth_physical_reachability(
        platform=SimpleNamespace(
            platform_type="HOPPER", content_sha256=_sha("hopper-capability")
        ),
        scene=object(),
        projected=object(),
        start_cell=(127, 128),
        bridge=Bridge(),
    )

    assert calls == [(request, 30.0, evidence_grid)]
    assert result.physical_reachability_algorithm_id == output.algorithm_id
    assert result.physical_safe_pose_count == 1
    assert result.physically_reachable_pose_count == 1
    assert result.physical_observation_pose_mask[127, 128]
    np.testing.assert_array_equal(
        result.observation_positions_m,
        np.asarray([[514.0, 510.0, 1.0]], dtype=np.float64),
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

    result = formal_cache_module._build_scene_platform_coverability(
        platform_type="HOPPER",
        capability_bundle=CapabilityBundle(),
        qualification=SimpleNamespace(cell=(3, 4), initial_candidate_count=1),
        scene=object(),
        projected=object(),
        mission_roi=np.ones((2, 2), dtype=np.bool_),
        detail_shape=(256, 256),
    )

    assert result.eligible is False
    assert result.ineligible_reason is IneligibleReason.UNSAFE_START
    assert result.qualified_start_cell is None


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
                cell=(3, 4), initial_candidate_count=1
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
    assert not any("primitive" in field for field in platform)
    assert platform["eligible"] is True
    assert platform["ineligible_reason"] is None
    assert platform["exact"] is True
    assert cache.formal_eligible is False
    assert arrays["elevation_m"].shape == (256, 256)
    assert arrays["wheeled_hard_feasible"].dtype == np.uint8
    assert arrays["wheeled_physical_observation_pose_bits"].dtype == np.uint8
    assert arrays["wheeled_coverable_detail_bits"].dtype == np.uint8
    assert arrays["wheeled_coverable_ratio"].dtype == np.float32
    with pytest.raises(FormalCacheError, match="full"):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=identity,
            require_full=True,
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

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import pathlib

import numpy as np
import pytest

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
    _formal_platform_eligibility_ready,
    load_formal_cache,
    platform_scenario_schedule_id,
    write_formal_cache,
)


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


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
        coverability[platform] = PlatformCoverability(
            platform_type=platform,
            qualified_start_cell=start,
            reachable_pose_mask=reachable,
            coverable_detail_shape=detail.shape,
            coverable_detail_bits=pack_detail_mask(detail),
            coverable_ratio=detail.astype(np.float32),
            mission_target_detail_cell_count=detail.size,
            coverable_detail_cell_count=count,
            mission_coverable_fraction=count / detail.size,
            initial_coverable_fraction=0.1,
            initial_candidate_count=1,
            reachability_algorithm_id=f"test-reachability/{platform.lower()}",
            visibility_algorithm_id="two-dimensional-detail-los/v1",
            reachable_mask_sha256=mask_sha256(reachable),
            coverable_mask_sha256=mask_sha256(detail),
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

    assert FORMAL_CACHE_SCHEMA == "lunar-formal-training-cache/v4"
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
    assert platform["eligible"] is True
    assert platform["ineligible_reason"] is None
    assert platform["exact"] is True
    assert cache.formal_eligible is False
    assert arrays["elevation_m"].shape == (256, 256)
    assert arrays["wheeled_hard_feasible"].dtype == np.uint8
    assert arrays["wheeled_reachable_pose_bits"].dtype == np.uint8
    assert arrays["wheeled_coverable_detail_bits"].dtype == np.uint8
    assert arrays["wheeled_coverable_ratio"].dtype == np.float32
    with pytest.raises(FormalCacheError, match="full"):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=identity,
            require_full=True,
        )


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

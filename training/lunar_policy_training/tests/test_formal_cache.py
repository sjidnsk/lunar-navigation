from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import pathlib

import numpy as np
import pytest

from lunar_policy_training.polar_data.formal_cache import (
    FormalCacheError,
    FormalCacheIdentity,
    StaticSceneData,
    _formal_start_qualification_ready,
    load_formal_cache,
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
        qualified_start_cells=(
            {
                "WHEELED": (127, 127),
                "LEGGED": (127, 127),
                "HOPPER": (127, 127),
            }
            if qualified_start_cells is None
            else qualified_start_cells
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

    assert manifest["materialization"] == "preflight"
    assert manifest["start_eligible_scene_count"] == 1
    assert manifest["start_eligible_split_counts"] == {"train": 1}
    assert manifest["scenes"][0]["start_qualification"] == {
        "common_eligible": True,
        "platform_start_cells": {
            "HOPPER": [127, 127],
            "LEGGED": [127, 127],
            "WHEELED": [127, 127],
        },
    }
    assert cache.formal_eligible is False
    assert arrays["elevation_m"].shape == (256, 256)
    assert arrays["wheeled_hard_feasible"].dtype == np.uint8
    with pytest.raises(FormalCacheError, match="full"):
        load_formal_cache(
            root / "cache-manifest.json",
            expected_identity=identity,
            require_full=True,
        )


def test_cache_records_a_scene_as_not_common_start_eligible(
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
                qualified_start_cells={
                    "WHEELED": (127, 127),
                    "LEGGED": (127, 127),
                    "HOPPER": None,
                },
            ),
        ),
        repository_root=REPOSITORY_ROOT,
    )

    assert manifest["start_eligible_scene_count"] == 0
    assert manifest["start_eligible_split_counts"] == {"train": 0}
    assert manifest["scenes"][0]["start_qualification"] == {
        "common_eligible": False,
        "platform_start_cells": {
            "HOPPER": None,
            "LEGGED": [127, 127],
            "WHEELED": [127, 127],
        },
    }


def test_formal_start_qualification_uses_a_common_subset_without_diluting_holdout(
) -> None:
    totals = {"train": 1536, "validation": 96, "test": 96, "holdout": 6}

    assert _formal_start_qualification_ready(
        "full",
        totals,
        {"train": 1475, "validation": 90, "test": 95, "holdout": 6},
    )
    assert not _formal_start_qualification_ready(
        "preflight",
        totals,
        {"train": 1475, "validation": 90, "test": 95, "holdout": 6},
    )
    assert not _formal_start_qualification_ready(
        "full",
        totals,
        {"train": 1475, "validation": 86, "test": 95, "holdout": 6},
    )
    assert not _formal_start_qualification_ready(
        "full",
        totals,
        {"train": 1475, "validation": 90, "test": 95, "holdout": 5},
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

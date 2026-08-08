from __future__ import annotations

import importlib
import json
import pathlib

import pytest


def _api():
    try:
        module = importlib.import_module(
            "lunar_policy_training.polar_data.scenario_manifest"
        )
        return module.build_scenario_manifest_document, module.ScenarioManifestError
    except (ImportError, AttributeError) as error:
        pytest.fail(f"formal scenario manifest API is missing: {error}")


def _io_api():
    try:
        module = importlib.import_module(
            "lunar_policy_training.polar_data.scenario_manifest"
        )
        return module.write_scenario_manifest, module.load_scenario_manifest
    except (ImportError, AttributeError) as error:
        pytest.fail(f"formal scenario manifest I/O API is missing: {error}")


def _split_document() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    assignments = (
        ("train", 192),
        ("validation", 48),
        ("test", 48),
    )
    index = 0
    for split, count in assignments:
        for _ in range(count):
            window_id = f"nasa-window-{index:03d}"
            rows.append(
                {
                    "source": "NASA_LOLA",
                    "split": split,
                    "window_id": window_id,
                    "window_sha256": f"{index + 1:064x}",
                    "world_bounds_m": [
                        float(index * 2048),
                        0.0,
                        float(index * 2048 + 1024),
                        1024.0,
                    ],
                    "valid_fraction": 1.0,
                    "split_sha256": "5" * 64,
                }
            )
            index += 1
    for site in ("CR1", "GR1", "GR2", "LP1", "MP1", "MP2"):
        rows.append(
            {
                "source": "JAXA_LUPEX",
                "split": "holdout",
                "window_id": site,
                "window_sha256": None,
                "world_bounds_m": None,
                "archive_sha256": "c" * 64,
                "archive_member_paths": [
                    f"DTMs/{site.lower()}_roi_sfs_1m-DEM.tif",
                    f"orthomosaics/{site.lower()}_roi.tif",
                    f"uncertainties/{site.lower()}_height_error.tif",
                ],
                "archive_member_sha256s": ["d" * 64, "e" * 64, "f" * 64],
                "split_sha256": "5" * 64,
            }
        )
    return {
        "schema": "lunar-polar-split-manifest/v2",
        "seed": 4080,
        "split_sha256": "5" * 64,
        "sources": [
            {"id": "NASA_LOLA_87S_DEM", "sha256": "1" * 64},
            {"id": "NASA_LOLA_87S_COUNT", "sha256": "2" * 64},
            {"id": "JAXA_LUPEX_DATA_S1", "sha256": "3" * 64},
        ],
        "rows": rows,
    }


def _build(document: dict[str, object] | None = None) -> dict[str, object]:
    build, _ = _api()
    return build(
        split_document=_split_document() if document is None else document,
        source_lock_file_sha256="a" * 64,
        split_manifest_file_sha256="b" * 64,
        source_sha256s={
            "NASA_LOLA_87S_DEM": "1" * 64,
            "NASA_LOLA_87S_COUNT": "2" * 64,
            "JAXA_LUPEX_DATA_S1": "3" * 64,
        },
    )


def test_polar_data_package_exports_formal_scenario_api_lazily() -> None:
    package = importlib.import_module("lunar_policy_training.polar_data")

    assert package.SCENARIO_MANIFEST_SCHEMA == "lunar-formal-scenario-manifest/v1"
    assert callable(package.build_scenario_manifest_document)
    assert callable(package.load_scenario_manifest)


def test_formal_catalog_expands_exact_frozen_scene_families() -> None:
    """Catches a seed/count change that would silently alter training data."""
    manifest = _build()

    assert manifest["schema"] == "lunar-formal-scenario-manifest/v1"
    assert manifest["generator_version"] == "lunar-polar-multires-hazards/v2"
    assert manifest["object_distribution"] == {
        "rocks": {
            "count": 260,
            "radius_m": [0.15, 1.2],
            "height_to_radius": [0.6, 1.5],
        },
        "craters": {
            "count": 32,
            "radius_m": [2.0, 16.0],
            "depth_m": [0.05, 1.0],
        },
        "no_go_polygons": {
            "count": 8,
            "circumradius_m": [2.0, 8.0],
            "vertices": 6,
        },
    }
    assert manifest["scenario_counts"] == {
        "train": 1536,
        "validation": 96,
        "test": 96,
        "holdout": 6,
    }
    assert len(manifest["scenarios"]) == 1734
    assert len({item["scene_id"] for item in manifest["scenarios"]}) == 1734

    first_window = [
        item
        for item in manifest["scenarios"]
        if item["window_id"] == "nasa-window-000"
    ]
    assert [item["scenario_seed"] for item in first_window] == list(
        range(408000, 408008)
    )
    validation = next(
        item
        for item in manifest["scenarios"]
        if item["split"] == "validation"
    )
    test = next(item for item in manifest["scenarios"] if item["split"] == "test")
    assert validation["scenario_seed"] == 409000
    assert test["scenario_seed"] == 410000


def test_jaxa_holdout_has_no_training_overlay() -> None:
    """Catches accidental random-overlay contamination of the real holdout."""
    holdout = [item for item in _build()["scenarios"] if item["split"] == "holdout"]

    assert [item["window_id"] for item in holdout] == [
        "CR1",
        "GR1",
        "GR2",
        "LP1",
        "MP1",
        "MP2",
    ]
    assert all(item["scenario_seed"] is None for item in holdout)
    assert all(item["procedural_overlay"] is False for item in holdout)


def test_manifest_identity_is_independent_of_input_row_order() -> None:
    """Catches nondeterministic JSON identity caused by source row ordering."""
    original = _split_document()
    reversed_document = {**original, "rows": list(reversed(original["rows"]))}

    assert _build(original) == _build(reversed_document)


def test_manifest_rejects_a_split_with_missing_nasa_window() -> None:
    """Catches a partial split being mislabeled as the formal scenario bank."""
    document = _split_document()
    document["rows"] = document["rows"][1:]
    _, error = _api()

    with pytest.raises(error, match="192.*48.*48"):
        _build(document)


def test_manifest_round_trip_revalidates_canonical_identity(
    tmp_path: pathlib.Path,
) -> None:
    """Catches a loader trusting a stale or hand-edited manifest digest."""
    write, load = _io_api()
    target = tmp_path / "formal-scenarios.json"
    expected = _build()

    written_sha = write(target, expected, repository_root=pathlib.Path.cwd())
    loaded = load(target)

    assert loaded == expected
    assert written_sha == expected["scenario_manifest_sha256"]
    assert target.read_bytes().endswith(b"\n")

    changed = json.loads(target.read_text(encoding="utf-8"))
    changed["scenarios"][0]["scenario_seed"] = 123
    target.write_text(json.dumps(changed), encoding="utf-8")
    _, error = _api()
    with pytest.raises(error, match="identity"):
        load(target)

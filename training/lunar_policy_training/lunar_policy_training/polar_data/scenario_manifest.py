"""Canonical formal NASA/JAXA scene-bank identity."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from .hazards import FORMAL_GENERATOR_VERSION, formal_hazard_distribution


SCENARIO_MANIFEST_SCHEMA = "lunar-formal-scenario-manifest/v1"
TRAIN_SCENARIO_SEEDS = tuple(range(408000, 408008))
VALIDATION_SCENARIO_SEEDS = tuple(range(409000, 409002))
TEST_SCENARIO_SEEDS = tuple(range(410000, 410002))
_NASA_COUNTS = {"train": 192, "validation": 48, "test": 48}
_JAXA_SITES = ("CR1", "GR1", "GR2", "LP1", "MP1", "MP2")
_SOURCE_IDS = (
    "NASA_LOLA_87S_DEM",
    "NASA_LOLA_87S_COUNT",
    "JAXA_LUPEX_DATA_S1",
)
_SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2, "holdout": 3}


class ScenarioManifestError(ValueError):
    """A split cannot identify the approved formal scene bank."""


def _require_sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ScenarioManifestError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _row_key(row: Mapping[str, object]) -> tuple[int, str, str]:
    split = row.get("split")
    source = row.get("source")
    window_id = row.get("window_id")
    if split not in _SPLIT_ORDER:
        raise ScenarioManifestError("split row has an unsupported split")
    if not isinstance(source, str) or not isinstance(window_id, str) or not window_id:
        raise ScenarioManifestError("split row source/window identity is invalid")
    return _SPLIT_ORDER[split], source, window_id


def _validated_rows(split_document: Mapping[str, object]) -> list[Mapping[str, object]]:
    if split_document.get("schema") != "lunar-polar-split-manifest/v1":
        raise ScenarioManifestError("split manifest schema is unsupported")
    if split_document.get("seed") != 4080:
        raise ScenarioManifestError("formal split seed must be 4080")
    split_sha = _require_sha(split_document.get("split_sha256"), "split")
    rows = split_document.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ScenarioManifestError("split rows must be a JSON array of objects")
    if any(row.get("split_sha256") != split_sha for row in rows):
        raise ScenarioManifestError("split row identity differs from manifest")

    nasa = Counter(
        row.get("split") for row in rows if row.get("source") == "NASA_LOLA"
    )
    if nasa != Counter(_NASA_COUNTS):
        raise ScenarioManifestError(
            "formal NASA split must contain exactly 192 train, 48 validation and 48 test windows"
        )
    holdout = sorted(
        row.get("window_id")
        for row in rows
        if row.get("source") == "JAXA_LUPEX" and row.get("split") == "holdout"
    )
    if holdout != sorted(_JAXA_SITES):
        raise ScenarioManifestError("formal JAXA holdout must contain the exact six sites")
    if len(rows) != sum(_NASA_COUNTS.values()) + len(_JAXA_SITES):
        raise ScenarioManifestError("formal split contains unexpected rows")
    return sorted(rows, key=_row_key)


def _nasa_scenarios(row: Mapping[str, object]) -> list[dict[str, object]]:
    split = str(row["split"])
    seeds = {
        "train": TRAIN_SCENARIO_SEEDS,
        "validation": VALIDATION_SCENARIO_SEEDS,
        "test": TEST_SCENARIO_SEEDS,
    }[split]
    window_sha = _require_sha(row.get("window_sha256"), "NASA window")
    bounds = row.get("world_bounds_m")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 4
        or any(not isinstance(value, (int, float)) for value in bounds)
    ):
        raise ScenarioManifestError("NASA world bounds are invalid")
    output: list[dict[str, object]] = []
    for scenario_seed in seeds:
        generated_seed = sha256(
            f"{window_sha}:{scenario_seed}:{FORMAL_GENERATOR_VERSION}".encode(
                "utf-8"
            )
        ).hexdigest()
        identity = {
            "source": "NASA_LOLA",
            "split": split,
            "window_id": row["window_id"],
            "window_sha256": window_sha,
            "scenario_seed": scenario_seed,
            "scene_seed": generated_seed,
            "procedural_overlay": True,
            "world_bounds_m": [float(value) for value in bounds],
            "archive_sha256": None,
            "archive_member_paths": None,
            "archive_member_sha256s": None,
        }
        output.append({"scene_id": _canonical_sha256(identity), **identity})
    return output


def _jaxa_scenario(row: Mapping[str, object]) -> dict[str, object]:
    archive_sha = _require_sha(row.get("archive_sha256"), "JAXA archive")
    paths = row.get("archive_member_paths")
    hashes = row.get("archive_member_sha256s")
    if (
        not isinstance(paths, list)
        or not isinstance(hashes, list)
        or len(paths) != 3
        or len(hashes) != 3
        or any(not isinstance(path, str) or not path for path in paths)
    ):
        raise ScenarioManifestError("JAXA member inventory is invalid")
    checked_hashes = [_require_sha(value, "JAXA member") for value in hashes]
    identity = {
        "source": "JAXA_LUPEX",
        "split": "holdout",
        "window_id": row["window_id"],
        "window_sha256": None,
        "scenario_seed": None,
        "scene_seed": _canonical_sha256(
            {
                "archive_sha256": archive_sha,
                "site": row["window_id"],
                "members": list(zip(paths, checked_hashes, strict=True)),
            }
        ),
        "procedural_overlay": False,
        "world_bounds_m": None,
        "archive_sha256": archive_sha,
        "archive_member_paths": list(paths),
        "archive_member_sha256s": checked_hashes,
    }
    return {"scene_id": _canonical_sha256(identity), **identity}


def build_scenario_manifest_document(
    *,
    split_document: Mapping[str, object],
    source_lock_file_sha256: str,
    split_manifest_file_sha256: str,
    source_sha256s: Mapping[str, str],
) -> dict[str, object]:
    """Expand a verified split into the exact formal scene catalogue."""
    if not isinstance(split_document, Mapping):
        raise ScenarioManifestError("split manifest must be an object")
    source_lock_sha = _require_sha(source_lock_file_sha256, "source lock file")
    split_file_sha = _require_sha(split_manifest_file_sha256, "split manifest file")
    if set(source_sha256s) != set(_SOURCE_IDS):
        raise ScenarioManifestError("formal source identity requires exactly three sources")
    checked_sources = {
        source_id: _require_sha(source_sha256s[source_id], source_id)
        for source_id in _SOURCE_IDS
    }
    split_sources = split_document.get("sources")
    if not isinstance(split_sources, list):
        raise ScenarioManifestError("split source inventory is missing")
    split_source_map = {
        item.get("id"): item.get("sha256")
        for item in split_sources
        if isinstance(item, Mapping)
    }
    if split_source_map != checked_sources:
        raise ScenarioManifestError("split source hashes differ from the source lock")

    scenarios: list[dict[str, object]] = []
    for row in _validated_rows(split_document):
        if row["source"] == "NASA_LOLA":
            scenarios.extend(_nasa_scenarios(row))
        else:
            scenarios.append(_jaxa_scenario(row))
    counts = Counter(item["split"] for item in scenarios)
    payload: dict[str, object] = {
        "schema": SCENARIO_MANIFEST_SCHEMA,
        "source_lock_file_sha256": source_lock_sha,
        "source_sha256s": checked_sources,
        "split_manifest_file_sha256": split_file_sha,
        "split_sha256": _require_sha(split_document.get("split_sha256"), "split"),
        "generator_version": FORMAL_GENERATOR_VERSION,
        "object_distribution": formal_hazard_distribution(),
        "geometry": {
            "global": {"size_m": 1024.0, "resolution_m": 4.0, "cells": 256},
            "local_tile": {"size_m": 64.0, "resolution_m": 0.2, "cells": 320},
            "network_local_crop": {
                "size_m": 6.4,
                "resolution_m": 0.2,
                "cells": 32,
            },
            "local_detail_provenance": "synthetic_subgrid_on_locked_dem",
        },
        "scenario_counts": {
            split: counts[split]
            for split in ("train", "validation", "test", "holdout")
        },
        "scenarios": scenarios,
    }
    payload["scenario_manifest_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_manifest_document(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or value.get("schema") != SCENARIO_MANIFEST_SCHEMA:
        raise ScenarioManifestError("scenario manifest schema is unsupported")
    document = dict(value)
    claimed = _require_sha(
        document.pop("scenario_manifest_sha256", None),
        "scenario manifest identity",
    )
    if _canonical_sha256(document) != claimed:
        raise ScenarioManifestError("scenario manifest identity mismatch")
    scenarios = document.get("scenarios")
    counts = document.get("scenario_counts")
    if not isinstance(scenarios, list) or not isinstance(counts, Mapping):
        raise ScenarioManifestError("scenario manifest inventory is invalid")
    actual_counts = Counter(
        item.get("split") for item in scenarios if isinstance(item, Mapping)
    )
    expected_counts = {
        "train": 1536,
        "validation": 96,
        "test": 96,
        "holdout": 6,
    }
    if dict(counts) != expected_counts or any(
        actual_counts[name] != expected for name, expected in expected_counts.items()
    ):
        raise ScenarioManifestError("scenario manifest counts are invalid")
    scene_ids = [
        item.get("scene_id") for item in scenarios if isinstance(item, Mapping)
    ]
    if (
        len(scene_ids) != len(scenarios)
        or len(set(scene_ids)) != len(scene_ids)
        or any(not isinstance(value, str) for value in scene_ids)
    ):
        raise ScenarioManifestError("scenario manifest scene identity is invalid")
    return {**document, "scenario_manifest_sha256": claimed}


def write_scenario_manifest(
    path: str | Path,
    document: Mapping[str, object],
    *,
    repository_root: str | Path,
) -> str:
    """Atomically write a validated manifest outside the Git repository."""
    validated = _validate_manifest_document(document)
    target = Path(path)
    repository = Path(repository_root).resolve(strict=True)
    if (
        not target.is_absolute()
        or target.is_symlink()
        or not target.parent.is_dir()
        or target.parent.resolve(strict=True).is_relative_to(repository)
    ):
        raise ScenarioManifestError(
            "scenario manifest output must be an absolute path outside the repository"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                validated,
                stream,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        if isinstance(error, ScenarioManifestError):
            raise
        raise ScenarioManifestError("scenario manifest write failed") from error
    return str(validated["scenario_manifest_sha256"])


def load_scenario_manifest(path: str | Path) -> dict[str, object]:
    """Load and revalidate a canonical formal scenario manifest."""
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ScenarioManifestError("scenario manifest must be a regular file")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScenarioManifestError("scenario manifest is invalid JSON") from error
    return _validate_manifest_document(value)


__all__ = [
    "FORMAL_GENERATOR_VERSION",
    "SCENARIO_MANIFEST_SCHEMA",
    "ScenarioManifestError",
    "TEST_SCENARIO_SEEDS",
    "TRAIN_SCENARIO_SEEDS",
    "VALIDATION_SCENARIO_SEEDS",
    "build_scenario_manifest_document",
    "load_scenario_manifest",
    "write_scenario_manifest",
]

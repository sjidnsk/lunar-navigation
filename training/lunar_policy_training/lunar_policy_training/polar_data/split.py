"""Deterministic, source-bound lunar polar PPO split catalogues."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Iterable, Sequence

import rasterio

from .source_lock import SourceLockError, load_aggregate_source_lock


NASA_WINDOW_METERS = 1024
MIN_CROSS_SPLIT_CENTER_DISTANCE_METERS = 2000
NASA_SPLIT_COUNTS = {"train": 192, "validation": 48, "test": 48}
JAXA_SITE_IDS = ("CR1", "GR1", "GR2", "LP1", "MP1", "MP2")
SPLIT_MANIFEST_SCHEMA = "lunar-polar-split-manifest/v1"


class SplitError(ValueError):
    """The supplied candidates or locked DEM cannot meet the frozen contract."""


@dataclass(frozen=True)
class PolarWindow:
    window_id: str
    center_x_m: float
    center_y_m: float
    width_m: int = NASA_WINDOW_METERS
    height_m: int = NASA_WINDOW_METERS
    source_pixel_bounds: tuple[int, int, int, int] | None = None
    source_sha256: str = "synthetic-nasa-87s"


@dataclass(frozen=True)
class JaxaSite:
    site_id: str


@dataclass(frozen=True)
class SplitCatalogRow:
    source: str
    split: str
    window_id: str
    center_x_m: float | None
    center_y_m: float | None
    width_m: int | None
    height_m: int | None
    split_sha256: str
    source_pixel_bounds: tuple[int, int, int, int] | None = None
    window_sha256: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def nasa_windows() -> tuple[PolarWindow, ...]:
    """Return frozen synthetic candidates for unit-level split validation only."""
    return tuple(
        PolarWindow(f"nasa-87s-{row:02d}-{column:02d}", column * 4096, row * 4096)
        for row in range(12) for column in range(24)
    )


def jaxa_sites() -> tuple[JaxaSite, ...]:
    return tuple(JaxaSite(site_id) for site_id in JAXA_SITE_IDS)


def nasa_windows_from_dem(path: str | Path, source_sha256: str) -> tuple[PolarWindow, ...]:
    """Derive the 288 stable candidates from a verified DEM's actual pixel grid."""
    try:
        with rasterio.open(path) as dataset:
            transform = dataset.transform
            if transform.b != 0 or transform.d != 0:
                raise SplitError("NASA DEM transform must be north-up")
            pixel_size = max(abs(transform.a), abs(transform.e))
            if pixel_size <= 0:
                raise SplitError("NASA DEM pixel size is invalid")
            window_pixels = math.ceil(NASA_WINDOW_METERS / pixel_size)
            stride_pixels = math.ceil(4096 / pixel_size)
            windows: list[PolarWindow] = []
            for row in range(12):
                for column in range(24):
                    col_off, row_off = column * stride_pixels, row * stride_pixels
                    col_stop, row_stop = col_off + window_pixels, row_off + window_pixels
                    if col_stop > dataset.width or row_stop > dataset.height:
                        raise SplitError("verified NASA DEM cannot provide 288 fixed windows")
                    center_x, center_y = transform * (
                        col_off + window_pixels / 2, row_off + window_pixels / 2
                    )
                    windows.append(PolarWindow(
                        window_id=f"nasa-87s-{row:02d}-{column:02d}",
                        center_x_m=float(center_x), center_y_m=float(center_y),
                        source_pixel_bounds=(col_off, row_off, col_stop, row_stop),
                        source_sha256=source_sha256,
                    ))
            return tuple(windows)
    except SplitError:
        raise
    except (OSError, rasterio.errors.RasterioError) as error:
        raise SplitError("verified NASA DEM cannot be read") from error


def build_split_catalog(nasa_candidates: Iterable[PolarWindow], jaxa_holdout_sites: Iterable[JaxaSite], *, seed: int = 4080) -> tuple[SplitCatalogRow, ...]:
    candidates, sites = tuple(nasa_candidates), tuple(jaxa_holdout_sites)
    if len(candidates) != sum(NASA_SPLIT_COUNTS.values()):
        raise SplitError("expected exactly 288 NASA windows")
    if tuple(site.site_id for site in sites) != JAXA_SITE_IDS:
        raise SplitError("expected exact JAXA site IDs CR1, GR1, GR2, LP1, MP1, MP2")
    _validate_nasa_windows(candidates)
    ordered = list(candidates)
    random.Random(seed).shuffle(ordered)
    assignments: dict[str, str] = {}
    offset = 0
    for split, count in NASA_SPLIT_COUNTS.items():
        for candidate in ordered[offset : offset + count]:
            assignments[candidate.window_id] = split
        offset += count
    _validate_cross_split_distance(candidates, assignments)
    rows = [
        SplitCatalogRow(
            source="NASA_LOLA", split=assignments[candidate.window_id], window_id=candidate.window_id,
            center_x_m=candidate.center_x_m, center_y_m=candidate.center_y_m,
            width_m=candidate.width_m, height_m=candidate.height_m, split_sha256="",
            source_pixel_bounds=candidate.source_pixel_bounds,
            window_sha256=_window_sha256(candidate),
        )
        for candidate in sorted(candidates, key=lambda item: item.window_id)
    ]
    rows.extend(SplitCatalogRow(
        source="JAXA_LUPEX", split="holdout", window_id=site.site_id, center_x_m=None,
        center_y_m=None, width_m=None, height_m=None, split_sha256="",
    ) for site in sites)
    digest = sha256(json.dumps([asdict(row) for row in rows], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return tuple(replace(row, split_sha256=digest) for row in rows)


def build_split_manifest(source_lock: str | Path, output: str | Path, *, seed: int, repository_root: str | Path) -> dict[str, object]:
    """Verify aggregate source locks and atomically emit a portable split manifest."""
    try:
        data_root, locks = load_aggregate_source_lock(source_lock, repository_root=repository_root)
    except SourceLockError as error:
        raise SplitError("source lock cannot be verified") from error
    dem = next((lock for lock in locks if lock.source_id == "NASA_LOLA_87S_DEM"), None)
    if dem is None or dem.artifact_kind != "raster":
        raise SplitError("aggregate source lock has no NASA DEM")
    catalog = build_split_catalog(nasa_windows_from_dem(data_root / dem.filename, dem.sha256), jaxa_sites(), seed=seed)
    document: dict[str, object] = {
        "schema": SPLIT_MANIFEST_SCHEMA, "seed": seed,
        "source": {"id": dem.source_id, "filename": dem.filename, "sha256": dem.sha256},
        "split_sha256": catalog[0].split_sha256, "rows": [asdict(row) for row in catalog],
    }
    _atomic_manifest_write(output, document, repository_root)
    return document


def _validate_nasa_windows(candidates: tuple[PolarWindow, ...]) -> None:
    for candidate in candidates:
        if candidate.width_m != NASA_WINDOW_METERS or candidate.height_m != NASA_WINDOW_METERS:
            raise SplitError("NASA windows must be fixed 1024 m squares")
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            if abs(left.center_x_m - right.center_x_m) < NASA_WINDOW_METERS and abs(left.center_y_m - right.center_y_m) < NASA_WINDOW_METERS:
                raise SplitError("NASA windows overlap")
    if len({candidate.window_id for candidate in candidates}) != len(candidates):
        raise SplitError("NASA window ids must be unique")


def _validate_cross_split_distance(candidates: tuple[PolarWindow, ...], assignments: dict[str, str]) -> None:
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            if assignments[left.window_id] != assignments[right.window_id] and max(abs(left.center_x_m - right.center_x_m), abs(left.center_y_m - right.center_y_m)) < MIN_CROSS_SPLIT_CENTER_DISTANCE_METERS:
                raise SplitError("NASA cross-split centers must be at least 2000 m apart")


def _window_sha256(window: PolarWindow) -> str:
    return sha256(json.dumps({"source_sha256": window.source_sha256, "window_id": window.window_id, "pixel_bounds": window.source_pixel_bounds}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _atomic_manifest_write(path: str | Path, document: dict[str, object], repository_root: str | Path) -> None:
    target, repository = Path(path), Path(repository_root).resolve(strict=True)
    if not target.is_absolute() or target.is_symlink() or not target.parent.is_dir() or target.parent.resolve(strict=True).is_relative_to(repository):
        raise SplitError("split output must be an absolute path outside the repository")
    temporary = target.with_name(f".{target.name}.part")
    if temporary.exists() or temporary.is_symlink():
        raise SplitError("existing split manifest part prevents write")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(document, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        raise SplitError("split manifest could not be written atomically") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=4080)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[4])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        build_split_manifest(arguments.source_lock, arguments.output, seed=arguments.seed, repository_root=arguments.repository_root)
    except SplitError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["JaxaSite", "PolarWindow", "SplitCatalogRow", "SplitError", "build_split_catalog", "build_split_manifest", "jaxa_sites", "main", "nasa_windows", "nasa_windows_from_dem"]

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
SPLIT_MANIFEST_SCHEMA = "lunar-polar-split-manifest/v2"


class SplitError(ValueError):
    """The supplied candidates or locked DEM cannot meet the frozen contract."""


@dataclass(frozen=True)
class PolarWindow:
    window_id: str
    center_x_m: float
    center_y_m: float
    width_m: int = NASA_WINDOW_METERS
    height_m: int = NASA_WINDOW_METERS
    source_pixel_bounds: tuple[float, float, float, float] | None = None
    world_bounds_m: tuple[float, float, float, float] | None = None
    read_pixel_envelope: tuple[int, int, int, int] | None = None
    source_sha256: str = "synthetic-nasa-87s"
    count_sha256: str = "synthetic-count-87s"
    count_quality: float = 0.0
    valid_fraction: float = 1.0


@dataclass(frozen=True)
class JaxaSite:
    site_id: str
    archive_sha256: str = ""
    archive_member_paths: tuple[str, ...] = ()
    archive_member_sha256s: tuple[str, ...] = ()


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
    valid_fraction: float | None = None
    source_pixel_bounds: tuple[float, float, float, float] | None = None
    world_bounds_m: tuple[float, float, float, float] | None = None
    read_pixel_envelope: tuple[int, int, int, int] | None = None
    window_sha256: str | None = None
    archive_sha256: str | None = None
    archive_member_paths: tuple[str, ...] | None = None
    archive_member_sha256s: tuple[str, ...] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def nasa_windows() -> tuple[PolarWindow, ...]:
    """Return frozen synthetic candidates for unit-level split validation only."""
    return tuple(
        PolarWindow(f"nasa-87s-{row:02d}-{column:02d}", column * 4096, row * 4096)
        for row in range(12) for column in range(24)
    )


def jaxa_sites(archive_lock: object | None = None) -> tuple[JaxaSite, ...]:
    if archive_lock is None:
        return tuple(JaxaSite(site_id) for site_id in JAXA_SITE_IDS)
    members = getattr(archive_lock, "archive_members", ())
    return tuple(
        JaxaSite(
            site_id,
            archive_sha256=getattr(archive_lock, "sha256"),
            archive_member_paths=tuple(member.path for member in members if member.site_id == site_id),
            archive_member_sha256s=tuple(member.sha256 for member in members if member.site_id == site_id),
        )
        for site_id in JAXA_SITE_IDS
    )


def nasa_window_from_world_bounds(
    transform: rasterio.Affine,
    world_bounds_m: tuple[float, float, float, float],
    source_sha256: str,
    count_sha256: str,
    *,
    window_id: str = "nasa-window",
    count_quality: float = 0.0,
) -> PolarWindow:
    """Keep exact 1024 m world geometry distinct from the integer read envelope."""
    left, bottom, right, top = world_bounds_m
    inverse = ~transform
    col0, row0 = inverse * (left, top)
    col1, row1 = inverse * (right, bottom)
    pixel_bounds = (float(min(col0, col1)), float(min(row0, row1)), float(max(col0, col1)), float(max(row0, row1)))
    envelope = (math.floor(pixel_bounds[0]), math.floor(pixel_bounds[1]), math.ceil(pixel_bounds[2]), math.ceil(pixel_bounds[3]))
    return PolarWindow(
        window_id=window_id, center_x_m=(left + right) / 2, center_y_m=(bottom + top) / 2,
        source_pixel_bounds=pixel_bounds, world_bounds_m=world_bounds_m,
        read_pixel_envelope=envelope, source_sha256=source_sha256, count_sha256=count_sha256,
        count_quality=count_quality,
    )


def nasa_windows_from_dem(path: str | Path, source_sha256: str, count_path: str | Path, count_sha256: str) -> tuple[PolarWindow, ...]:
    """Derive the 288 stable candidates from a verified DEM's actual pixel grid."""
    try:
        with rasterio.open(path) as dataset, rasterio.open(count_path) as count_dataset:
            transform = dataset.transform
            if transform.b != 0 or transform.d != 0:
                raise SplitError("NASA DEM transform must be north-up")
            _validate_dem_count_alignment(dataset, count_dataset)
            windows: list[PolarWindow] = []
            left = math.ceil(dataset.bounds.left / 4096) * 4096
            top = math.floor(dataset.bounds.top / 4096) * 4096
            row = 0
            while top - row * 4096 - NASA_WINDOW_METERS >= dataset.bounds.bottom:
                column = 0
                while left + column * 4096 + NASA_WINDOW_METERS <= dataset.bounds.right:
                    x0, y1 = left + column * 4096, top - row * 4096
                    candidate = nasa_window_from_world_bounds(
                        transform, (x0, y1 - NASA_WINDOW_METERS, x0 + NASA_WINDOW_METERS, y1),
                        source_sha256, count_sha256, window_id=f"nasa-87s-{row:03d}-{column:03d}",
                    )
                    envelope = candidate.read_pixel_envelope
                    assert envelope is not None
                    window = rasterio.windows.Window(
                        envelope[0],
                        envelope[1],
                        envelope[2] - envelope[0],
                        envelope[3] - envelope[1],
                    )
                    valid_fraction = float(
                        dataset.read_masks(1, window=window).astype(bool).mean()
                    )
                    if math.isclose(valid_fraction, 1.0, rel_tol=0.0, abs_tol=0.0):
                        quality = float(count_dataset.read(1, window=window).mean())
                        windows.append(
                            replace(
                                candidate,
                                count_quality=quality,
                                valid_fraction=valid_fraction,
                            )
                        )
                    column += 1
                row += 1
            return _select_count_stratified_windows(windows, seed=4080)
    except SplitError:
        raise
    except (OSError, rasterio.errors.RasterioError) as error:
        raise SplitError("verified NASA DEM cannot be read") from error


def build_split_catalog(nasa_candidates: Iterable[PolarWindow], jaxa_holdout_sites: Iterable[JaxaSite], *, seed: int = 4080) -> tuple[SplitCatalogRow, ...]:
    candidates, sites = tuple(nasa_candidates), tuple(jaxa_holdout_sites)
    candidates = _select_count_stratified_windows(list(candidates), seed=seed)
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
            world_bounds_m=candidate.world_bounds_m,
            read_pixel_envelope=candidate.read_pixel_envelope,
            window_sha256=_window_sha256(candidate),
            valid_fraction=candidate.valid_fraction,
        )
        for candidate in sorted(candidates, key=lambda item: item.window_id)
    ]
    rows.extend(SplitCatalogRow(
        source="JAXA_LUPEX", split="holdout", window_id=site.site_id, center_x_m=None,
        center_y_m=None, width_m=None, height_m=None, split_sha256="",
        valid_fraction=None,
        archive_sha256=site.archive_sha256 or None,
        archive_member_paths=site.archive_member_paths or None,
        archive_member_sha256s=site.archive_member_sha256s or None,
    ) for site in sites)
    digest = sha256(json.dumps([asdict(row) for row in rows], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return tuple(replace(row, split_sha256=digest) for row in rows)


def build_split_manifest(source_lock: str | Path, output: str | Path, *, seed: int, repository_root: str | Path) -> dict[str, object]:
    """Verify aggregate source locks and atomically emit a portable split manifest."""
    try:
        data_root, locks = load_aggregate_source_lock(source_lock, repository_root=repository_root)
    except SourceLockError as error:
        raise SplitError("source lock cannot be verified") from error
    required = {lock.source_id: lock for lock in locks if lock.source_id in {"NASA_LOLA_87S_DEM", "NASA_LOLA_87S_COUNT", "JAXA_LUPEX_DATA_S1"}}
    if set(required) != {"NASA_LOLA_87S_DEM", "NASA_LOLA_87S_COUNT", "JAXA_LUPEX_DATA_S1"}:
        raise SplitError("aggregate source lock requires all three NASA DEM/count and JAXA sources")
    dem, count, jaxa = required["NASA_LOLA_87S_DEM"], required["NASA_LOLA_87S_COUNT"], required["JAXA_LUPEX_DATA_S1"]
    if dem.artifact_kind != "raster" or count.artifact_kind != "raster" or jaxa.artifact_kind != "archive":
        raise SplitError("aggregate source lock artifact kinds are invalid")
    catalog = build_split_catalog(
        nasa_windows_from_dem(data_root / dem.filename, dem.sha256, data_root / count.filename, count.sha256),
        jaxa_sites(jaxa), seed=seed,
    )
    document: dict[str, object] = {
        "schema": SPLIT_MANIFEST_SCHEMA, "seed": seed,
        "sources": [
            {"id": lock.source_id, "filename": lock.filename, "sha256": lock.sha256}
            for lock in (dem, count, jaxa)
        ],
        "split_sha256": catalog[0].split_sha256, "rows": [asdict(row) for row in catalog],
    }
    _atomic_manifest_write(output, document, repository_root)
    return document


def _validate_dem_count_alignment(
    dem: rasterio.io.DatasetReader, count: rasterio.io.DatasetReader
) -> None:
    if (
        dem.crs != count.crs
        or dem.transform != count.transform
        or dem.width != count.width
        or dem.height != count.height
    ):
        raise SplitError("NASA DEM/count alignment drift")


def _select_count_stratified_windows(
    candidates: list[PolarWindow], *, seed: int
) -> tuple[PolarWindow, ...]:
    required = sum(NASA_SPLIT_COUNTS.values())
    if len(candidates) < required:
        raise SplitError("verified NASA DEM cannot provide 288 fixed windows")
    if len(candidates) == required:
        return tuple(candidates)
    ordered = sorted(candidates, key=lambda item: (item.count_quality, item.window_id))
    midpoint = ordered[len(ordered) // 2].count_quality
    low = [item for item in ordered if item.count_quality < midpoint]
    high = [item for item in ordered if item.count_quality >= midpoint]
    random.Random(seed).shuffle(low)
    random.Random(seed + 1).shuffle(high)
    selected = low[: required // 2] + high[: required - min(len(low), required // 2)]
    if len(selected) < required:
        selected_ids = {item.window_id for item in selected}
        remaining = [item for item in ordered if item.window_id not in selected_ids]
        random.Random(seed + 2).shuffle(remaining)
        selected.extend(remaining[: required - len(selected)])
    return tuple(sorted(selected, key=lambda item: item.window_id))


def _validate_nasa_windows(candidates: tuple[PolarWindow, ...]) -> None:
    for candidate in candidates:
        if candidate.width_m != NASA_WINDOW_METERS or candidate.height_m != NASA_WINDOW_METERS:
            raise SplitError("NASA windows must be fixed 1024 m squares")
        if not math.isclose(
            candidate.valid_fraction, 1.0, rel_tol=0.0, abs_tol=0.0
        ):
            raise SplitError("NASA windows must be fully valid with no NoData")
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
    return sha256(json.dumps({"source_sha256": window.source_sha256, "count_sha256": window.count_sha256, "window_id": window.window_id, "world_bounds_m": window.world_bounds_m, "source_pixel_bounds": window.source_pixel_bounds, "valid_fraction": window.valid_fraction}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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

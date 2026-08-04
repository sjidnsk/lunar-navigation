"""Deterministic, spatially separated polar PPO split catalogues."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import random
from typing import Iterable


NASA_WINDOW_METERS = 1024
MIN_CROSS_SPLIT_CENTER_DISTANCE_METERS = 2000
NASA_SPLIT_COUNTS = {"train": 192, "validation": 48, "test": 48}


class SplitError(ValueError):
    """The supplied split candidates cannot meet the frozen spatial contract."""


@dataclass(frozen=True)
class PolarWindow:
    window_id: str
    center_x_m: int
    center_y_m: int
    width_m: int = NASA_WINDOW_METERS
    height_m: int = NASA_WINDOW_METERS


@dataclass(frozen=True)
class JaxaSite:
    site_id: str


@dataclass(frozen=True)
class SplitCatalogRow:
    source: str
    split: str
    window_id: str
    center_x_m: int | None
    center_y_m: int | None
    width_m: int | None
    height_m: int | None
    split_sha256: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def nasa_windows() -> tuple[PolarWindow, ...]:
    """Return the frozen 288 non-overlapping 1024 m NASA candidate windows."""
    return tuple(
        PolarWindow(
            window_id=f"nasa-87s-{row:02d}-{column:02d}",
            center_x_m=column * 4096,
            center_y_m=row * 4096,
        )
        for row in range(12)
        for column in range(24)
    )


def jaxa_sites() -> tuple[JaxaSite, ...]:
    """Return the six immutable LUPEX holdout site identifiers."""
    return tuple(JaxaSite(site_id=f"jaxa-lupex-site-{index:02d}") for index in range(1, 7))


def build_split_catalog(
    nasa_candidates: Iterable[PolarWindow],
    jaxa_holdout_sites: Iterable[JaxaSite],
    *,
    seed: int = 4080,
) -> tuple[SplitCatalogRow, ...]:
    """Build a portable 192/48/48 NASA split plus JAXA-only holdout rows."""
    candidates = tuple(nasa_candidates)
    sites = tuple(jaxa_holdout_sites)
    expected_count = sum(NASA_SPLIT_COUNTS.values())
    if len(candidates) != expected_count:
        raise SplitError(f"expected exactly {expected_count} NASA windows")
    if len(sites) != 6:
        raise SplitError("expected exactly six JAXA holdout sites")
    _validate_nasa_windows(candidates)
    if len({site.site_id for site in sites}) != len(sites):
        raise SplitError("JAXA holdout site ids must be unique")

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
            source="NASA_LOLA",
            split=assignments[candidate.window_id],
            window_id=candidate.window_id,
            center_x_m=candidate.center_x_m,
            center_y_m=candidate.center_y_m,
            width_m=candidate.width_m,
            height_m=candidate.height_m,
            split_sha256="",
        )
        for candidate in sorted(candidates, key=lambda item: item.window_id)
    ]
    rows.extend(
        SplitCatalogRow(
            source="JAXA_LUPEX",
            split="holdout",
            window_id=site.site_id,
            center_x_m=None,
            center_y_m=None,
            width_m=None,
            height_m=None,
            split_sha256="",
        )
        for site in sorted(sites, key=lambda item: item.site_id)
    )
    digest = sha256(
        json.dumps(
            [asdict(row) for row in rows], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return tuple(replace(row, split_sha256=digest) for row in rows)


def _validate_nasa_windows(candidates: tuple[PolarWindow, ...]) -> None:
    for candidate in candidates:
        if candidate.width_m != NASA_WINDOW_METERS or candidate.height_m != NASA_WINDOW_METERS:
            raise SplitError("NASA windows must be fixed 1024 m squares")
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if (
                abs(left.center_x_m - right.center_x_m) < NASA_WINDOW_METERS
                and abs(left.center_y_m - right.center_y_m) < NASA_WINDOW_METERS
            ):
                raise SplitError("NASA windows overlap")
    if len({candidate.window_id for candidate in candidates}) != len(candidates):
        raise SplitError("NASA window ids must be unique")


def _validate_cross_split_distance(
    candidates: tuple[PolarWindow, ...], assignments: dict[str, str]
) -> None:
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if assignments[left.window_id] == assignments[right.window_id]:
                continue
            if max(
                abs(left.center_x_m - right.center_x_m),
                abs(left.center_y_m - right.center_y_m),
            ) < MIN_CROSS_SPLIT_CENTER_DISTANCE_METERS:
                raise SplitError("NASA cross-split centers must be at least 2000 m apart")


__all__ = [
    "JaxaSite",
    "NASA_SPLIT_COUNTS",
    "PolarWindow",
    "SplitCatalogRow",
    "SplitError",
    "build_split_catalog",
    "jaxa_sites",
    "nasa_windows",
]

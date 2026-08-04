from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_policy_training.polar_data.split import (  # noqa: E402
    SplitError,
    build_split_catalog,
    jaxa_sites,
    nasa_windows,
)


def test_jaxa_sites_are_holdout_only() -> None:
    catalog = build_split_catalog(nasa_windows(), jaxa_sites(), seed=4080)

    assert {row.split for row in catalog if row.source == "JAXA_LUPEX"} == {"holdout"}
    assert sum(row.split == "train" for row in catalog) == 192
    assert sum(row.split == "validation" for row in catalog) == 48
    assert sum(row.split == "test" for row in catalog) == 48


def test_split_catalog_is_deterministic_redacted_and_spatially_separated() -> None:
    first = build_split_catalog(nasa_windows(), jaxa_sites(), seed=4080)
    second = build_split_catalog(nasa_windows(), jaxa_sites(), seed=4080)

    assert first == second
    assert len({row.split_sha256 for row in first}) == 1
    assert all(row.width_m == row.height_m == 1024 for row in first if row.source == "NASA_LOLA")
    assert all("/" not in row.window_id and "\\" not in row.window_id for row in first)
    assert all("kai" not in row.to_json().lower() for row in first)
    nasa_rows = [row for row in first if row.source == "NASA_LOLA"]
    for index, left in enumerate(nasa_rows):
        for right in nasa_rows[index + 1 :]:
            if left.split != right.split:
                assert max(abs(left.center_x_m - right.center_x_m), abs(left.center_y_m - right.center_y_m)) >= 2000


def test_split_catalog_rejects_overlapping_or_too_close_nasa_windows() -> None:
    windows = list(nasa_windows())
    windows[1] = windows[0]

    with pytest.raises(SplitError, match="overlap|2000"):
        build_split_catalog(windows, jaxa_sites(), seed=4080)

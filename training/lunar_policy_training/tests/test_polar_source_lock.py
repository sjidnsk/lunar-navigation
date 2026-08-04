from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_policy_training.polar_data.source_lock import (  # noqa: E402
    PolarSourceLock,
    SourceLockError,
    verify_source_lock,
)


def _write_raster(
    path: Path,
    *,
    crs: str | None = "EPSG:4326",
    nodata: float | None = -9999.0,
    identity_transform: bool = False,
) -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import Affine, from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs=crs,
        transform=Affine.identity() if identity_transform else from_origin(12.0, 34.0, 5.0, 5.0),
        nodata=nodata,
    ) as dataset:
        dataset.write(np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"), 1)


def _locked_fixture(root: Path) -> tuple[Path, PolarSourceLock]:
    root.mkdir()
    fixture = root / "fixture.tif"
    _write_raster(fixture)
    return fixture, PolarSourceLock.from_file("NASA_LOLA_87S_DEM", fixture)


def test_verify_source_lock_accepts_matching_external_raster(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "polar-data"
    fixture, lock = _locked_fixture(data_root)

    assert verify_source_lock(data_root, lock, repository_root=repository) == fixture


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("missing", "missing"),
        ("size", "size"),
        ("hash", "sha256"),
        ("crs", "CRS"),
        ("transform", "transform"),
        ("nodata", "NoData"),
    ],
)
def test_verify_source_lock_rejects_missing_or_changed_raster(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "polar-data"
    fixture, lock = _locked_fixture(data_root)

    if mutation == "missing":
        fixture.unlink()
    elif mutation == "size":
        lock = PolarSourceLock(
            **{**lock.to_dict(), "size_bytes": lock.size_bytes + 1}
        )
    elif mutation == "hash":
        lock = PolarSourceLock(
            **{**lock.to_dict(), "sha256": "0" * 64}
        )
    elif mutation == "crs":
        _write_raster(fixture, crs=None)
    elif mutation == "transform":
        _write_raster(fixture, identity_transform=True)
    else:
        _write_raster(fixture, nodata=None)

    with pytest.raises(SourceLockError, match=expected):
        verify_source_lock(data_root, lock, repository_root=repository)


def test_verify_source_lock_rejects_repository_data_root_and_symlinks(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    in_repository = repository / "data"
    fixture, lock = _locked_fixture(in_repository)

    with pytest.raises(SourceLockError, match="outside"):
        verify_source_lock(in_repository, lock, repository_root=repository)

    external_root = tmp_path / "external"
    external_fixture, external_lock = _locked_fixture(external_root)
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(external_root, target_is_directory=True)
    with pytest.raises(SourceLockError, match="symbolic link"):
        verify_source_lock(linked_root, external_lock, repository_root=repository)

    linked_file = external_root / "linked.tif"
    linked_file.symlink_to(external_fixture)
    linked_lock = PolarSourceLock(
        **{**external_lock.to_dict(), "filename": linked_file.name}
    )
    with pytest.raises(SourceLockError, match="symbolic link"):
        verify_source_lock(external_root, linked_lock, repository_root=repository)

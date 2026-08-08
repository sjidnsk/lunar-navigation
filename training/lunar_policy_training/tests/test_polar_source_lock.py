from __future__ import annotations

import importlib.util
import io
import json
import math
import sys
import warnings
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_policy_training.polar_data.source_lock import (  # noqa: E402
    PolarSourceLock,
    SourceLockError,
    load_aggregate_source_lock,
    write_aggregate_source_lock,
    verify_source_lock,
)


FETCH_TOOL = Path(__file__).resolve().parents[2] / "tools/fetch_polar_data.py"
_fetch_spec = importlib.util.spec_from_file_location("fetch_polar_data", FETCH_TOOL)
assert _fetch_spec is not None and _fetch_spec.loader is not None
fetch_polar_data = importlib.util.module_from_spec(_fetch_spec)
_fetch_spec.loader.exec_module(fetch_polar_data)


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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
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


def _write_count_raster(path: Path, *, nodata: int | None = None) -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(12.0, 34.0, 5.0, 5.0),
        nodata=nodata,
    ) as dataset:
        dataset.write(np.array([[0, 1], [4, 5]], dtype="uint8"), 1)


def _locked_fixture(root: Path) -> tuple[Path, PolarSourceLock]:
    root.mkdir()
    fixture = root / "fixture.tif"
    _write_raster(fixture)
    return fixture, PolarSourceLock.from_file(
        "NASA_LOLA_87S_DEM", fixture,
        citation="NASA PGDA product 81; https://doi.org/10.1016/j.pss.2020.105119",
        license="NASA reproduction guidance",
        final_url="https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldem_87s_5mpp.tif",
    )


def _raster_bytes(*, nodata: float | None = -9999.0) -> bytes:
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(12.0, 34.0, 5.0, 5.0),
            nodata=nodata,
        ) as dataset:
            dataset.write(np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"), 1)
        return memory.read()


def _write_jaxa_archive(path: Path, *, omit_site: str | None = None) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for site_id in ("CR1", "GR1", "GR2", "LP1", "MP1", "MP2"):
            if site_id != omit_site:
                archive.writestr(f"{site_id}/{site_id}_DTM.tif", _raster_bytes())


def _write_real_layout_jaxa_archive(
    path: Path,
    *,
    extra_raster: str | None = None,
    null_nodata_member: str | None = None,
) -> None:
    patterns = (
        ("DTMs", "{site}_roi_sfs_1m-DEM.tif"),
        ("orthomosaics", "{site}_roi_sfs_1m-ORTHO.tif"),
        ("uncertainties", "{site}_sfs-height-error-final.tif"),
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for site_id in ("cr1", "gr1", "gr2", "lp1", "mp1", "mp2"):
            for directory, pattern in patterns:
                member = f"{directory}/{pattern.format(site=site_id)}"
                archive.writestr(
                    member,
                    _raster_bytes(nodata=None if member == null_nodata_member else -9999.0),
                )
        if extra_raster is not None:
            archive.writestr(extra_raster, _raster_bytes())


def test_verify_source_lock_accepts_matching_external_raster(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "polar-data"
    fixture, lock = _locked_fixture(data_root)

    assert verify_source_lock(data_root, lock, repository_root=repository) == fixture


def test_verify_source_lock_accepts_matching_nan_nodata_raster(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "polar-data"
    data_root.mkdir()
    fixture = data_root / "dem.tif"
    _write_raster(fixture, nodata=float("nan"))
    lock = PolarSourceLock.from_file(
        "NASA_LOLA_87S_DEM",
        fixture,
        citation="NASA PGDA product 81",
        license="NASA reproduction guidance",
        final_url="https://example.invalid/dem.tif",
    )

    assert math.isnan(lock.nodata)
    assert verify_source_lock(data_root, lock, repository_root=repository) == fixture


def test_nan_nodata_survives_aggregate_json_roundtrip(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "polar-data"
    data_root.mkdir()
    fixture = data_root / "dem.tif"
    _write_raster(fixture, nodata=float("nan"))
    lock = PolarSourceLock.from_file(
        "NASA_LOLA_87S_DEM",
        fixture,
        citation="NASA PGDA product 81",
        license="NASA reproduction guidance",
        final_url="https://example.invalid/dem.tif",
    )
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    aggregate = lock_dir / "polar_source_lock_v1.json"
    write_aggregate_source_lock(
        aggregate, data_root, (lock,), repository_root=repository
    )

    loaded_root, loaded_locks = load_aggregate_source_lock(
        aggregate, repository_root=repository
    )

    assert loaded_root == data_root.resolve()
    assert len(loaded_locks) == 1
    assert math.isnan(loaded_locks[0].nodata)
    assert verify_source_lock(
        loaded_root, loaded_locks[0], repository_root=repository
    ) == fixture


@pytest.mark.parametrize(
    ("source_nodata", "locked_nodata"),
    [(float("nan"), -9999.0), (-9999.0, float("nan"))],
)
def test_verify_source_lock_rejects_one_sided_nan_nodata_drift(
    tmp_path: Path, source_nodata: float, locked_nodata: float
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "polar-data"
    data_root.mkdir()
    fixture = data_root / "dem.tif"
    _write_raster(fixture, nodata=source_nodata)
    lock = PolarSourceLock.from_file(
        "NASA_LOLA_87S_DEM",
        fixture,
        citation="NASA PGDA product 81",
        license="NASA reproduction guidance",
        final_url="https://example.invalid/dem.tif",
    )
    drifted_lock = PolarSourceLock(
        **{**lock.to_dict(), "nodata": locked_nodata}
    )

    with pytest.raises(SourceLockError, match="metadata"):
        verify_source_lock(data_root, drifted_lock, repository_root=repository)


def test_official_lola_count_accepts_explicit_null_nodata_roundtrip_and_detects_drift(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "polar-data"
    data_root.mkdir()
    fixture = data_root / "ldec_87s_5mpp.tif"
    _write_count_raster(fixture)

    lock = PolarSourceLock.from_file(
        "NASA_LOLA_87S_COUNT",
        fixture,
        citation="NASA PGDA product 81",
        license="NASA reproduction guidance",
        final_url="https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldec_87s_5mpp.tif",
    )

    document = lock.to_dict()
    assert "nodata" in document and document["nodata"] is None
    assert PolarSourceLock.from_dict(document) == lock
    assert verify_source_lock(data_root, lock, repository_root=repository) == fixture

    _write_count_raster(fixture, nodata=255)
    with pytest.raises(SourceLockError, match="metadata"):
        verify_source_lock(data_root, lock, repository_root=repository)


def test_official_lola_count_dict_requires_explicit_nodata_field(tmp_path: Path) -> None:
    fixture = tmp_path / "ldec_87s_5mpp.tif"
    _write_count_raster(fixture)
    lock = PolarSourceLock.from_file(
        "NASA_LOLA_87S_COUNT",
        fixture,
        citation="NASA PGDA product 81",
        license="NASA reproduction guidance",
        final_url="https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldec_87s_5mpp.tif",
    )
    document = lock.to_dict()
    document.pop("nodata")

    with pytest.raises(SourceLockError, match="NoData.*explicit"):
        PolarSourceLock.from_dict(document)


@pytest.mark.parametrize("source_id", ["NASA_LOLA_87S_DEM", "NASA_LOLA_87S_COUNT_COPY"])
def test_null_nodata_remains_invalid_for_non_count_rasters(
    tmp_path: Path, source_id: str
) -> None:
    fixture = tmp_path / "source.tif"
    _write_count_raster(fixture)

    with pytest.raises(SourceLockError, match="NoData"):
        PolarSourceLock.from_file(
            source_id,
            fixture,
            citation="official source",
            license="official license",
            final_url="https://example.invalid/source.tif",
        )


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


def test_jaxa_archive_lock_inventories_exact_six_raster_sites(tmp_path: Path) -> None:
    archive = tmp_path / "DataS1.zip"
    _write_jaxa_archive(archive)

    lock = PolarSourceLock.from_file(
        "JAXA_LUPEX_DATA_S1",
        archive,
        citation="https://doi.org/10.5281/zenodo.17153447",
        license="cc-by-4.0",
        final_url="https://zenodo.org/api/files/DataS1.zip",
    )

    assert lock.artifact_kind == "archive"
    assert lock.crs is lock.transform is lock.nodata is None
    assert {member.site_id for member in lock.archive_members} == {
        "CR1",
        "GR1",
        "GR2",
        "LP1",
        "MP1",
        "MP2",
    }
    assert all(member.crs and member.transform and member.nodata is not None for member in lock.archive_members)


def test_jaxa_archive_lock_maps_all_real_lowercase_basename_prefixes(tmp_path: Path) -> None:
    archive = tmp_path / "DataS1.zip"
    _write_real_layout_jaxa_archive(archive)

    lock = PolarSourceLock.from_file(
        "JAXA_LUPEX_DATA_S1",
        archive,
        citation="https://doi.org/10.5281/zenodo.17153447",
        license="cc-by-4.0",
        final_url="https://zenodo.org/api/files/DataS1.zip",
    )

    assert len(lock.archive_members) == 18
    assert {
        site_id: sum(member.site_id == site_id for member in lock.archive_members)
        for site_id in ("CR1", "GR1", "GR2", "LP1", "MP1", "MP2")
    } == {site_id: 3 for site_id in ("CR1", "GR1", "GR2", "LP1", "MP1", "MP2")}
    assert lock.archive_members == tuple(sorted(lock.archive_members, key=lambda member: member.path))


@pytest.mark.parametrize(
    "extra_raster",
    [
        "DTMs/xx1_roi_sfs_1m-DEM.tif",
        "DTMs/xcr1_roi_sfs_1m-DEM.tif",
        "DTMs/cr1x_roi_sfs_1m-DEM.tif",
        "DTMs/prefix-cr1_roi_sfs_1m-DEM.tif",
    ],
)
def test_jaxa_archive_lock_rejects_unknown_or_unanchored_basename_prefix(
    tmp_path: Path, extra_raster: str
) -> None:
    archive = tmp_path / "DataS1.zip"
    _write_real_layout_jaxa_archive(archive, extra_raster=extra_raster)

    with pytest.raises(SourceLockError, match="site id"):
        PolarSourceLock.from_file(
            "JAXA_LUPEX_DATA_S1",
            archive,
            citation="https://doi.org/10.5281/zenodo.17153447",
            license="cc-by-4.0",
            final_url="https://zenodo.org/api/files/DataS1.zip",
        )


def test_jaxa_archive_lock_keeps_numeric_nodata_requirement_for_real_layout(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "DataS1.zip"
    _write_real_layout_jaxa_archive(
        archive,
        null_nodata_member="DTMs/cr1_roi_sfs_1m-DEM.tif",
    )

    with pytest.raises(SourceLockError, match="NoData"):
        PolarSourceLock.from_file(
            "JAXA_LUPEX_DATA_S1",
            archive,
            citation="https://doi.org/10.5281/zenodo.17153447",
            license="cc-by-4.0",
            final_url="https://zenodo.org/api/files/DataS1.zip",
        )


def test_jaxa_archive_lock_fails_closed_when_one_site_is_missing(tmp_path: Path) -> None:
    archive = tmp_path / "DataS1.zip"
    _write_jaxa_archive(archive, omit_site="MP2")

    with pytest.raises(SourceLockError, match="MP2|six"):
        PolarSourceLock.from_file(
            "JAXA_LUPEX_DATA_S1",
            archive,
            citation="https://doi.org/10.5281/zenodo.17153447",
            license="cc-by-4.0",
            final_url="https://zenodo.org/api/files/DataS1.zip",
        )


@pytest.mark.parametrize(
    "malicious_member",
    [
        "C:" + "/" + "Users" + "/kai/CR1/CR1_DTM.tif",
        "//server/share/CR1/CR1_DTM.tif",
        r"CR1\CR1_DTM.tif",
        "/CR1/CR1_DTM.tif",
        "../CR1/CR1_DTM.tif",
    ],
)
def test_jaxa_archive_lock_rejects_windows_and_posix_unsafe_member_paths(
    tmp_path: Path, malicious_member: str
) -> None:
    archive = tmp_path / "DataS1.zip"
    _write_jaxa_archive(archive)
    with ZipFile(archive, "a", compression=ZIP_DEFLATED) as zip_file:
        zip_file.writestr(malicious_member, _raster_bytes())

    with pytest.raises(SourceLockError, match="unsafe"):
        PolarSourceLock.from_file(
            "JAXA_LUPEX_DATA_S1",
            archive,
            citation="https://doi.org/10.5281/zenodo.17153447",
            license="cc-by-4.0",
            final_url="https://zenodo.org/api/files/DataS1.zip",
        )


def test_existing_destination_rejects_lock_filename_mismatch(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    actual = data_root / "other.tif"
    _write_raster(actual)
    lock = PolarSourceLock.from_file(
        "NASA_LOLA_87S_DEM", actual, citation="NASA PGDA product 81",
        license="NASA reproduction guidance", final_url="https://example.invalid/other.tif",
    )
    (data_root / "requested.tif").write_bytes(b"placeholder")
    (data_root / "requested.tif.source-lock.json").write_text(
        json.dumps(lock.to_dict()), encoding="utf-8"
    )

    with pytest.raises(fetch_polar_data.PolarFetchError, match="filename"):
        fetch_polar_data.fetch_source(
            {"id": "NASA_LOLA_87S_DEM", "filename": "requested.tif", "url": "https://example.invalid/requested.tif", "citation": "fixture", "license": "fixture"},
            data_root,
            repository_root=repository,
        )


class _Response(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        status: int,
        content_length: int | str | None = None,
        content_range: str | None = None,
        final_url: str = "https://example.invalid/file",
    ) -> None:
        super().__init__(body)
        self.status = status
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        if content_range is not None:
            self.headers["Content-Range"] = content_range
        self._final_url = final_url

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def geturl(self) -> str:
        return self._final_url


def _download_source(expected_size_bytes: int) -> dict[str, object]:
    return {
        "id": "NASA_LOLA_87S_COUNT",
        "filename": "source.tif",
        "url": "https://example.invalid/source.tif",
        "expected_size_bytes": expected_size_bytes,
        "citation": "NASA PGDA product 81",
        "license": "NASA reproduction guidance",
    }


def test_initial_normal_eof_before_content_length_retains_part_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(
        fetch_polar_data,
        "urlopen",
        lambda *_args, **_kwargs: _Response(b"abc", status=200, content_length=6),
    )

    with pytest.raises(fetch_polar_data.PolarFetchError, match="length|truncated|part retained"):
        fetch_polar_data.fetch_source(
            _download_source(6), data_root, repository_root=repository
        )

    assert (data_root / "source.tif.part").read_bytes() == b"abc"
    assert not (data_root / "source.tif").exists()
    assert not (data_root / "source.tif.source-lock.json").exists()


def test_resume_normal_eof_before_content_range_end_retains_appended_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    part = data_root / "source.tif.part"
    part.write_bytes(b"abc")

    def fake_open(request: object, **_: object) -> _Response:
        assert request.get_header("Range") == "bytes=3-"
        return _Response(b"de", status=206, content_range="bytes 3-5/6")

    monkeypatch.setattr(fetch_polar_data, "urlopen", fake_open)
    with pytest.raises(fetch_polar_data.PolarFetchError, match="length|truncated|part retained"):
        fetch_polar_data.fetch_source(
            _download_source(6), data_root, repository_root=repository
        )

    assert part.read_bytes() == b"abcde"
    assert not (data_root / "source.tif").exists()
    assert not (data_root / "source.tif.source-lock.json").exists()


def test_resume_rejects_advertised_total_registry_conflict_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    part = data_root / "source.tif.part"
    part.write_bytes(b"abc")
    monkeypatch.setattr(
        fetch_polar_data,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            b"def", status=206, content_range="bytes 3-5/6"
        ),
    )

    with pytest.raises(fetch_polar_data.PolarFetchError, match="registry|total|length"):
        fetch_polar_data.fetch_source(
            _download_source(7), data_root, repository_root=repository
        )

    assert part.read_bytes() == b"abc"
    assert not (data_root / "source.tif").exists()
    assert not (data_root / "source.tif.source-lock.json").exists()


def test_range_resume_appends_only_matching_partial_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part = tmp_path / "source.tif.part"
    destination = tmp_path / "source.tif"
    part.write_bytes(b"abc")

    def fake_open(request: object, **_: object) -> _Response:
        assert request.get_header("Range") == "bytes=3-"
        return _Response(
            b"def", status=206, content_length=3, content_range="bytes 3-5/6"
        )

    monkeypatch.setattr(fetch_polar_data, "urlopen", fake_open)
    assert fetch_polar_data._download_atomic(
        "https://example.invalid/file", part, destination, expected_size_bytes=6
    ) == "https://example.invalid/file"
    assert destination.read_bytes() == b"abcdef"


def test_overlong_partial_response_rolls_back_to_original_resume_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = tmp_path / "source.tif.part"
    destination = tmp_path / "source.tif"
    part.write_bytes(b"abc")
    monkeypatch.setattr(
        fetch_polar_data,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            b"defg", status=206, content_length=3, content_range="bytes 3-5/6"
        ),
    )

    with pytest.raises(fetch_polar_data.PolarFetchError, match="extra|length|part retained"):
        fetch_polar_data._download_atomic(
            "https://example.invalid/file", part, destination, expected_size_bytes=6
        )

    assert part.read_bytes() == b"abc"
    assert not destination.exists()


def test_partial_response_read_error_rolls_back_to_original_resume_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = tmp_path / "source.tif.part"
    destination = tmp_path / "source.tif"
    part.write_bytes(b"abc")

    class InterruptedResponse(_Response):
        def read(self, size: int = -1) -> bytes:
            if self.tell() == 0:
                return super().read(size)
            raise OSError("network interrupted")

    monkeypatch.setattr(
        fetch_polar_data,
        "urlopen",
        lambda *_args, **_kwargs: InterruptedResponse(
            b"def", status=206, content_length=3, content_range="bytes 3-5/6"
        ),
    )

    with pytest.raises(fetch_polar_data.PolarFetchError, match="part retained"):
        fetch_polar_data._download_atomic(
            "https://example.invalid/file", part, destination, expected_size_bytes=6
        )

    assert part.read_bytes() == b"abc"
    assert not destination.exists()


def test_range_resume_restarts_when_server_ignores_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part = tmp_path / "source.tif.part"
    destination = tmp_path / "source.tif"
    part.write_bytes(b"abc")

    def fake_open(request: object, **_: object) -> _Response:
        assert request.get_header("Range") == "bytes=3-"
        return _Response(b"fresh", status=200, content_length=5)

    monkeypatch.setattr(fetch_polar_data, "urlopen", fake_open)
    fetch_polar_data._download_atomic(
        "https://example.invalid/file", part, destination, expected_size_bytes=5
    )
    assert destination.read_bytes() == b"fresh"


def test_overlong_200_fallback_preserves_original_resume_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = tmp_path / "source.tif.part"
    destination = tmp_path / "source.tif"
    part.write_bytes(b"abc")
    monkeypatch.setattr(
        fetch_polar_data,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            b"freshX", status=200, content_length=5
        ),
    )

    with pytest.raises(fetch_polar_data.PolarFetchError, match="extra|length|part retained"):
        fetch_polar_data._download_atomic(
            "https://example.invalid/file", part, destination, expected_size_bytes=5
        )

    assert part.read_bytes() == b"abc"
    assert not (tmp_path / ".source.tif.part.restart").exists()
    assert not destination.exists()


def test_overlong_initial_200_keeps_one_byte_short_retryable_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = tmp_path / "source.tif.part"
    destination = tmp_path / "source.tif"
    monkeypatch.setattr(
        fetch_polar_data,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            b"abcdefg", status=200, content_length=6
        ),
    )

    with pytest.raises(fetch_polar_data.PolarFetchError, match="extra|length|part retained"):
        fetch_polar_data._download_atomic(
            "https://example.invalid/file", part, destination, expected_size_bytes=6
        )

    assert part.read_bytes() == b"abcde"
    assert not destination.exists()

    def retry_open(request: object, **_: object) -> _Response:
        assert request.get_header("Range") == "bytes=5-"
        return _Response(
            b"f", status=206, content_length=1, content_range="bytes 5-5/6"
        )

    monkeypatch.setattr(fetch_polar_data, "urlopen", retry_open)
    fetch_polar_data._download_atomic(
        "https://example.invalid/file", part, destination, expected_size_bytes=6
    )
    assert destination.read_bytes() == b"abcdef"
    assert not part.exists()


def test_range_resume_rejects_end_outside_advertised_total_and_retains_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = tmp_path / "source.tif.part"
    destination = tmp_path / "source.tif"
    part.write_bytes(b"abc")
    monkeypatch.setattr(
        fetch_polar_data,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            b"defg", status=206, content_range="bytes 3-6/6"
        ),
    )

    with pytest.raises(fetch_polar_data.PolarFetchError, match="range"):
        fetch_polar_data._download_atomic(
            "https://example.invalid/file", part, destination, expected_size_bytes=6
        )
    assert part.read_bytes() == b"abc"
    assert not destination.exists()


def test_range_resume_rejects_https_downgrade_and_retains_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part = tmp_path / "source.tif.part"
    destination = tmp_path / "source.tif"
    part.write_bytes(b"abc")

    monkeypatch.setattr(
        fetch_polar_data,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            b"def", status=206, content_range="bytes 3-5/6", final_url="http://unsafe.invalid/file"
        ),
    )
    with pytest.raises(fetch_polar_data.PolarFetchError, match="HTTPS"):
        fetch_polar_data._download_atomic(
            "https://example.invalid/file", part, destination, expected_size_bytes=6
        )
    assert part.read_bytes() == b"abc"
    assert not destination.exists()


def test_aggregate_lock_uses_relative_external_source_root(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "raw"
    fixture, lock = _locked_fixture(data_root)
    lock_path = tmp_path / "locks" / "polar_source_lock_v1.json"
    lock_path.parent.mkdir()

    write_aggregate_source_lock(lock_path, data_root, (lock,), repository_root=repository)
    document = json.loads(lock_path.read_text(encoding="utf-8"))
    assert document["source_root"] == "../raw"
    assert str(data_root) not in lock_path.read_text(encoding="utf-8")
    assert fixture.exists()


def test_relative_data_root_is_rejected_before_any_download(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(SourceLockError, match="absolute"):
        verify_source_lock("relative-data", _locked_fixture(tmp_path / "external")[1], repository_root=repository)


def test_fetch_cli_writes_aggregate_lock_with_small_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "raw"
    data_root.mkdir()
    locks = tmp_path / "locks"
    locks.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema": "lunar-polar-source-registry/v1", "sources": [{
        "id": "NASA_LOLA_87S_DEM", "filename": "dem.tif", "url": "https://example.invalid/dem.tif",
        "citation": "NASA PGDA product 81; https://doi.org/10.1016/j.pss.2020.105119", "license": "NASA reproduction guidance"
    }]}), encoding="utf-8")

    def fake_fetch(source: dict[str, object], root: Path, *, repository_root: Path) -> Path:
        destination = root / str(source["filename"])
        _write_raster(destination)
        lock = PolarSourceLock.from_file(
            str(source["id"]), destination, citation=str(source["citation"]),
            license=str(source["license"]), final_url="https://example.invalid/dem.tif",
        )
        (root / f"{destination.name}.source-lock.json").write_text(json.dumps(lock.to_dict()), encoding="utf-8")
        return destination

    monkeypatch.setattr(fetch_polar_data, "fetch_source", fake_fetch)
    assert fetch_polar_data.main(["--registry", str(registry), "--data-root", str(data_root), "--lock-output", str(locks / "aggregate.json"), "--repository-root", str(repository)]) == 0
    aggregate = json.loads((locks / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["source_root"] == "../raw"
    assert str(data_root) not in json.dumps(aggregate)


def test_registry_carries_official_license_citation_and_exact_source_sizes() -> None:
    registry = fetch_polar_data.load_registry(
        Path(__file__).resolve().parents[2] / "data_sources/polar_source_registry_v1.json"
    )
    jaxa = next(source for source in registry if source["id"] == "JAXA_LUPEX_DATA_S1")
    nasa_dem = next(source for source in registry if source["id"] == "NASA_LOLA_87S_DEM")
    nasa_count = next(source for source in registry if source["id"] == "NASA_LOLA_87S_COUNT")
    assert nasa_dem["expected_size_bytes"] == 3465285714
    assert nasa_count["expected_size_bytes"] == 145895726
    assert jaxa["expected_size_bytes"] == 76134049
    assert jaxa["license"] == "cc-by-4.0"
    assert "10.5281/zenodo.17153447" in str(jaxa["citation"])
    assert "10.1016/j.pss.2020.105119" in str(nasa_dem["citation"])


def test_existing_source_reuse_must_match_registry_expected_size(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    data_root = tmp_path / "data"
    fixture, lock = _locked_fixture(data_root)
    lock_path = data_root / f"{fixture.name}.source-lock.json"
    lock_path.write_text(json.dumps(lock.to_dict()), encoding="utf-8")
    source = {
        "id": lock.source_id,
        "filename": lock.filename,
        "url": "https://example.invalid/fixture.tif",
        "expected_size_bytes": lock.size_bytes + 1,
        "citation": lock.citation,
        "license": lock.license,
    }

    with pytest.raises(fetch_polar_data.PolarFetchError, match="registry|size"):
        fetch_polar_data.fetch_source(source, data_root, repository_root=repository)

    assert fixture.is_file()
    assert lock_path.is_file()


def test_interrupted_download_retains_part_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part = tmp_path / "source.tif.part"
    destination = tmp_path / "source.tif"

    class InterruptedResponse(_Response):
        def read(self, size: int = -1) -> bytes:
            if self.tell() == 0:
                return super().read(size)
            raise OSError("network interrupted")

    monkeypatch.setattr(fetch_polar_data, "urlopen", lambda *_args, **_kwargs: InterruptedResponse(b"partial", status=200, content_length=7))
    with pytest.raises(fetch_polar_data.PolarFetchError, match="part retained"):
        fetch_polar_data._download_atomic(
            "https://example.invalid/file", part, destination, expected_size_bytes=7
        )
    assert part.read_bytes() == b"partia"
    assert not destination.exists()

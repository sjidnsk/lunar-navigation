from __future__ import annotations

import importlib.util
import io
import json
import sys
import warnings
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_policy_training.polar_data.source_lock import (  # noqa: E402
    PolarSourceLock,
    SourceLockError,
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


def _raster_bytes() -> bytes:
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
            nodata=-9999.0,
        ) as dataset:
            dataset.write(np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"), 1)
        return memory.read()


def _write_jaxa_archive(path: Path, *, omit_site: str | None = None) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for site_id in ("CR1", "GR1", "GR2", "LP1", "MP1", "MP2"):
            if site_id != omit_site:
                archive.writestr(f"{site_id}/{site_id}_DTM.tif", _raster_bytes())


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
    def __init__(self, body: bytes, *, status: int, content_range: str | None = None, final_url: str = "https://example.invalid/file") -> None:
        super().__init__(body)
        self.status = status
        self.headers = {} if content_range is None else {"Content-Range": content_range}
        self._final_url = final_url

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def geturl(self) -> str:
        return self._final_url


def test_range_resume_appends_only_matching_partial_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part = tmp_path / "source.tif.part"
    destination = tmp_path / "source.tif"
    part.write_bytes(b"abc")

    def fake_open(request: object, **_: object) -> _Response:
        assert request.get_header("Range") == "bytes=3-"
        return _Response(b"def", status=206, content_range="bytes 3-5/6")

    monkeypatch.setattr(fetch_polar_data, "urlopen", fake_open)
    assert fetch_polar_data._download_atomic("https://example.invalid/file", part, destination) == "https://example.invalid/file"
    assert destination.read_bytes() == b"abcdef"


def test_range_resume_restarts_when_server_ignores_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part = tmp_path / "source.tif.part"
    destination = tmp_path / "source.tif"
    part.write_bytes(b"abc")

    def fake_open(request: object, **_: object) -> _Response:
        assert request.get_header("Range") == "bytes=3-"
        return _Response(b"fresh", status=200)

    monkeypatch.setattr(fetch_polar_data, "urlopen", fake_open)
    fetch_polar_data._download_atomic("https://example.invalid/file", part, destination)
    assert destination.read_bytes() == b"fresh"


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
        fetch_polar_data._download_atomic("https://example.invalid/file", part, destination)
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


def test_registry_carries_official_license_citation_and_jaxa_size() -> None:
    registry = fetch_polar_data.load_registry(
        Path(__file__).resolve().parents[2] / "data_sources/polar_source_registry_v1.json"
    )
    jaxa = next(source for source in registry if source["id"] == "JAXA_LUPEX_DATA_S1")
    nasa = next(source for source in registry if source["id"] == "NASA_LOLA_87S_DEM")
    assert jaxa["expected_size_bytes"] == 76134049
    assert jaxa["license"] == "cc-by-4.0"
    assert "10.5281/zenodo.17153447" in str(jaxa["citation"])
    assert "10.1016/j.pss.2020.105119" in str(nasa["citation"])


def test_interrupted_download_retains_part_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part = tmp_path / "source.tif.part"
    destination = tmp_path / "source.tif"

    class InterruptedResponse(_Response):
        def read(self, size: int = -1) -> bytes:
            if self.tell() == 0:
                return super().read(size)
            raise OSError("network interrupted")

    monkeypatch.setattr(fetch_polar_data, "urlopen", lambda *_args, **_kwargs: InterruptedResponse(b"partial", status=200))
    with pytest.raises(fetch_polar_data.PolarFetchError, match="part retained"):
        fetch_polar_data._download_atomic("https://example.invalid/file", part, destination)
    assert part.read_bytes() == b"partial"
    assert not destination.exists()

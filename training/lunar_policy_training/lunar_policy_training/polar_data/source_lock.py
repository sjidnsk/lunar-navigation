"""Fail-closed locks for external lunar-polar raster and archive sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping
from zipfile import BadZipFile, ZipFile

import rasterio
from rasterio.io import MemoryFile


SOURCE_LOCK_SCHEMA = "lunar-polar-source-lock/v1"
AGGREGATE_SOURCE_LOCK_SCHEMA = "lunar-polar-source-lock-aggregate/v1"
JAXA_SITE_IDS = ("CR1", "GR1", "GR2", "LP1", "MP1", "MP2")
_NULL_NODATA_RASTER_SOURCE_ID = "NASA_LOLA_87S_COUNT"


class SourceLockError(ValueError):
    """A source, lock, or external-root invariant did not pass verification."""


@dataclass(frozen=True)
class ArchiveMemberLock:
    """Verified GeoTIFF identity for one JAXA archive member."""

    path: str
    site_id: str
    size_bytes: int
    sha256: str
    crs: str
    transform: tuple[float, float, float, float, float, float]
    nodata: float

    def __post_init__(self) -> None:
        _validated_archive_member_path(self.path)
        if self.site_id not in JAXA_SITE_IDS:
            raise SourceLockError("archive member site id is invalid")
        if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise SourceLockError("archive member size is invalid")
        _validate_sha256(self.sha256, "archive member")
        if not self.crs:
            raise SourceLockError("archive member CRS is missing")
        normalized = _normalize_transform(self.transform)
        object.__setattr__(self, "transform", normalized)
        if not isinstance(self.nodata, (int, float)) or isinstance(self.nodata, bool):
            raise SourceLockError("archive member NoData is invalid")

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "ArchiveMemberLock":
        return cls(
            path=document.get("path"),
            site_id=document.get("site_id"),
            size_bytes=document.get("size_bytes"),
            sha256=document.get("sha256"),
            crs=document.get("crs"),
            transform=document.get("transform"),
            nodata=document.get("nodata"),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PolarSourceLock:
    """Portable official-source identity; archive metadata stays at member level."""

    schema: str
    source_id: str
    filename: str
    size_bytes: int
    sha256: str
    crs: str | None
    transform: tuple[float, float, float, float, float, float] | None
    nodata: float | None
    artifact_kind: str
    citation: str
    license: str
    final_url: str
    archive_members: tuple[ArchiveMemberLock, ...]

    def __post_init__(self) -> None:
        if self.schema != SOURCE_LOCK_SCHEMA:
            raise SourceLockError("unsupported source lock schema")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise SourceLockError("source id is missing")
        if not isinstance(self.filename, str) or Path(self.filename).name != self.filename:
            raise SourceLockError("source lock filename must not contain a path")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise SourceLockError("source lock size is invalid")
        _validate_sha256(self.sha256, "source lock")
        if not isinstance(self.citation, str) or not self.citation:
            raise SourceLockError("source citation is missing")
        if not isinstance(self.license, str) or not self.license:
            raise SourceLockError("source license is missing")
        _require_https(self.final_url, "source final URL")
        members = tuple(self.archive_members)
        object.__setattr__(self, "archive_members", members)
        if self.artifact_kind == "raster":
            if not isinstance(self.crs, str) or not self.crs:
                raise SourceLockError("raster CRS is missing")
            if self.transform is None:
                raise SourceLockError("raster transform is missing")
            object.__setattr__(self, "transform", _normalize_transform(self.transform))
            if self.nodata is None and self.source_id != _NULL_NODATA_RASTER_SOURCE_ID:
                raise SourceLockError("raster NoData is missing")
            if self.nodata is not None and (
                not isinstance(self.nodata, (int, float)) or isinstance(self.nodata, bool)
            ):
                raise SourceLockError("raster NoData is invalid")
            if members:
                raise SourceLockError("raster source must not contain archive members")
        elif self.artifact_kind == "archive":
            if self.crs is not None or self.transform is not None or self.nodata is not None:
                raise SourceLockError("archive source metadata must be null")
            if {member.site_id for member in members} != set(JAXA_SITE_IDS):
                raise SourceLockError("archive member inventory must contain the exact six JAXA sites")
        else:
            raise SourceLockError("source artifact kind is invalid")

    @classmethod
    def from_file(
        cls,
        source_id: str,
        path: str | Path,
        *,
        citation: str,
        license: str,
        final_url: str,
    ) -> "PolarSourceLock":
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise SourceLockError("source must be an existing regular file")
        common = {
            "schema": SOURCE_LOCK_SCHEMA,
            "source_id": source_id,
            "filename": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": _file_sha256(source),
            "citation": citation,
            "license": license,
            "final_url": final_url,
        }
        if source.suffix.lower() == ".zip":
            return cls(
                **common,
                artifact_kind="archive",
                crs=None,
                transform=None,
                nodata=None,
                archive_members=_archive_member_inventory(source),
            )
        crs, transform, nodata = _read_raster_metadata(
            source,
            allow_null_nodata=source_id == _NULL_NODATA_RASTER_SOURCE_ID,
        )
        return cls(
            **common,
            artifact_kind="raster",
            crs=crs,
            transform=transform,
            nodata=nodata,
            archive_members=(),
        )

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "PolarSourceLock":
        if "nodata" not in document:
            raise SourceLockError("source lock NoData must be explicit")
        members = document.get("archive_members", ())
        if not isinstance(members, list):
            raise SourceLockError("archive member inventory is invalid")
        return cls(
            schema=document.get("schema"), source_id=document.get("source_id"),
            filename=document.get("filename"), size_bytes=document.get("size_bytes"),
            sha256=document.get("sha256"), crs=document.get("crs"),
            transform=document.get("transform"), nodata=document.get("nodata"),
            artifact_kind=document.get("artifact_kind", "raster"),
            citation=document.get("citation", ""), license=document.get("license", ""),
            final_url=document.get("final_url", ""),
            archive_members=tuple(ArchiveMemberLock.from_dict(item) for item in members if isinstance(item, dict)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema, "source_id": self.source_id, "filename": self.filename,
            "size_bytes": self.size_bytes, "sha256": self.sha256, "artifact_kind": self.artifact_kind,
            "citation": self.citation, "license": self.license, "final_url": self.final_url,
            "crs": self.crs, "transform": self.transform, "nodata": self.nodata,
            "archive_members": [member.to_dict() for member in self.archive_members],
        }


def verify_source_lock(data_root: str | Path, source_lock: PolarSourceLock, *, repository_root: str | Path) -> Path:
    """Return a verified source path, rejecting unsafe roots, drift, and fake archive CRS."""
    root = _validate_data_root(data_root, repository_root)
    source = root / source_lock.filename
    if source.is_symlink():
        raise SourceLockError("locked source must not be a symbolic link")
    if not source.is_file():
        raise SourceLockError("locked source is missing")
    if source_lock.artifact_kind == "archive":
        if source.suffix.lower() != ".zip":
            raise SourceLockError("archive lock does not match source type")
        if _archive_member_inventory(source) != source_lock.archive_members:
            raise SourceLockError("archive member inventory does not match")
    else:
        crs, transform, nodata = _read_raster_metadata(
            source,
            allow_null_nodata=source_lock.source_id == _NULL_NODATA_RASTER_SOURCE_ID,
        )
        if (crs, transform, nodata) != (source_lock.crs, source_lock.transform, source_lock.nodata):
            raise SourceLockError("locked source raster metadata does not match")
    if source.stat().st_size != source_lock.size_bytes:
        raise SourceLockError("locked source size does not match")
    if _file_sha256(source) != source_lock.sha256:
        raise SourceLockError("locked source sha256 does not match")
    return source


def validate_external_data_root(data_root: str | Path, *, repository_root: str | Path) -> Path:
    return _validate_data_root(data_root, repository_root)


def write_aggregate_source_lock(path: str | Path, data_root: str | Path, locks: tuple[PolarSourceLock, ...], *, repository_root: str | Path) -> None:
    """Write a portable aggregate lock with a relative source-root reference."""
    target = _validate_external_output(path, repository_root)
    root = _validate_data_root(data_root, repository_root)
    if not locks or len({lock.source_id for lock in locks}) != len(locks):
        raise SourceLockError("aggregate source locks are invalid")
    relative_root = os.path.relpath(root, start=target.parent)
    if Path(relative_root).is_absolute():
        raise SourceLockError("aggregate source root must be relative")
    document = {
        "schema": AGGREGATE_SOURCE_LOCK_SCHEMA,
        "source_root": relative_root,
        "sources": [lock.to_dict() for lock in sorted(locks, key=lambda item: item.source_id)],
    }
    _atomic_json_write(target, document)


def load_aggregate_source_lock(path: str | Path, *, repository_root: str | Path) -> tuple[Path, tuple[PolarSourceLock, ...]]:
    """Resolve and validate a portable aggregate lock without exposing local paths."""
    target = Path(path)
    if not target.is_absolute() or target.is_symlink() or not target.is_file():
        raise SourceLockError("aggregate lock must be an existing absolute regular file")
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceLockError("aggregate lock is not valid UTF-8 JSON") from error
    if not isinstance(document, dict) or document.get("schema") != AGGREGATE_SOURCE_LOCK_SCHEMA:
        raise SourceLockError("unsupported aggregate lock schema")
    relative_root = document.get("source_root")
    sources = document.get("sources")
    if not isinstance(relative_root, str) or Path(relative_root).is_absolute() or not isinstance(sources, list):
        raise SourceLockError("aggregate lock content is invalid")
    root = _validate_data_root(target.parent / relative_root, repository_root)
    locks = tuple(PolarSourceLock.from_dict(source) for source in sources if isinstance(source, dict))
    if not locks or len(locks) != len(sources) or len({lock.source_id for lock in locks}) != len(locks):
        raise SourceLockError("aggregate source locks are invalid")
    for lock in locks:
        verify_source_lock(root, lock, repository_root=repository_root)
    return root, locks


def _validate_data_root(data_root: str | Path, repository_root: str | Path) -> Path:
    root = Path(data_root)
    repository = Path(repository_root)
    if not root.is_absolute():
        raise SourceLockError("data root must be absolute")
    if not root.is_dir():
        raise SourceLockError("data root must be an existing directory")
    if _has_symlink_component(root):
        raise SourceLockError("data root must not be a symbolic link")
    resolved_root = root.resolve(strict=True)
    resolved_repository = repository.resolve(strict=True)
    if resolved_root.is_relative_to(resolved_repository):
        raise SourceLockError("data root must be outside the repository")
    return resolved_root


def _validate_external_output(path: str | Path, repository_root: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute() or target.is_symlink() or not target.parent.is_dir():
        raise SourceLockError("aggregate lock output must be an absolute regular path")
    if _has_symlink_component(target.parent):
        raise SourceLockError("aggregate lock output must not use a symbolic link")
    resolved_repository = Path(repository_root).resolve(strict=True)
    if target.parent.resolve(strict=True).is_relative_to(resolved_repository):
        raise SourceLockError("aggregate lock output must be outside the repository")
    return target


def _archive_member_inventory(source: Path) -> tuple[ArchiveMemberLock, ...]:
    try:
        with ZipFile(source) as archive:
            members: list[ArchiveMemberLock] = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member_path = _validated_archive_member_path(info.filename)
                if member_path.suffix.lower() not in {".tif", ".tiff", ".dem", ".dtm", ".img", ".vrt"}:
                    continue
                site_token = member_path.name.split("_", 1)[0].upper()
                if site_token not in JAXA_SITE_IDS:
                    raise SourceLockError("archive raster member has no exact JAXA site id")
                contents = archive.read(info)
                crs, transform, nodata = _read_raster_metadata_bytes(contents)
                members.append(ArchiveMemberLock(
                    path=info.filename, site_id=site_token, size_bytes=info.file_size,
                    sha256=sha256(contents).hexdigest(), crs=crs, transform=transform, nodata=nodata,
                ))
    except (OSError, BadZipFile) as error:
        raise SourceLockError("archive inventory cannot be read") from error
    ordered = tuple(sorted(members, key=lambda item: item.path))
    if {member.site_id for member in ordered} != set(JAXA_SITE_IDS):
        raise SourceLockError("archive member inventory must contain the exact six JAXA sites")
    return ordered


def _validated_archive_member_path(path: object) -> PurePosixPath:
    """Accept only canonical relative POSIX archive paths on every host OS."""
    if not isinstance(path, str) or not path or "\\" in path:
        raise SourceLockError("archive member path is unsafe")
    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or "." in posix_path.parts
        or posix_path.as_posix() != path
    ):
        raise SourceLockError("archive member path is unsafe")
    return posix_path


def _read_raster_metadata(
    source: Path,
    *,
    allow_null_nodata: bool = False,
) -> tuple[str, tuple[float, float, float, float, float, float], float | None]:
    try:
        with rasterio.open(source) as dataset:
            return _dataset_metadata(dataset, allow_null_nodata=allow_null_nodata)
    except SourceLockError:
        raise
    except (OSError, rasterio.errors.RasterioError) as error:
        raise SourceLockError("raster metadata cannot be read") from error


def _read_raster_metadata_bytes(contents: bytes) -> tuple[str, tuple[float, float, float, float, float, float], float]:
    try:
        with MemoryFile(contents) as memory, memory.open() as dataset:
            crs, transform, nodata = _dataset_metadata(dataset)
            if nodata is None:
                raise SourceLockError("raster NoData is missing")
            return crs, transform, nodata
    except SourceLockError:
        raise
    except (OSError, rasterio.errors.RasterioError) as error:
        raise SourceLockError("archive raster metadata cannot be read") from error


def _dataset_metadata(
    dataset: rasterio.io.DatasetReader,
    *,
    allow_null_nodata: bool = False,
) -> tuple[str, tuple[float, float, float, float, float, float], float | None]:
    if dataset.crs is None:
        raise SourceLockError("raster CRS is missing")
    if dataset.transform is None or dataset.transform.is_identity:
        raise SourceLockError("raster transform is missing")
    if dataset.nodata is None and not allow_null_nodata:
        raise SourceLockError("raster NoData is missing")
    nodata = None if dataset.nodata is None else float(dataset.nodata)
    return dataset.crs.to_string(), tuple(float(value) for value in dataset.transform)[:6], nodata


def _atomic_json_write(path: Path, document: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists() or temporary.is_symlink():
        raise SourceLockError("existing aggregate lock part prevents write")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(document, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as error:
        raise SourceLockError("aggregate lock could not be written atomically") from error


def _file_sha256(source: Path) -> str:
    digest = sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_transform(transform: object) -> tuple[float, float, float, float, float, float]:
    try:
        normalized = tuple(float(value) for value in transform)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise SourceLockError("raster transform is invalid") from error
    if len(normalized) != 6:
        raise SourceLockError("raster transform is invalid")
    return normalized  # type: ignore[return-value]


def _validate_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise SourceLockError(f"{label} sha256 is invalid")


def _require_https(url: object, label: str) -> None:
    from urllib.parse import urlparse
    if not isinstance(url, str) or urlparse(url).scheme != "https":
        raise SourceLockError(f"{label} must use HTTPS")


def _has_symlink_component(path: Path) -> bool:
    current = path
    while current != current.parent:
        if current.is_symlink():
            return True
        current = current.parent
    return False


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "AGGREGATE_SOURCE_LOCK_SCHEMA", "ArchiveMemberLock", "JAXA_SITE_IDS", "PolarSourceLock",
    "SOURCE_LOCK_SCHEMA", "SourceLockError", "load_aggregate_source_lock", "validate_external_data_root",
    "verify_source_lock", "write_aggregate_source_lock",
]

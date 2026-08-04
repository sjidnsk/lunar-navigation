"""Fail-closed source locks for external lunar polar data."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import rasterio


SOURCE_LOCK_SCHEMA = "lunar-polar-source-lock/v1"


class SourceLockError(ValueError):
    """A data root or its locked source did not pass verification."""


@dataclass(frozen=True)
class PolarSourceLock:
    """Portable content and raster-metadata identity for one official source."""

    schema: str
    source_id: str
    filename: str
    size_bytes: int
    sha256: str
    crs: str | None
    transform: tuple[float, float, float, float, float, float] | None
    nodata: float | None

    def __post_init__(self) -> None:
        if self.schema != SOURCE_LOCK_SCHEMA:
            raise SourceLockError("unsupported source lock schema")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise SourceLockError("source id is missing")
        if not isinstance(self.filename, str) or Path(self.filename).name != self.filename:
            raise SourceLockError("source lock filename must not contain a path")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise SourceLockError("source lock size is invalid")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise SourceLockError("source lock sha256 is invalid")
        if self.crs is not None and not isinstance(self.crs, str):
            raise SourceLockError("source lock CRS is invalid")
        if self.nodata is not None and (
            not isinstance(self.nodata, (int, float)) or isinstance(self.nodata, bool)
        ):
            raise SourceLockError("source lock NoData is invalid")
        if self.transform is not None:
            try:
                normalized_transform = tuple(float(value) for value in self.transform)
            except (TypeError, ValueError) as error:
                raise SourceLockError("source lock transform is invalid") from error
            if len(normalized_transform) != 6:
                raise SourceLockError("source lock transform is invalid")
            object.__setattr__(self, "transform", normalized_transform)

    @classmethod
    def from_file(cls, source_id: str, path: str | Path) -> "PolarSourceLock":
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise SourceLockError("source must be an existing regular file")
        if source.suffix.lower() == ".zip":
            return cls(
                schema=SOURCE_LOCK_SCHEMA,
                source_id=source_id,
                filename=source.name,
                size_bytes=source.stat().st_size,
                sha256=_file_sha256(source),
                crs=None,
                transform=None,
                nodata=None,
            )
        crs, transform, nodata = _read_raster_metadata(source)
        return cls(
            schema=SOURCE_LOCK_SCHEMA,
            source_id=source_id,
            filename=source.name,
            size_bytes=source.stat().st_size,
            sha256=_file_sha256(source),
            crs=crs,
            transform=transform,
            nodata=nodata,
        )

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "PolarSourceLock":
        return cls(
            schema=document.get("schema"),
            source_id=document.get("source_id"),
            filename=document.get("filename"),
            size_bytes=document.get("size_bytes"),
            sha256=document.get("sha256"),
            crs=document.get("crs"),
            transform=document.get("transform"),
            nodata=document.get("nodata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_id": self.source_id,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": self.nodata,
        }


def verify_source_lock(
    data_root: str | Path,
    source_lock: PolarSourceLock,
    *,
    repository_root: str | Path,
) -> Path:
    """Return a verified source path, rejecting unsafe roots and metadata drift."""
    root = _validate_data_root(data_root, repository_root)
    source = root / source_lock.filename
    if source.is_symlink():
        raise SourceLockError("locked source must not be a symbolic link")
    if not source.is_file():
        raise SourceLockError("locked source is missing")
    if source.suffix.lower() != ".zip":
        crs, transform, nodata = _read_raster_metadata(source)
        if source_lock.crs is None or source_lock.transform is None or source_lock.nodata is None:
            raise SourceLockError("raster source lock metadata is missing")
        if crs != source_lock.crs:
            raise SourceLockError("locked source CRS does not match")
        if transform != source_lock.transform:
            raise SourceLockError("locked source transform does not match")
        if nodata != source_lock.nodata:
            raise SourceLockError("locked source NoData does not match")
    if source.stat().st_size != source_lock.size_bytes:
        raise SourceLockError("locked source size does not match")
    if _file_sha256(source) != source_lock.sha256:
        raise SourceLockError("locked source sha256 does not match")
    return source


def validate_external_data_root(
    data_root: str | Path, *, repository_root: str | Path
) -> Path:
    """Validate the root accepted by download and source-lock operations."""
    return _validate_data_root(data_root, repository_root)


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


def _has_symlink_component(path: Path) -> bool:
    current = path
    while current != current.parent:
        if current.is_symlink():
            return True
        current = current.parent
    return False


def _read_raster_metadata(
    source: Path,
) -> tuple[str, tuple[float, float, float, float, float, float], float]:
    try:
        with rasterio.open(source) as dataset:
            if dataset.crs is None:
                raise SourceLockError("raster CRS is missing")
            if dataset.transform is None or dataset.transform.is_identity:
                raise SourceLockError("raster transform is missing")
            if dataset.nodata is None:
                raise SourceLockError("raster NoData is missing")
            return (
                dataset.crs.to_string(),
                tuple(float(value) for value in dataset.transform)[:6],
                float(dataset.nodata),
            )
    except SourceLockError:
        raise
    except (OSError, rasterio.errors.RasterioError) as error:
        raise SourceLockError("raster metadata cannot be read") from error


def _file_sha256(source: Path) -> str:
    digest = sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PolarSourceLock",
    "SOURCE_LOCK_SCHEMA",
    "SourceLockError",
    "validate_external_data_root",
    "verify_source_lock",
]

#!/usr/bin/env python3
"""Fetch official lunar-polar sources into an external, source-locked data root."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "lunar_policy_training"
sys.path.insert(0, str(PACKAGE_ROOT))

from lunar_policy_training.polar_data.source_lock import (  # noqa: E402
    PolarSourceLock,
    SourceLockError,
    validate_external_data_root,
    verify_source_lock,
)


REGISTRY_SCHEMA = "lunar-polar-source-registry/v1"


class PolarFetchError(ValueError):
    """The registry, destination, or remote response was unsafe or invalid."""


def load_registry(path: str | Path) -> tuple[dict[str, str], ...]:
    """Load the narrowly versioned official-source registry."""
    registry_path = Path(path)
    if registry_path.is_symlink() or not registry_path.is_file():
        raise PolarFetchError("registry must be an existing regular file")
    try:
        document = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PolarFetchError("registry is not valid UTF-8 JSON") from error
    if not isinstance(document, dict) or document.get("schema") != REGISTRY_SCHEMA:
        raise PolarFetchError("unsupported polar source registry schema")
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PolarFetchError("registry sources are missing")
    validated: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise PolarFetchError("registry source must be an object")
        source_id = source.get("id")
        filename = source.get("filename")
        url = source.get("url") or source.get("record_api")
        if not isinstance(source_id, str) or not source_id or source_id in seen_ids:
            raise PolarFetchError("registry source id is invalid")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise PolarFetchError("registry filename is invalid")
        if not isinstance(url, str) or urlparse(url).scheme != "https":
            raise PolarFetchError("registry URL must use HTTPS")
        if set(source) - {"id", "filename", "url", "record_api"}:
            raise PolarFetchError("registry source has unsupported fields")
        if ("url" in source) == ("record_api" in source):
            raise PolarFetchError("registry source must contain exactly one URL")
        validated.append({key: value for key, value in source.items() if isinstance(value, str)})
        seen_ids.add(source_id)
    return tuple(validated)


def fetch_source(
    source: Mapping[str, str],
    data_root: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    """Fetch one source with a same-directory part file and durable lock sidecar."""
    root = validate_external_data_root(data_root, repository_root=repository_root)
    source_id = source.get("id")
    filename = source.get("filename")
    if not isinstance(source_id, str) or not isinstance(filename, str):
        raise PolarFetchError("source id and filename are required")
    if Path(filename).name != filename:
        raise PolarFetchError("source filename is invalid")
    destination = root / filename
    lock_path = root / f"{filename}.source-lock.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise PolarFetchError("destination must not be a symbolic link")
        if not lock_path.is_file() or lock_path.is_symlink():
            raise PolarFetchError("existing source has no reusable matching lock")
        lock = _read_lock(lock_path)
        if lock.source_id != source_id:
            raise PolarFetchError("existing source lock has a different source id")
        try:
            return verify_source_lock(root, lock, repository_root=repository_root)
        except SourceLockError as error:
            raise PolarFetchError("existing source does not match its lock") from error
    if lock_path.exists() or lock_path.is_symlink():
        raise PolarFetchError("stale source lock prevents download")

    part = root / f"{filename}.part"
    if part.exists() or part.is_symlink():
        raise PolarFetchError("existing part file prevents download")
    url = _source_download_url(source)
    _download_atomic(url, part, destination)
    try:
        lock = PolarSourceLock.from_file(source_id, destination)
        _write_lock_atomic(lock_path, lock)
    except Exception as error:
        if isinstance(error, (PolarFetchError, SourceLockError)):
            raise
        raise PolarFetchError("downloaded source lock could not be written") from error
    return destination


def _source_download_url(source: Mapping[str, str]) -> str:
    direct = source.get("url")
    if direct is not None:
        return _require_https(direct)
    record_api = source.get("record_api")
    filename = source.get("filename")
    if record_api is None or filename is None:
        raise PolarFetchError("source URL is missing")
    try:
        with urlopen(_require_https(record_api), timeout=30) as response:
            record = json.load(response)
    except (OSError, json.JSONDecodeError) as error:
        raise PolarFetchError("Zenodo record could not be read") from error
    if not isinstance(record, dict) or not isinstance(record.get("files"), list):
        raise PolarFetchError("Zenodo record files are missing")
    for entry in record["files"]:
        if not isinstance(entry, dict) or entry.get("key") != filename:
            continue
        links = entry.get("links")
        if isinstance(links, dict) and isinstance(links.get("self"), str):
            return _require_https(links["self"])
    raise PolarFetchError("Zenodo record does not contain the requested filename")


def _download_atomic(url: str, part: Path, destination: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            with urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, destination)
        _fsync_directory(destination.parent)
    except OSError as error:
        raise PolarFetchError("source download failed without reuse") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_lock(path: Path) -> PolarSourceLock:
    try:
        document: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise PolarFetchError("source lock is not an object")
        return PolarSourceLock.from_dict(document)
    except (OSError, UnicodeError, json.JSONDecodeError, SourceLockError) as error:
        raise PolarFetchError("source lock is invalid") from error


def _write_lock_atomic(path: Path, lock: PolarSourceLock) -> None:
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists() or temporary.is_symlink():
        raise PolarFetchError("existing lock part file prevents download")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(lock.to_dict(), sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as error:
        raise PolarFetchError("source lock could not be written atomically") from error


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_https(url: str) -> str:
    if urlparse(url).scheme != "https":
        raise PolarFetchError("source URL must use HTTPS")
    return url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_sources/polar_source_registry_v1.json",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--source", action="append", dest="source_ids")
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        registry = load_registry(arguments.registry)
        selected = tuple(
            source
            for source in registry
            if arguments.source_ids is None or source["id"] in arguments.source_ids
        )
        if not selected or (arguments.source_ids and len(selected) != len(set(arguments.source_ids))):
            raise PolarFetchError("requested registry source is missing")
        for source in selected:
            print(fetch_source(source, arguments.data_root, repository_root=arguments.repository_root))
    except (PolarFetchError, SourceLockError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fetch official lunar-polar sources into an external aggregate source lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "lunar_policy_training"
sys.path.insert(0, str(PACKAGE_ROOT))

from lunar_policy_training.polar_data.source_lock import (  # noqa: E402
    PolarSourceLock,
    SourceLockError,
    validate_external_data_root,
    verify_source_lock,
    write_aggregate_source_lock,
)


REGISTRY_SCHEMA = "lunar-polar-source-registry/v1"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-\d+/\d+$")


class PolarFetchError(ValueError):
    """The registry, destination, or remote response was unsafe or invalid."""


def load_registry(path: str | Path) -> tuple[dict[str, object], ...]:
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
    validated: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    allowed = {"id", "filename", "url", "record_api", "citation", "license", "expected_size_bytes"}
    for source in sources:
        if not isinstance(source, dict) or set(source) - allowed:
            raise PolarFetchError("registry source is invalid")
        source_id, filename = source.get("id"), source.get("filename")
        citation, license_text = source.get("citation"), source.get("license")
        endpoint = source.get("url") or source.get("record_api")
        if not isinstance(source_id, str) or not source_id or source_id in seen_ids:
            raise PolarFetchError("registry source id is invalid")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise PolarFetchError("registry filename is invalid")
        if not isinstance(citation, str) or not citation or not isinstance(license_text, str) or not license_text:
            raise PolarFetchError("registry citation or license is missing")
        if not isinstance(endpoint, str) or urlparse(endpoint).scheme != "https":
            raise PolarFetchError("registry URL must use HTTPS")
        if ("url" in source) == ("record_api" in source):
            raise PolarFetchError("registry source must contain exactly one URL")
        expected_size = source.get("expected_size_bytes")
        if expected_size is not None and (not isinstance(expected_size, int) or expected_size < 0):
            raise PolarFetchError("registry expected size is invalid")
        validated.append(dict(source))
        seen_ids.add(source_id)
    return tuple(validated)


def fetch_source(source: Mapping[str, object], data_root: str | Path, *, repository_root: str | Path) -> Path:
    """Fetch one source; only matching source-id/filename locks allow reuse."""
    root = validate_external_data_root(data_root, repository_root=repository_root)
    source_id = _required_string(source, "id")
    filename = _required_string(source, "filename")
    citation, license_text = _required_string(source, "citation"), _required_string(source, "license")
    if Path(filename).name != filename:
        raise PolarFetchError("source filename is invalid")
    destination, lock_path = root / filename, root / f"{filename}.source-lock.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise PolarFetchError("destination must not be a symbolic link")
        if not lock_path.is_file() or lock_path.is_symlink():
            raise PolarFetchError("existing source has no reusable matching lock")
        lock = _read_lock(lock_path)
        if lock.source_id != source_id or lock.filename != filename:
            raise PolarFetchError("existing source lock source id or filename does not match")
        try:
            return verify_source_lock(root, lock, repository_root=repository_root)
        except SourceLockError as error:
            raise PolarFetchError("existing source does not match its lock") from error
    if lock_path.exists() or lock_path.is_symlink():
        raise PolarFetchError("stale source lock prevents download")

    part = root / f"{filename}.part"
    if part.is_symlink():
        raise PolarFetchError("part file must not be a symbolic link")
    url = _source_download_url(source)
    final_url = _download_atomic(url, part, destination)
    try:
        lock = PolarSourceLock.from_file(
            source_id, destination, citation=citation, license=license_text, final_url=final_url
        )
        expected_size = source.get("expected_size_bytes")
        if expected_size is not None and lock.size_bytes != expected_size:
            raise PolarFetchError("downloaded source size does not match registry")
        _write_lock_atomic(lock_path, lock)
    except (PolarFetchError, SourceLockError):
        raise
    except Exception as error:
        raise PolarFetchError("downloaded source lock could not be written") from error
    return destination


def _source_download_url(source: Mapping[str, object]) -> str:
    direct = source.get("url")
    if isinstance(direct, str):
        return _require_https(direct)
    record_api, filename = _required_string(source, "record_api"), _required_string(source, "filename")
    try:
        with urlopen(Request(_require_https(record_api)), timeout=30) as response:
            _require_https(response.geturl())
            record = json.load(response)
    except (OSError, json.JSONDecodeError) as error:
        raise PolarFetchError("Zenodo record could not be read") from error
    if not isinstance(record, dict) or not isinstance(record.get("files"), list):
        raise PolarFetchError("Zenodo record files are missing")
    for entry in record["files"]:
        links = entry.get("links") if isinstance(entry, dict) else None
        if isinstance(entry, dict) and entry.get("key") == filename and isinstance(links, dict):
            if isinstance(links.get("self"), str):
                return _require_https(links["self"])
    raise PolarFetchError("Zenodo record does not contain the requested filename")


def _download_atomic(url: str, part: Path, destination: Path) -> str:
    """Durably resume only an exact 206 response, otherwise safely restart."""
    offset = part.stat().st_size if part.exists() else 0
    request = Request(url, headers={"Range": f"bytes={offset}-"}) if offset else Request(url)
    try:
        with urlopen(request, timeout=60) as response:
            final_url = _require_https(response.geturl())
            status = response.status if hasattr(response, "status") else response.getcode()
            append = False
            if offset:
                content_range = response.headers.get("Content-Range")
                match = _CONTENT_RANGE.fullmatch(content_range or "")
                if status == 206 and match and int(match.group(1)) == offset:
                    append = True
                elif status != 200:
                    raise PolarFetchError("range response is not a validated partial response")
            elif status != 200:
                raise PolarFetchError("initial response must be HTTP 200")
            mode = "ab" if append else "wb"
            with part.open(mode) as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    except PolarFetchError:
        raise
    except OSError as error:
        raise PolarFetchError("source download interrupted; part retained") from error
    if destination.exists() or destination.is_symlink():
        raise PolarFetchError("destination appeared during download; part retained")
    os.replace(part, destination)
    _fsync_directory(destination.parent)
    return final_url


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
        raise PolarFetchError("existing lock part file prevents write")
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


def _required_string(source: Mapping[str, object], field: str) -> str:
    value = source.get(field)
    if not isinstance(value, str) or not value:
        raise PolarFetchError(f"source {field} is required")
    return value


def _require_https(url: str) -> str:
    if urlparse(url).scheme != "https":
        raise PolarFetchError("source final URL must use HTTPS")
    return url


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path(__file__).resolve().parents[1] / "data_sources/polar_source_registry_v1.json")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--lock-output", required=True, type=Path)
    parser.add_argument("--source", action="append", dest="source_ids")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        registry = load_registry(arguments.registry)
        selected = tuple(source for source in registry if arguments.source_ids is None or source["id"] in arguments.source_ids)
        if not selected or (arguments.source_ids and len(selected) != len(set(arguments.source_ids))):
            raise PolarFetchError("requested registry source is missing")
        paths = [fetch_source(source, arguments.data_root, repository_root=arguments.repository_root) for source in selected]
        locks = tuple(_read_lock(path.with_name(f"{path.name}.source-lock.json")) for path in paths)
        write_aggregate_source_lock(arguments.lock_output, arguments.data_root, locks, repository_root=arguments.repository_root)
        for path in paths:
            print(path.name)
    except (PolarFetchError, SourceLockError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

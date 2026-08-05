from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest

from lunar_policy_training.capability_freeze import (
    CAPABILITY_TYPES,
    CapabilityFreezeError,
    FrozenCapabilityEnvironmentFactory,
    load_frozen_capability_bundle,
)


PLATFORMS = ("WHEELED", "LEGGED", "HOPPER")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bundle(
    root: Path,
    *,
    formal_eligible: bool = True,
    test_only: bool = False,
    proxy: bool = False,
    reverse_fields: bool = False,
) -> Path:
    root.mkdir()
    entries = []
    for index, platform in enumerate(PLATFORMS):
        version = f"{platform.lower()}-2026.08"
        content_items = [
            ("schema", CAPABILITY_TYPES[platform]),
            ("platform_type", platform),
            ("capability_version", version),
            (
                "content",
                {
                    "maximum_slope_rad": 0.20 + index * 0.01,
                    "base_frame_id": "base_link",
                },
            ),
        ]
        if reverse_fields:
            content_items.reverse()
        content_bytes = (
            json.dumps(dict(content_items), separators=(",", ":")) + "\n"
        ).encode("utf-8")
        content_path = Path(platform.lower()) / "capability.json"
        (root / content_path).parent.mkdir()
        (root / content_path).write_bytes(content_bytes)
        resources = []
        for kind, suffix in (
            ("document", "md"),
            ("urdf", "urdf"),
            ("mesh", "stl"),
        ):
            relative = Path(platform.lower()) / f"resource.{suffix}"
            data = f"{platform}-{kind}\n".encode("utf-8")
            (root / relative).write_bytes(data)
            resources.append(
                {"kind": kind, "path": relative.as_posix(), "sha256": _sha256(data)}
            )
        entries.append(
            {
                "platform_type": platform,
                "capability_type": CAPABILITY_TYPES[platform],
                "capability_version": version,
                "content_path": content_path.as_posix(),
                "content_file_sha256": _sha256(content_bytes),
                "resources": resources,
            }
        )
    if reverse_fields:
        entries.reverse()
    lock = {
        "schema": "lunar-training-capability-freeze/v1",
        "formal_eligible": formal_eligible,
        "test_only": test_only,
        "proxy": proxy,
        "platforms": entries,
    }
    lock_path = root / "capability-lock.json"
    lock_path.write_text(
        json.dumps(lock, sort_keys=reverse_fields, indent=2) + "\n",
        encoding="utf-8",
    )
    return lock_path


def test_formal_bundle_is_canonical_pickle_safe_and_relocation_independent(
    tmp_path: Path,
) -> None:
    """Would fail if JSON order, absolute location, or pickle changed identity."""
    first = load_frozen_capability_bundle(
        _write_bundle(tmp_path / "first"), run_kind="formal"
    )
    second = load_frozen_capability_bundle(
        _write_bundle(tmp_path / "relocated", reverse_fields=True),
        run_kind="formal",
    )

    assert tuple(item.platform_type for item in first.platforms) == PLATFORMS
    assert first.formal_eligible is True
    assert first.bundle_sha256 == second.bundle_sha256
    assert pickle.loads(pickle.dumps(first)) == first
    assert str(tmp_path) not in repr(first)


@pytest.mark.parametrize("failure", ("missing", "duplicate", "unknown"))
def test_bundle_requires_exactly_three_known_platforms(
    tmp_path: Path, failure: str
) -> None:
    """Would fail if formal training could silently omit or alias one platform."""
    lock_path = _write_bundle(tmp_path / failure)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if failure == "missing":
        payload["platforms"].pop()
    elif failure == "duplicate":
        payload["platforms"][-1] = dict(payload["platforms"][0])
    else:
        payload["platforms"][-1]["platform_type"] = "FLYING"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CapabilityFreezeError, match="platform"):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.__setitem__("schema", "wrong/v1"), "schema"),
        (
            lambda payload: payload["platforms"][0].__setitem__(
                "capability_type", "unknown/v9"
            ),
            "type",
        ),
        (
            lambda payload: payload["platforms"][0].__setitem__(
                "capability_version", "wrong"
            ),
            "version",
        ),
    ),
)
def test_bundle_rejects_schema_type_or_version_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    """Would fail if lock metadata could diverge from parsed capability content."""
    lock_path = _write_bundle(tmp_path / message)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    mutation(payload)
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CapabilityFreezeError, match=message):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


@pytest.mark.parametrize("kind", ("document", "urdf", "mesh"))
def test_bundle_rejects_each_resource_kind_hash_drift(
    tmp_path: Path, kind: str
) -> None:
    """Would fail if a referenced geometry/document resource escaped closure hash."""
    lock_path = _write_bundle(tmp_path / kind)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    resource = next(
        item
        for item in payload["platforms"][0]["resources"]
        if item["kind"] == kind
    )
    resource["sha256"] = "0" * 64
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CapabilityFreezeError, match="hash"):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


def test_bundle_rejects_content_drift_and_path_escape(tmp_path: Path) -> None:
    """Would fail if capability bytes or a path outside closure were accepted."""
    drift_lock = _write_bundle(tmp_path / "drift")
    drift_payload = json.loads(drift_lock.read_text(encoding="utf-8"))
    drift_payload["platforms"][0]["content_file_sha256"] = "0" * 64
    drift_lock.write_text(json.dumps(drift_payload), encoding="utf-8")
    with pytest.raises(CapabilityFreezeError, match="hash"):
        load_frozen_capability_bundle(drift_lock, run_kind="formal")

    escape_lock = _write_bundle(tmp_path / "escape")
    escape_payload = json.loads(escape_lock.read_text(encoding="utf-8"))
    escape_payload["platforms"][0]["content_path"] = "../outside.json"
    escape_lock.write_text(json.dumps(escape_payload), encoding="utf-8")
    with pytest.raises(CapabilityFreezeError, match="path"):
        load_frozen_capability_bundle(escape_lock, run_kind="formal")


@pytest.mark.parametrize(
    "flags",
    (
        {"formal_eligible": False},
        {"test_only": True},
        {"proxy": True},
    ),
)
def test_formal_rejects_nonformal_test_or_proxy_lock(
    tmp_path: Path, flags: dict[str, bool]
) -> None:
    """Would fail if a development fixture could become a formal capability."""
    lock_path = _write_bundle(tmp_path / next(iter(flags)), **flags)
    with pytest.raises(CapabilityFreezeError, match="formal"):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


def _record_injected_capability(worker_index, platform_type, capability, scenario):
    return worker_index, platform_type, capability, scenario


def test_environment_factory_uses_loaded_bundle_without_source_reread(
    tmp_path: Path,
) -> None:
    """Would fail if spawned workers reopened mutable capability source files."""
    lock_path = _write_bundle(tmp_path / "closure")
    bundle = load_frozen_capability_bundle(lock_path, run_kind="formal")
    factory = FrozenCapabilityEnvironmentFactory(
        bundle=bundle,
        scenario_schedule_id="nasa-polar-train/v1",
        builder=_record_injected_capability,
    )
    lock_path.unlink()

    result = pickle.loads(pickle.dumps(factory))(7, "LEGGED")

    assert result[2] == bundle.for_platform("LEGGED")
    assert result[3].capability_version == result[2].capability_version
    assert result[3].capability_sha256 == result[2].content_sha256
    assert result[3].scenario_schedule_id == "nasa-polar-train/v1"

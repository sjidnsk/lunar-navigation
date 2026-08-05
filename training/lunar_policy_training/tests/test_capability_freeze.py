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
        platform_root = Path(platform.lower())
        platform_document_path = platform_root / "platform.yaml"
        observation_document_path = platform_root / "observation.json"
        urdf_path = platform_root / "rover.urdf"
        mesh_path = platform_root / "body.stl"
        resource_bytes = {
            ("document", platform_document_path): b"test-only platform source\n",
            ("document", observation_document_path): (
                b'{"sensor_range_m":25.0,"sensor_fov_deg":90.0}\n'
            ),
            ("urdf", urdf_path): (
                b'<robot name="test"><link name="base_link"/></robot>\n'
            ),
            ("mesh", mesh_path): b"solid test\nendsolid test\n",
        }
        resources = []
        for (kind, relative), data in resource_bytes.items():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_bytes(data)
            resources.append(
                {"kind": kind, "path": relative.as_posix(), "sha256": _sha256(data)}
            )
        content_items = [
            ("schema", CAPABILITY_TYPES[platform]),
            ("platform_type", platform),
            ("capability_version", version),
            (
                "content",
                {
                    "platform_id": f"test-only-{platform.lower()}",
                    "base_frame_id": "base_link",
                    "platform_document_path": platform_document_path.as_posix(),
                    "observation_document_path": (
                        observation_document_path.as_posix()
                    ),
                    "urdf_path": urdf_path.as_posix(),
                    "mesh_paths": [mesh_path.as_posix()],
                    "observation": {
                        "sensor_range_m": 25.0 + index,
                        "sensor_fov_deg": 90.0,
                    },
                    "capability": _test_typed_content(platform),
                },
            ),
        ]
        if reverse_fields:
            content_items.reverse()
        content_bytes = (
            json.dumps(dict(content_items), separators=(",", ":")) + "\n"
        ).encode("utf-8")
        content_path = Path(platform.lower()) / "capability.json"
        (root / content_path).parent.mkdir(parents=True, exist_ok=True)
        (root / content_path).write_bytes(content_bytes)
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


def _test_typed_content(platform: str) -> dict[str, object]:
    if platform == "WHEELED":
        return {
            "footprint_xy_m": [
                [-0.2, -0.2],
                [0.2, -0.2],
                [0.2, 0.2],
                [-0.2, 0.2],
            ],
            "minimum_body_z_m": -0.1,
            "maximum_body_z_m": 0.5,
            "maximum_slope_rad": 0.4,
            "maximum_obstacle_height_m": 0.2,
            "maximum_forward_speed_mps": 1.0,
            "maximum_reverse_speed_mps": 0.5,
            "maximum_spin_rate_radps": 1.0,
            "maximum_acceleration_mps2": 1.0,
            "maximum_braking_deceleration_mps2": 1.0,
            "maximum_yaw_acceleration_radps2": 1.0,
            "maximum_lateral_acceleration_mps2": 1.0,
            "maximum_curvature_per_m": 1.0,
            "minimum_clearance_m": 0.05,
            "motion_primitives": [
                {
                    "primitive_id": "test-forward",
                    "kind": "FORWARD",
                    "relative_end_pose": {
                        "position_m": [1.0, 0.0, 0.0],
                        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                    "nominal_duration_s": 1.0,
                }
            ],
        }
    if platform == "LEGGED":
        return {
            "reference_point": "test-body",
            "body_half_extent_m": [0.2, 0.2, 0.3],
            "maximum_slope_rad": 0.4,
            "maximum_roughness_m": 0.2,
            "maximum_step_height_m": 0.3,
            "maximum_gap_width_m": 0.4,
            "minimum_confidence": 0.8,
            "minimum_body_clearance_m": 0.1,
            "body_height_m": [0.4, 0.6],
            "forward_speed_mps": [-0.5, 0.5],
            "lateral_speed_mps": [-0.5, 0.5],
            "vertical_speed_mps": [-0.1, 0.1],
            "yaw_rate_radps": [-1.0, 1.0],
            "maximum_linear_acceleration_mps2": 0.5,
            "maximum_yaw_acceleration_radps2": 1.0,
            "motion_primitives": [
                {
                    "primitive_id": "test-forward",
                    "kind": "FORWARD",
                    "body_frame_displacement_m": [1.0, 0.0, 0.0],
                    "yaw_change_rad": 0.0,
                    "nominal_duration_s": 2.0,
                }
            ],
        }
    return {
        "body_half_extent_m": [0.35, 0.25, 0.5],
        "platform_mass_kg": 10.0,
        "gravity_mps2": [0.0, 0.0, -1.62],
        "maximum_landing_slope_rad": 0.4,
        "maximum_landing_roughness_m": 0.1,
        "maximum_plane_residual_m": 0.05,
        "minimum_overhead_clearance_m": 0.0,
        "minimum_lateral_clearance_m": 0.0,
        "minimum_landing_region_area_m2": 0.2,
        "maximum_launch_speed_mps": 8.0,
        "maximum_launch_impulse_newton_seconds": 100.0,
        "minimum_flight_time_s": 0.5,
        "maximum_flight_time_s": 10.0,
        "maximum_landing_speed_mps": 8.0,
        "minimum_downward_impact_speed_mps": 0.1,
        "minimum_landing_clearance_m": 0.0,
        "maximum_angular_speed_radps": 2.0,
        "maximum_angular_acceleration_radps2": 4.0,
        "maximum_initial_angular_speed_radps": 0.2,
        "minimum_settle_guard_s": 0.1,
        "actuator_or_impulse_profile": {"profile_id": "test-impulse"},
        "motion_primitives": [{"primitive_id": "test-hop"}],
    }


def _rewrite_content(
    lock_path: Path,
    platform: str,
    mutation,
) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in lock["platforms"] if item["platform_type"] == platform
    )
    content_path = lock_path.parent / entry["content_path"]
    content = json.loads(content_path.read_text(encoding="utf-8"))
    mutation(content)
    content_bytes = (json.dumps(content, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    content_path.write_bytes(content_bytes)
    entry["content_file_sha256"] = _sha256(content_bytes)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")


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


def test_formal_rejects_arbitrary_nonempty_capability_mapping(
    tmp_path: Path,
) -> None:
    """Would fail if any non-empty JSON object could authorize formal training."""
    lock_path = _write_bundle(tmp_path / "arbitrary")
    _rewrite_content(
        lock_path,
        "WHEELED",
        lambda document: document.__setitem__("content", {"foo": "bar"}),
    )

    with pytest.raises(CapabilityFreezeError, match="content|field|schema"):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


@pytest.mark.parametrize(
    ("platform", "mutation"),
    (
        (
            "WHEELED",
            lambda document: document["content"].pop("platform_id"),
        ),
        (
            "WHEELED",
            lambda document: document["content"].__setitem__(
                "base_frame_id", 7
            ),
        ),
        (
            "WHEELED",
            lambda document: document["content"]["capability"].__setitem__(
                "maximum_forward_speed_mps", 0.0
            ),
        ),
        (
            "LEGGED",
            lambda document: document["content"]["capability"].__setitem__(
                "forward_speed_mps", [0.1, 1.0]
            ),
        ),
        (
            "HOPPER",
            lambda document: document["content"]["capability"].__setitem__(
                "gravity_mps2", [0.0, 0.0, 1.62]
            ),
        ),
    ),
)
def test_typed_content_rejects_missing_wrong_type_range_or_relation(
    tmp_path: Path,
    platform: str,
    mutation,
) -> None:
    """Would fail if loader-incompatible typed values reached v3 workers."""
    lock_path = _write_bundle(tmp_path / platform.lower())
    _rewrite_content(lock_path, platform, mutation)

    with pytest.raises(CapabilityFreezeError):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


@pytest.mark.parametrize("platform", PLATFORMS)
def test_each_platform_requires_closed_unique_motion_primitives(
    tmp_path: Path,
    platform: str,
) -> None:
    """Would fail if source primitive identity were empty or ambiguous."""
    lock_path = _write_bundle(tmp_path / platform.lower())

    def break_primitives(document: dict[str, object]) -> None:
        primitives = document["content"]["capability"]["motion_primitives"]
        if platform == "LEGGED":
            primitives.append(dict(primitives[0]))
        else:
            primitives.clear()

    _rewrite_content(lock_path, platform, break_primitives)

    with pytest.raises(CapabilityFreezeError, match="primitive"):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


@pytest.mark.parametrize(
    "failure", ("missing", "extra", "wrong-kind", "wrong-type")
)
def test_resource_closure_exactly_matches_content_references(
    tmp_path: Path,
    failure: str,
) -> None:
    """Would fail if geometry/document closure contained drift or ambiguity."""
    lock_path = _write_bundle(tmp_path / failure)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    resources = lock["platforms"][0]["resources"]
    if failure == "missing":
        resources.pop()
    elif failure == "wrong-kind":
        resources[-1]["kind"] = "document"
    elif failure == "wrong-type":
        resources[-1]["kind"] = 7
    else:
        relative = Path("wheeled") / "unreferenced.stl"
        data = b"solid extra\nendsolid extra\n"
        (lock_path.parent / relative).write_bytes(data)
        resources.append(
            {
                "kind": "mesh",
                "path": relative.as_posix(),
                "sha256": _sha256(data),
            }
        )
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(CapabilityFreezeError, match="resource|closure"):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


def test_complete_typed_bundle_maps_all_platforms_to_v3_bridge(
    tmp_path: Path,
) -> None:
    """Would fail if the pickle payload could not recreate C++ v3 capabilities."""
    import lunar_planner_training_bridge as bridge_api

    loaded = load_frozen_capability_bundle(
        _write_bundle(tmp_path / "typed"), run_kind="formal"
    )
    bundle = pickle.loads(pickle.dumps(loaded))

    wheel = bundle.for_platform("WHEELED")
    legged = bundle.for_platform("LEGGED")
    hopper = bundle.for_platform("HOPPER")
    assert type(wheel.typed_capability).__name__ == "FrozenWheeledCapability"
    assert type(legged.typed_capability).__name__ == "FrozenLeggedCapability"
    assert type(hopper.typed_capability).__name__ == "FrozenHopperCapability"

    wheel_bridge = wheel.to_bridge_capability()
    legged_bridge = legged.to_bridge_capability()
    hopper_bridge = hopper.to_bridge_capability()
    assert isinstance(wheel_bridge, bridge_api.WheeledCapability)
    assert wheel_bridge.motion_primitives[0].primitive_id == "test-forward"
    assert isinstance(legged_bridge, bridge_api.LeggedCapability)
    assert legged_bridge.motion_primitives[0].primitive_id == "test-forward"
    assert isinstance(hopper_bridge, bridge_api.HopperCapability)
    assert hopper.typed_capability.source_motion_primitive_ids == ("test-hop",)
    assert hopper_bridge.platform_mass_kg == 10.0


def _record_injected_capability(worker_index, platform_type, capability, scenario):
    return worker_index, platform_type, capability, scenario


def test_environment_factory_uses_loaded_bundle_without_source_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(
            AssertionError(f"worker reread capability source: {self}")
        ),
    )

    result = pickle.loads(pickle.dumps(factory))(7, "LEGGED")

    assert result[2] == bundle.for_platform("LEGGED")
    assert result[3].capability_version == result[2].capability_version
    assert result[3].capability_sha256 == result[2].content_sha256
    assert result[3].scenario_schedule_id == "nasa-polar-train/v1"

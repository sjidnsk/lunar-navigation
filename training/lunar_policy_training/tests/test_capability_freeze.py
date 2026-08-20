from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import pytest
import yaml

from lunar_policy_training.capability_freeze import (
    CAPABILITY_TYPES,
    CapabilityFreezeError,
    FrozenCapabilityEnvironmentFactory,
    FrozenObservationCapability,
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
    for platform in PLATFORMS:
        version = f"{platform.lower()}-2026.08"
        platform_id = f"test-only-{platform.lower()}"
        base_frame_id = "base_footprint" if platform == "WHEELED" else "base_link"
        typed_content = _test_typed_content(platform)
        platform_root = Path(platform.lower())
        platform_document_path = platform_root / "platform.yaml"
        observation_document_path = platform_root / "observation.json"
        urdf_path = platform_root / "rover.urdf"
        mesh_path = platform_root / "body.stl"
        platform_document = {
            "schema_version": "platform-control-capability-source/v2",
            "platform": {
                "platform_id": platform_id,
                "platform_type": platform,
                "capability_version": version,
                "base_frame_id": base_frame_id,
            },
            "geometry_source": {"urdf_file": urdf_path.as_posix()},
            platform.lower(): typed_content,
        }
        observation_document = {
            "sensor_range_m": 30.0,
            "sensor_fov_deg": 360.0,
        }
        resource_bytes = {
            ("document", platform_document_path): yaml.safe_dump(
                platform_document, sort_keys=False
            ).encode("utf-8"),
            ("document", observation_document_path): (
                json.dumps(observation_document, separators=(",", ":")) + "\n"
            ).encode("utf-8"),
            ("urdf", urdf_path): (
                f'<robot name="test"><link name="{base_frame_id}"><visual><geometry>'.encode(
                    "utf-8"
                )
                +
                b'<mesh filename="body.stl" scale="1 1 1"/>'
                b'</geometry></visual></link></robot>\n'
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
                    "platform_id": platform_id,
                    "base_frame_id": base_frame_id,
                    "platform_document_path": platform_document_path.as_posix(),
                    "observation_document_path": (
                        observation_document_path.as_posix()
                    ),
                    "urdf_path": urdf_path.as_posix(),
                    "mesh_paths": [mesh_path.as_posix()],
                    "observation": {
                        "sensor_range_m": 30.0,
                        "sensor_fov_deg": 360.0,
                    },
                    "capability": typed_content,
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
            "reference_point": "base_footprint",
            "footprint_xy_m": [
                [-0.591, -0.409],
                [0.591, -0.409],
                [0.591, 0.409],
                [-0.591, 0.409],
            ],
            "body_extent_m": [1.182, 0.818, 1.29996],
            "wheel_diameter_m": 0.319,
            "wheel_width_m": 0.148,
            "wheelbase_m": 0.8175,
            "track_width_m": 0.67,
            "minimum_underbody_clearance_m": 0.21,
            "maximum_local_obstacle_relief_m": 0.2,
            "allow_unsupported_gap": False,
            "maximum_slope_rad": 0.3490658503988659,
            "maximum_forward_speed_mps": 1.5,
            "maximum_reverse_speed_mps": 1.5,
            "maximum_spin_rate_radps": 1.0,
            "maximum_acceleration_mps2": 1.0,
            "maximum_braking_deceleration_mps2": 1.0,
            "maximum_yaw_acceleration_radps2": 1.0,
            "maximum_lateral_acceleration_mps2": 1.0,
            "maximum_curvature_per_m": 1.0,
            "minimum_clearance_m": 0.2,
            "roughness_handling": "COST_SPEED_AND_LOCAL_RECHECK",
            "motion_primitives": [
                {
                    "primitive_id": "test-forward",
                    "kind": "FORWARD",
                    "relative_end_pose": {
                        "position_m": [1.0, 0.0, 0.0],
                        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                }
            ],
        }
    if platform == "LEGGED":
        return {
            "reference_point": "base_link",
            "body_extent_m": [0.68, 0.33, 0.35],
            "nominal_body_height_m": 0.33,
            "platform_mass_kg": 15.89,
            "nominal_payload_kg": 8.0,
            "maximum_payload_kg": 10.0,
            "maximum_slope_rad": 0.5235987755982988,
            "maximum_step_height_m": 0.5,
            "maximum_gap_width_m": 0.3,
            "minimum_body_clearance_m": 0.3,
            "step_vertical_rate_mps": 0.1,
            "body_height_m": [0.28, 0.38],
            "forward_speed_mps": [-1.5, 1.5],
            "lateral_speed_mps": [-0.8, 0.8],
            "yaw_rate_radps": [-1.0, 1.0],
            "maximum_linear_acceleration_mps2": 0.5,
            "maximum_yaw_acceleration_radps2": 1.0,
            "roughness_handling": "DIAGNOSTIC_ONLY",
            "unknown_is_traversable": False,
            "motion_primitives": [
                {
                    "primitive_id": "test-forward",
                    "kind": "FORWARD",
                    "body_frame_displacement_m": [1.0, 0.0, 0.0],
                    "yaw_change_rad": 0.0,
                }
            ],
        }
    return {
        "specific_impulse_s": 301.0,
        "reference_total_mass_kg": 20.0,
        "reference_propellant_mass_kg": 0.2,
        "landing_support_radius_m": 0.45,
        "flight_collision_radius_m": 0.55,
        "maximum_landing_plane_residual_m": 0.05,
        "landing_lateral_margin_m": 0.2,
        "flight_map_margin_m": 0.2,
        "reachability_delta_v_margin_ratio": 0.1,
        "standard_gravity_mps2": 9.80665,
        "maximum_landing_slope_rad": 0.17453292519943295,
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


def _rewrite_locked_resource(
    lock_path: Path,
    platform: str,
    relative_path: str,
    data: bytes,
) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in lock["platforms"] if item["platform_type"] == platform
    )
    resource = next(
        item for item in entry["resources"] if item["path"] == relative_path
    )
    (lock_path.parent / relative_path).write_bytes(data)
    resource["sha256"] = _sha256(data)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")


def _rewrite_observation_source_and_content(
    lock_path: Path,
    platform: str,
    *,
    sensor_range_m: float,
    sensor_fov_deg: float,
) -> None:
    observation = {
        "sensor_range_m": sensor_range_m,
        "sensor_fov_deg": sensor_fov_deg,
    }
    _rewrite_content(
        lock_path,
        platform,
        lambda document: document["content"].__setitem__(
            "observation", observation
        ),
    )
    _rewrite_locked_resource(
        lock_path,
        platform,
        f"{platform.lower()}/observation.json",
        (json.dumps(observation, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )


def _rewrite_typed_source_and_content(
    lock_path: Path,
    platform: str,
    mutation,
) -> None:
    _rewrite_content(
        lock_path,
        platform,
        lambda document: mutation(document["content"]["capability"]),
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in lock["platforms"] if item["platform_type"] == platform
    )
    content_path = lock_path.parent / entry["content_path"]
    content = json.loads(content_path.read_text(encoding="utf-8"))
    source_path = content["content"]["platform_document_path"]
    source = yaml.safe_load((lock_path.parent / source_path).read_text(encoding="utf-8"))
    mutation(source[platform.lower()])
    _rewrite_locked_resource(
        lock_path,
        platform,
        source_path,
        yaml.safe_dump(source, sort_keys=False).encode("utf-8"),
    )


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
    assert all(
        platform.observation_capability
        == FrozenObservationCapability(
            sensor_range_m=30.0,
            sensor_fov_rad=2.0 * math.pi,
        )
        for platform in first.platforms
    )
    assert pickle.loads(pickle.dumps(first)) == first
    assert str(tmp_path) not in repr(first)


def test_formal_bundle_rejects_shared_nonapproved_observation_capability(
    tmp_path: Path,
) -> None:
    """A self-consistent 120-degree bundle must not become formal eligible."""
    lock_path = _write_bundle(tmp_path / "formal-120-degree")
    for platform in PLATFORMS:
        _rewrite_observation_source_and_content(
            lock_path,
            platform,
            sensor_range_m=30.0,
            sensor_fov_deg=120.0,
        )

    with pytest.raises(
        CapabilityFreezeError, match="formal observation capability"
    ):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


def test_formal_bundle_rejects_per_platform_observation_drift(
    tmp_path: Path,
) -> None:
    """All three platforms share one conservative system-level observation."""
    lock_path = _write_bundle(tmp_path / "formal-platform-drift")
    _rewrite_observation_source_and_content(
        lock_path,
        "LEGGED",
        sensor_range_m=29.0,
        sensor_fov_deg=360.0,
    )

    with pytest.raises(
        CapabilityFreezeError, match="formal observation capability"
    ):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


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
                "standard_gravity_mps2", 0.0
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


@pytest.mark.parametrize("platform", ("WHEELED", "LEGGED"))
def test_each_platform_requires_closed_unique_motion_primitives(
    tmp_path: Path,
    platform: str,
) -> None:
    """Would fail if source primitive identity were empty or ambiguous."""
    lock_path = _write_bundle(tmp_path / platform.lower())

    def break_primitives(document: dict[str, object]) -> None:
        primitives = document["content"]["capability"]["motion_primitives"]
        primitives.append(dict(primitives[0]))

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


@pytest.mark.parametrize(
    "mutation",
    (
        lambda source: source["wheeled"].__setitem__(
            "maximum_forward_speed_mps", 2.0
        ),
        lambda source: source["platform"].__setitem__(
            "platform_id", "different-platform"
        ),
        lambda source: source["platform"].__setitem__(
            "capability_version", "different-version"
        ),
        lambda source: source["geometry_source"].__setitem__(
            "urdf_file", "wheeled/different.urdf"
        ),
    ),
)
def test_platform_document_is_authoritative_over_duplicate_typed_payload(
    tmp_path: Path,
    mutation,
) -> None:
    """Would fail if capability.json could disagree with the v3 source document."""
    lock_path = _write_bundle(tmp_path / "platform-drift")
    source_path = "wheeled/platform.yaml"
    source = yaml.safe_load(
        (lock_path.parent / source_path).read_text(encoding="utf-8")
    )
    mutation(source)
    _rewrite_locked_resource(
        lock_path,
        "WHEELED",
        source_path,
        yaml.safe_dump(source, sort_keys=False).encode("utf-8"),
    )

    with pytest.raises(CapabilityFreezeError, match="source|document|mismatch"):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


def test_platform_document_must_be_parseable_v2_yaml(tmp_path: Path) -> None:
    """Would fail if a hashed arbitrary text file counted as capability source."""
    lock_path = _write_bundle(tmp_path / "arbitrary-source")
    _rewrite_locked_resource(
        lock_path,
        "WHEELED",
        "wheeled/platform.yaml",
        b"this is not a capability document\n",
    )

    with pytest.raises(CapabilityFreezeError, match="document|schema"):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


def test_observation_document_is_authoritative(tmp_path: Path) -> None:
    """Would fail if observation range/FOV drift were hidden by capability.json."""
    lock_path = _write_bundle(tmp_path / "observation-drift")
    changed = b'{"sensor_range_m":99.0,"sensor_fov_deg":90.0}\n'
    _rewrite_locked_resource(
        lock_path,
        "WHEELED",
        "wheeled/observation.json",
        changed,
    )

    with pytest.raises(CapabilityFreezeError, match="observation|mismatch"):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


@pytest.mark.parametrize(
    "urdf",
    (
        (
            '<robot><link name="base_link"><visual><geometry>'
            '<mesh filename="body.stl"/></geometry></visual></link></robot>\n'
        ),
        '<robot name="test"><link name="base_link"/></robot>\n',
        (
            '<robot name="test"><link name="other"><visual><geometry>'
            '<mesh filename="body.stl"/></geometry></visual></link></robot>\n'
        ),
        (
            '<robot name="test"><link name="base_link"><visual><geometry>'
            '<mesh filename="body.stl" scale="1 -1 1"/>'
            '</geometry></visual></link></robot>\n'
        ),
        (
            '<robot name="test"><link name="base_link"><collision><geometry>'
            '<mesh filename="unlocked.stl"/></geometry></collision></link></robot>\n'
        ),
    ),
)
def test_urdf_geometry_must_close_over_base_link_and_mesh_resources(
    tmp_path: Path,
    urdf: str,
) -> None:
    """Would fail if formal geometry were unparsed or detached from its mesh set."""
    lock_path = _write_bundle(tmp_path / "urdf")
    _rewrite_locked_resource(
        lock_path,
        "WHEELED",
        "wheeled/rover.urdf",
        urdf.encode("utf-8"),
    )

    with pytest.raises(CapabilityFreezeError, match="URDF|mesh|base"):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


def test_missing_urdf_bridge_validator_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = _write_bundle(tmp_path / "missing-urdf-validator")
    monkeypatch.setitem(sys.modules, "lunar_planner_training_bridge", None)

    with pytest.raises(CapabilityFreezeError, match="URDF validator"):
        load_frozen_capability_bundle(lock_path, run_kind="formal")


@pytest.mark.parametrize(
    ("platform", "retired_field", "retired_value"),
    (
        ("WHEELED", "maximum_obstacle_height_m", 0.2),
        ("LEGGED", "maximum_roughness_m", 0.2),
        ("HOPPER", "maximum_launch_impulse_newton_seconds", 100.0),
    ),
)
def test_capability_v2_rejects_retired_platform_fields(
    tmp_path: Path,
    platform: str,
    retired_field: str,
    retired_value: object,
) -> None:
    """Volume 3 capability-v1 fields must not leak into current training."""
    lock_path = _write_bundle(tmp_path / retired_field)
    _rewrite_typed_source_and_content(
        lock_path,
        platform,
        lambda capability: capability.__setitem__(retired_field, retired_value),
    )

    with pytest.raises(CapabilityFreezeError, match="schema fields"):
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
    assert legged_bridge.nominal_body_height_m == pytest.approx(0.33)
    assert legged_bridge.nominal_payload_kg == pytest.approx(8.0)
    assert legged_bridge.unknown_is_traversable is False
    assert legged_bridge.motion_primitives[0].primitive_id == "test-forward"
    assert isinstance(hopper_bridge, bridge_api.HopperCapability)
    assert hopper.source_motion_primitive_ids == ()
    assert hopper_bridge.specific_impulse_s == 301.0


class _SensorClosedEnvironment:
    sensor_closed_loop = True


class _FormalWorkerRecord:
    def __init__(self, *payload) -> None:
        self.environment = _SensorClosedEnvironment()
        self.payload = payload


def _record_injected_capability(worker_index, platform_type, capability, scenario):
    return _FormalWorkerRecord(
        worker_index, platform_type, capability, scenario
    )


_record_injected_capability.sensor_closed_loop = True


def _record_legacy_environment(worker_index, platform_type, capability, scenario):
    return object()


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

    assert result.payload[2] == bundle.for_platform("LEGGED")
    assert result.payload[3].capability_version == result.payload[2].capability_version
    assert result.payload[3].capability_sha256 == result.payload[2].content_sha256
    assert result.payload[3].scenario_schedule_id == "nasa-polar-train/v1"


def test_formal_environment_factory_rejects_legacy_non_sensor_loop(
    tmp_path: Path,
) -> None:
    bundle = load_frozen_capability_bundle(
        _write_bundle(tmp_path / "formal-sensor-gate"), run_kind="formal"
    )
    with pytest.raises(CapabilityFreezeError, match="sensor-closed"):
        FrozenCapabilityEnvironmentFactory(
            bundle=bundle,
            scenario_schedule_id="nasa-polar-train/v1",
            builder=_record_legacy_environment,
        )

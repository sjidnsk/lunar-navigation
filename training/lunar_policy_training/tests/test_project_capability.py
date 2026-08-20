from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from lunar_policy_training.capability_freeze import CapabilityFreezeError
from lunar_policy_training.project_capability import (
    APPROVED_PROJECT_CAPABILITY_SHA256,
    load_project_formal_capability,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_RELATIVE = Path("ros2_ws/src/lunar_navigation_config/config")


def _copy_project_capability_tree(destination: Path) -> Path:
    source = REPOSITORY_ROOT / CONFIG_RELATIVE
    target = destination / CONFIG_RELATIVE
    target.mkdir(parents=True)
    for filename in (
        "platform_capability_schema_v2.yaml",
        "three_platform_capability_freeze_v1.yaml",
    ):
        shutil.copyfile(source / filename, target / filename)
    return destination


def _rewrite_yaml(root: Path, dotted_path: str, value: object) -> None:
    path = root / CONFIG_RELATIVE / "three_platform_capability_freeze_v1.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    cursor = document
    components = dotted_path.split(".")
    for component in components[:-1]:
        cursor = cursor[component]
    cursor[components[-1]] = value
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def test_project_formal_capability_uses_approved_freeze() -> None:
    bundle = load_project_formal_capability(REPOSITORY_ROOT)

    assert bundle.formal_eligible is True
    assert bundle.bundle_sha256 == APPROVED_PROJECT_CAPABILITY_SHA256
    assert tuple(item.platform_type for item in bundle.platforms) == (
        "WHEELED",
        "LEGGED",
        "HOPPER",
    )
    assert all(
        item.observation_capability.sensor_range_m == 30.0
        for item in bundle.platforms
    )
    assert all(
        item.observation_capability.sensor_fov_rad == pytest.approx(2.0 * 3.141592653589793)
        for item in bundle.platforms
    )

    wheel = bundle.for_platform("WHEELED").typed_capability
    legged = bundle.for_platform("LEGGED").typed_capability
    hopper = bundle.for_platform("HOPPER").typed_capability
    assert tuple(item.primitive_id for item in wheel.motion_primitives) == (
        "forward",
        "reverse",
        "forward-arc-left",
        "forward-arc-right",
        "reverse-arc-left",
        "reverse-arc-right",
        "spin-left",
        "spin-right",
        "stop-switch",
    )
    assert tuple(item.primitive_id for item in legged.motion_primitives) == (
        "forward",
        "backward",
        "lateral-left",
        "lateral-right",
        "spin-left",
        "spin-right",
    )
    assert legged.nominal_body_height_m == pytest.approx(0.33)
    assert legged.nominal_payload_kg == pytest.approx(8.0)
    assert legged.unknown_is_traversable is False
    assert legged.motion_primitives[4].yaw_change_rad == pytest.approx(
        3.141592653589793 / 32.0
    )
    assert hopper.reference_total_mass_kg == 20.0
    assert hopper.reference_propellant_mass_kg == 0.2
    assert (hopper.gravity_mps2.x, hopper.gravity_mps2.y, hopper.gravity_mps2.z) == (
        0.0,
        0.0,
        -1.62,
    )
    assert hopper.reference_horizontal_range_m == 100.0
    assert hopper.reference_elevation_delta_m == 0.0
    assert hopper.runtime_fallback_allowed is False


def test_project_formal_capability_rejects_digest_drift(tmp_path: Path) -> None:
    root = _copy_project_capability_tree(tmp_path / "drift")
    _rewrite_yaml(
        root,
        "platforms.WHEELED.kinematics.maximum_forward_speed_mps",
        9.0,
    )

    with pytest.raises(CapabilityFreezeError, match="freeze_digest_sha256"):
        load_project_formal_capability(root)


def test_project_formal_capability_does_not_require_urdf_or_mesh(
    tmp_path: Path,
) -> None:
    root = _copy_project_capability_tree(tmp_path / "no-assets")

    bundle = load_project_formal_capability(root)

    assert all(item.urdf_path == "" for item in bundle.platforms)
    assert all(item.mesh_paths == () for item in bundle.platforms)
    assert all(item.resources == () for item in bundle.platforms)

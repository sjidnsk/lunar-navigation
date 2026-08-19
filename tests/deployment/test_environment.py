from __future__ import annotations

import pytest
from pathlib import Path

from deployment.luna_runtime.config import load_profile
from deployment.luna_runtime.environment import (
    EnvironmentLockError,
    load_environment_lock,
    verify_environment_lock,
)


class FakeRosdepResolver:
    def __init__(self, resolved: dict[str, str]) -> None:
        self.resolved = resolved

    def resolve(self, _roots: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(self.resolved.values())


def test_orin_environment_lock_keeps_tensorrt_out_of_fallback_requirements() -> None:
    lock = load_environment_lock("jetson-orin-r36")
    assert lock.profile_id == "jetson-orin-r36"
    assert lock.l4t_prefix == "R36"
    assert lock.jetpack_major == 6
    assert "trtexec" not in lock.fallback_required_executables
    assert lock.model_required_executables == ("trtexec",)


def test_environment_lock_exposes_nav2_as_an_optional_group() -> None:
    lock = load_environment_lock("ubuntu22-humble-amd64")
    assert "ros-humble-nav2-core" not in lock.required_apt_packages
    assert lock.optional_apt_groups["nav2_adapter"] == (
        "ros-humble-nav2-core",
        "ros-humble-nav2-costmap-2d",
        "ros-humble-pluginlib",
    )


def test_environment_lock_rejects_a_resolved_dependency_missing_from_the_lock() -> None:
    resolver = FakeRosdepResolver(
        {"rclcpp": "ros-humble-rclcpp", "new_key": "ros-humble-new-key"}
    )
    with pytest.raises(EnvironmentLockError, match="DEPENDENCY_LOCK_DRIFT"):
        verify_environment_lock(
            repo_root=Path(__file__).resolve().parents[2],
            profile=load_profile("ubuntu22-humble-amd64"),
            resolver=resolver,
        )

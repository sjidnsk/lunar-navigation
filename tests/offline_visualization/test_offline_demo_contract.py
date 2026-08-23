#!/usr/bin/env python3
"""Keep the standalone visual demo independent of ROS adapters."""

from pathlib import Path


THIS_DIRECTORY = Path(__file__).resolve().parent
DEMO = THIS_DIRECTORY / "offline_wheel_demo.cpp"
BUILD = THIS_DIRECTORY / "CMakeLists.txt"


def test_demo_uses_only_the_published_pure_core_api() -> None:
    source = DEMO.read_text(encoding="utf-8")
    cmake = BUILD.read_text(encoding="utf-8")

    assert '#include "wheel/wheel_planner.hpp"' in source
    assert "rclcpp" not in source
    assert "trusted_bridge" not in source
    assert "find_package(lunar_pure_planner_core REQUIRED)" in cmake
    assert "LUNAR_CORE_SOURCE_DIR" in cmake
    assert "find_package(rclcpp" not in cmake

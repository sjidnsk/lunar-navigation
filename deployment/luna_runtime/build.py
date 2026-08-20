from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import RuntimeConfig
from .state import RuntimePaths


REQUIRED_PACKAGES = (
    "lunar_navigation_msgs",
    "lunar_planning_msgs",
    "lunar_navigation_config",
    "lunar_model_contract",
    "lunar_planner_core",
    "lunar_planner_ros",
    "lunar_policy_runtime",
)


@dataclass(frozen=True)
class BuildPlan:
    base_paths: tuple[Path, Path]
    packages: tuple[str, ...]
    build_base: Path
    install_base: Path
    log_base: Path


class BuildRunner(Protocol):
    def run(self, command: tuple[str, ...]) -> None: ...


def make_build_plan(config: RuntimeConfig, paths: RuntimePaths, repo_root: Path) -> BuildPlan:
    packages = REQUIRED_PACKAGES
    if config.input_adapters["mode"] == "task3_adapted":
        packages += ("luna_t3_map_adapter",)
    if config.planner["enable_nav2_adapter"]:
        packages += ("lunar_nav2_adapter",)
    return BuildPlan(
        base_paths=(repo_root / "ros2_ws" / "src", repo_root / "model_contract"),
        packages=packages,
        build_base=paths.data / "build",
        install_base=paths.data / "install",
        log_base=paths.data / "log",
    )


def build_runtime(plan: BuildPlan, runner: BuildRunner) -> None:
    command = (
        "colcon",
        "--log-base",
        str(plan.log_base),
        "build",
        "--base-paths",
        *(str(path) for path in plan.base_paths),
        "--packages-select",
        *plan.packages,
        "--build-base",
        str(plan.build_base),
        "--install-base",
        str(plan.install_base),
        "--cmake-args",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
    )
    runner.run(command)

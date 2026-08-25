#!/usr/bin/env python3
"""Create the source-only ROS 2 Humble production bundle for Jetson Orin."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path


CORE_PRODUCTION_SOURCES = (
    "src/planner.cpp",
    "src/global_goal_feasibility.cpp",
    "src/grid_v1/grid_v1_planner.cpp",
    "src/grid_v1/traversability_map.cpp",
    "src/hierarchical/frame_transform.cpp",
    "src/hierarchical/global_route_planner.cpp",
    "src/hierarchical/reference_composer.cpp",
    "src/hierarchical/surface_portal_set.cpp",
    "src/hierarchical/surface_rolling_session.cpp",
    "src/hierarchical/surface_global_search.cpp",
    "src/hopper/anytime_hopper_planner.cpp",
    "src/hopper/ballistic_envelope.cpp",
    "src/hopper/ballistic_kinematics.cpp",
    "src/legged/anytime_legged_planner.cpp",
    "src/shared/active_planner_cache.cpp",
    "src/shared/anytime_ara_star.cpp",
    "src/shared/cell_area_distance_transform.cpp",
    "src/shared/global_occupancy_projection.cpp",
    "src/shared/goal_distance_field.cpp",
    "src/shared/local_terrain_projection.cpp",
    "src/shared/map_snapshot.cpp",
    "src/shared/obstacle_height_estimator.cpp",
    "src/shared/planning_timing.cpp",
    "src/shared/request_local_start_patch.cpp",
    "src/shared/search_control.cpp",
    "src/wheel/anytime_wheel_planner.cpp",
)

ROS_PRODUCTION_SOURCES = (
    "src/platform_config.cpp",
    "src/input_store.cpp",
    "src/traversability_input.cpp",
    "src/map_adapters.cpp",
    "src/center_distance_transform.cpp",
    "src/incremental_traversability.cpp",
    "src/elevation_occupancy.cpp",
    "src/traversability_qos.cpp",
    "src/trusted_bridge.cpp",
    "src/state_adapter.cpp",
    "src/message_conversion.cpp",
    "src/request_diagnostics.cpp",
    "src/pure_plan_motion_server.cpp",
    "src/rviz_goal_bridge.cpp",
    "src/main.cpp",
    "src/rviz_goal_bridge_main.cpp",
    "src/local_traversability_main.cpp",
    "src/local_traversability_node.cpp",
    "src/elevation_occupancy_main.cpp",
    "src/elevation_occupancy_node.cpp",
)

BUILD_SCRIPT = """#!/usr/bin/env bash
# Build only the ROS 2 Humble production packages on Jetson Orin.
set -eo pipefail

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${bundle_root}/ros2_ws"

if [[ ! -r /opt/ros/humble/setup.bash ]]; then
  echo "未找到 ROS 2 Humble：/opt/ros/humble/setup.bash" >&2
  exit 1
fi

source /opt/ros/humble/setup.bash
cd "${workspace}"
colcon build \\
  --packages-up-to lunar_pure_planner_ros lunar_pure_wheeled_controller \\
  --cmake-args \\
    -DBUILD_TESTING=OFF \\
    -DLUNAR_BUILD_DEMO=OFF \\
    -DCMAKE_BUILD_TYPE=Release \\
  --event-handlers console_direct+
"""

README = """# car_orin

Jetson Orin 的 ROS 2 Humble 轮式生产部署源码包。目录不包含测试、探索、
Demo、RViz、Git 历史或任何 x86 构建产物。

## Orin 依赖

- Ubuntu 22.04 / ROS 2 Humble
- `colcon`、CMake 和支持 C++20 的编译器
- package.xml 中列出的 ROS 依赖，特别是 `grid_map_msgs` 和 `yaml-cpp`

可在联网的 Orin 上安装声明依赖：

```bash
cd /path/to/car_orin
source /opt/ros/humble/setup.bash
rosdep install --from-paths ros2_ws/src --ignore-src -r -y
```

## 构建

```bash
cd /path/to/car_orin
bash scripts/build.sh
```

构建产物只在 Orin 本机生成到 `ros2_ws/build`、`ros2_ws/install` 和
`ros2_ws/log`，这些目录不属于传输包。

## 启动

终端 1：

```bash
cd /path/to/car_orin
scripts/start_all.sh wheel 2
```

终端 2：

```bash
cd /path/to/car_orin
scripts/send_goal.sh
```

`environment_mode=2` 使用 `odom` 目标；月表模式使用 `1` 和 `map` 目标。

## 外部输入边界

本目录不生成定位或地图。启动前必须有外部节点持续提供：

- `/Car/T3/mapping/global_overview`
- `/Car/T3/mapping/grid_map`
- `/Car/T3/localization/odometry`
- `/tf`

正式成功必须同时看到 `planning_outcome: 0` 和 `has_reference: true`。
"""

INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("test", "tests", "__pycache__", "*.pyc"),
    )


def resolve_local_include(package: Path, owner: Path, include: str) -> Path | None:
    candidates = (
        owner.parent / include,
        package / "include" / include,
        package / "src" / include,
    )
    package_root = package.resolve()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved == package_root or package_root in resolved.parents:
            return resolved
    return None


def source_closure(package: Path, seeds: tuple[str, ...]) -> set[Path]:
    pending = [(package / relative).resolve() for relative in seeds]
    selected: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in selected:
            continue
        if not current.is_file():
            raise FileNotFoundError(current)
        selected.add(current)
        if current.suffix not in {".cpp", ".h", ".hpp"}:
            continue
        for include in INCLUDE_PATTERN.findall(current.read_text(encoding="utf-8")):
            dependency = resolve_local_include(package, current, include)
            if dependency is not None and dependency not in selected:
                pending.append(dependency)
    return selected


def copy_source_package(
    source_root: Path,
    output_root: Path,
    package_name: str,
    production_sources: tuple[str, ...],
) -> None:
    package = source_root / "ros2_ws" / "src" / package_name
    destination = output_root / "ros2_ws" / "src" / package_name
    for relative in ("CMakeLists.txt", "package.xml"):
        copy_file(package / relative, destination / relative)
    for source in source_closure(package, production_sources):
        copy_file(source, destination / source.relative_to(package))


def write_text(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def write_manifest(output: Path) -> None:
    lines = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "MANIFEST.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output).as_posix()}\n")
    write_text(output / "MANIFEST.sha256", "".join(lines))


def create_bundle(source_root: Path, output: Path) -> None:
    source_root = source_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}")
    if not (source_root / "ros2_ws" / "src").is_dir():
        raise FileNotFoundError(f"不是有效源码根目录：{source_root}")
    output.mkdir(parents=True)

    for name in ("external_interfaces.yaml", "pure_planner.yaml", "wheel.yaml"):
        copy_file(source_root / "config" / name, output / "config" / name)
    for name in (
        "local_traversability.launch.py",
        "pure_planner.launch.py",
        "rviz_goal_bridge.launch.py",
    ):
        copy_file(source_root / "launch" / name, output / "launch" / name)

    messages = source_root / "ros2_ws" / "src" / "lunar_planning_msgs"
    copy_tree(messages, output / "ros2_ws" / "src" / messages.name)
    copy_source_package(
        source_root,
        output,
        "lunar_pure_planner_core",
        CORE_PRODUCTION_SOURCES,
    )
    copy_source_package(
        source_root,
        output,
        "lunar_pure_planner_ros",
        ROS_PRODUCTION_SOURCES,
    )

    controller = source_root / "ros2_ws" / "src" / "lunar_pure_wheeled_controller"
    controller_destination = output / "ros2_ws" / "src" / controller.name
    for relative in ("CMakeLists.txt", "package.xml", "launch", "python", "scripts"):
        source = controller / relative
        destination = controller_destination / relative
        if source.is_dir():
            copy_tree(source, destination)
        else:
            copy_file(source, destination)

    for name in ("start_all.sh", "send_goal.sh"):
        destination = output / "scripts" / name
        copy_file(source_root / "scripts" / name, destination)
        destination.chmod(destination.stat().st_mode | 0o111)
    write_text(output / "scripts" / "build.sh", BUILD_SCRIPT, executable=True)
    write_text(output / "README_DEPLOY.md", README)
    write_manifest(output)


def main() -> None:
    arguments = parse_args()
    create_bundle(arguments.source_root, arguments.output)


if __name__ == "__main__":
    main()

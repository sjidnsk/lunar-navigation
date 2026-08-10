"""把规范 ROS 栅格投影为 fed9 策略使用的 C++ v3 可通行层。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

import numpy as np
import yaml

import lunar_planner_training_bridge as bridge_api
from lunar_policy_training.environment.observation_builder import (
    ObservedWorld,
    PlatformProjection,
)
from lunar_policy_training.project_capability import load_project_formal_capability

from .grid_map_runtime import DecodedGridMap, REQUIRED_LAYERS


# 这些是 interface-v1 模型清单中冻结的完整 YAML 文件身份，不是后续能力版本。
_APPROVED_PROFILE_SHA256: Final = {
    "WHEELED": "3a4f87e310cf7721be71818f5e4abc6cc73e78644bcdb0f8465426e8bcef9360",
    "LEGGED": "1f25b2fc4796e50ef7e966473c03cf902b1073291e2d1ccb09983ec90f0b17f0",
    "HOPPER": "1af41026d4c81500ce3351639d1fa7443f0c161b4de67f5da13d643b44dff841",
}


def _vec3(x: float, y: float, z: float = 0.0):
    value = bridge_api.Vec3()
    value.x = x
    value.y = y
    value.z = z
    return value


def _bridge_grid_map(grid: DecodedGridMap):
    result = bridge_api.GridMap()
    result.frame_id = grid.frame_id
    result.stamp.nanoseconds_since_epoch = grid.stamp_ns
    result.width = grid.width
    result.height = grid.height
    result.resolution_m = grid.resolution_m
    result.origin_m = _vec3(*grid.origin_xy_m)

    # decode_grid_map 已经输出 C++ 规范的南向起始 row-major 顺序。
    dtypes = {
        "valid_mask": np.uint8,
        "obstacle": np.uint8,
        "forbidden": np.uint8,
        "observation_count": np.uint32,
    }
    result.layers = {
        name: bridge_api.GridLayer(
            np.ascontiguousarray(
                grid.layers[name], dtype=dtypes.get(name, np.float32)
            ).reshape(-1)
        )
        for name in REQUIRED_LAYERS
    }
    return result


def _center_crop(values: np.ndarray, cells: int = 32) -> np.ndarray:
    if values.shape[0] < cells or values.shape[1] < cells:
        raise ValueError("local projection is smaller than 32x32")
    row = (values.shape[0] - cells) // 2
    column = (values.shape[1] - cells) // 2
    return np.ascontiguousarray(values[row : row + cells, column : column + cells])


class CppV3Projector:
    """严格选择一个旧平台能力，并复用 planner core 的投影实现。"""

    def __init__(self, repository_root: str | Path, profile_file: str | Path) -> None:
        root = Path(repository_root).resolve(strict=True)
        profile_path = Path(profile_file).resolve(strict=True)
        raw_bytes = profile_path.read_bytes()
        document = yaml.safe_load(raw_bytes)
        if not isinstance(document, dict) or document.get("schema_version") != "lunar-platform-profile/v1":
            raise ValueError("selected profile schema is invalid")
        identity = document.get("platform")
        if not isinstance(identity, dict):
            raise ValueError("selected profile identity is invalid")
        platform_type = identity.get("platform_type")
        if platform_type not in _APPROVED_PROFILE_SHA256:
            raise ValueError("selected profile platform is invalid")
        profile_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if profile_sha256 != _APPROVED_PROFILE_SHA256[platform_type]:
            raise ValueError("selected profile is not frozen for interface-v1")

        bundle = load_project_formal_capability(root)
        capability = bundle.for_platform(platform_type)
        if (
            identity.get("platform_id") != capability.platform_id
            or identity.get("capability_version") != capability.capability_version
        ):
            raise ValueError("selected profile identity differs from frozen capability")
        self.platform_type = platform_type
        self.profile_sha256 = profile_sha256
        self.capability = capability
        self._bridge = bridge_api.PlannerBridge()

    def _project_one(self, grid: DecodedGridMap):
        request = bridge_api.TrainingPlanRequest()
        request.request_id = f"interface-v1/projection/{grid.content_id}"
        request.mission_id = "interface-v1/projection"
        request.mission_revision = 1
        request.platform_id = self.capability.platform_id
        request.capability_version = self.capability.capability_version
        request.global_map_generation = 1
        request.local_map_generation = 1
        request.map_from_odom_generation = 1
        request.state_time.nanoseconds_since_epoch = grid.stamp_ns
        request.capability = self.capability.to_bridge_capability()
        request.world.local_map = _bridge_grid_map(grid)
        request.config.global_map.base_resolution_m = grid.resolution_m
        request.config.wheel.xy_resolution_m = grid.resolution_m
        request.config.wheel.yaw_bin_count = 64
        request.config.legged.xy_resolution_m = grid.resolution_m
        request.config.legged.yaw_bin_count = 64
        return self._bridge.project_traversability(request)

    @staticmethod
    def _north_up(output) -> tuple[np.ndarray, np.ndarray]:
        known = np.ascontiguousarray(np.flipud(output.known).astype(bool))
        feasible = np.ascontiguousarray(
            np.flipud(output.hard_feasible).astype(bool)
        )
        return known, feasible

    def project(
        self,
        world: ObservedWorld,
        platform_type: str,
        *,
        global_map: DecodedGridMap,
        local_map: DecodedGridMap,
    ) -> PlatformProjection:
        if platform_type != self.platform_type:
            raise ValueError("platform differs from selected profile")
        global_output = self._project_one(global_map)
        local_output = self._project_one(local_map)
        known, feasible = self._north_up(global_output)
        local_known, local_feasible = self._north_up(local_output)
        traversable = (known & feasible).astype(np.float32)
        local_traversable = _center_crop(
            (local_known & local_feasible).astype(np.float32)
        )

        sensor_range = self.capability.observation_capability.sensor_range_m
        clearance_m = np.ascontiguousarray(
            np.flipud(global_output.clearance_m).astype(np.float32)
        )
        clearance_norm = np.where(
            known,
            np.clip(
                np.nan_to_num(
                    clearance_m,
                    nan=0.0,
                    posinf=sensor_range,
                    neginf=0.0,
                ),
                0.0,
                sensor_range,
            )
            / sensor_range,
            0.0,
        ).astype(np.float32)
        return PlatformProjection(
            canvas=world.canvas,
            traversable_ratio=traversable,
            local_traversable_ratio=local_traversable,
            clearance_margin_norm=clearance_norm,
            source=f"cpp_v3/{self.capability.content_sha256}",
        )


__all__ = ["CppV3Projector"]

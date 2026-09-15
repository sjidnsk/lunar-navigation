"""Training-side view of the canonical native platform capability file."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from types import MappingProxyType
from typing import Mapping

from .contracts import Pose


@dataclass(frozen=True)
class GraphConfig:
    """Graph geometry constants are independent of sensor range."""
    position_unit_m: float = 10.0
    frontier_unit_cells: float = 100.0
    history_tolerance_m: float = 0.1
    platform: "PlatformConfig | None" = None

    def __post_init__(self):
        if self.position_unit_m != 10.0 or self.frontier_unit_cells != 100.0:
            raise ValueError("task_graph_v1 fixes position and frontier normalization")
        if not np.isfinite(self.history_tolerance_m) or self.history_tolerance_m <= 0:
            raise ValueError("positive history tolerance required")


@dataclass(frozen=True)
class PlatformConfig:
    maximum_forward_speed_mps: float
    maximum_reverse_speed_mps: float
    maximum_spin_rate_radps: float
    footprint_radius_m: float
    capability: Mapping = field(default_factory=dict)
    capability_path: str = ""

    def actor_context(
        self,
        pose: Pose,
        *,
        linear_speed_mps: float,
        angular_speed_radps: float,
        sensor_range_m: float,
        sensor_fov_rad: float
    ) -> np.ndarray:
        return np.asarray(
            [
                np.cos(pose.yaw),
                np.sin(pose.yaw),
                linear_speed_mps / 0.2,
                angular_speed_radps / self.maximum_spin_rate_radps,
                sensor_range_m / 10.0,
                sensor_fov_rad / np.pi,
                self.footprint_radius_m,
                self.maximum_forward_speed_mps / 0.2,
            ],
            dtype=np.float32,
        )


def load_platform_config(path: Path | None = None) -> PlatformConfig:
    """Read wheel capability values from the native source of truth."""
    if path is None:
        try:
            from ament_index_python.packages import (
                get_package_share_directory,
                PackageNotFoundError,
            )
        except ImportError:
            path = Path(__file__).resolve().parents[4] / "config" / "wheel.yaml"
        else:
            try:
                path = (
                    Path(
                        get_package_share_directory("lunar_incremental_navigation_ros")
                    )
                    / "config"
                    / "wheel.yaml"
                )
            except PackageNotFoundError:
                path = Path(__file__).resolve().parents[4] / "config" / "wheel.yaml"
    with Path(path).open(encoding="utf-8") as stream:
        capability = yaml.safe_load(stream)["capability"]
    footprint = np.asarray(capability["footprint_xy_m"], dtype=np.float64)
    radius = float(np.max(np.linalg.norm(footprint, axis=1)))
    return PlatformConfig(
        maximum_forward_speed_mps=float(capability["maximum_forward_speed_mps"]),
        maximum_reverse_speed_mps=float(capability["maximum_reverse_speed_mps"]),
        maximum_spin_rate_radps=float(capability["maximum_spin_rate_radps"]),
        footprint_radius_m=radius,
        capability=MappingProxyType(capability),
        capability_path=str(Path(path).resolve()),
    )


@dataclass(frozen=True)
class ModelConfig:
    """Primitive network settings; importing observed workers never imports Torch."""
    width: int = 128
    heads: int = 8
    layers: int = 6

    def __post_init__(self):
        if self.width <= 0 or self.heads <= 0 or self.width % self.heads or self.layers <= 0:
            raise ValueError("positive width divisible by heads and positive layers required")


@dataclass(frozen=True)
class LearningConfig:
    batch_size: int = 64
    microbatch_size: int = 16
    learning_rate: float = 1e-5
    gamma: float = 1.0
    polyak: float = 0.005
    initial_alpha: float = 5e-5
    maximum_alpha: float = 1e-4
    target_entropy_factor: float = 0.01

    def __post_init__(self):
        if self.batch_size != 64 or not 0 < self.microbatch_size <= self.batch_size:
            raise ValueError("effective batch is 64; microbatch must be in [1,64]")
        if self.gamma != 1.0:
            raise ValueError("task reward contract requires gamma=1")
        if not (0 < self.initial_alpha <= self.maximum_alpha and
                0 < self.polyak <= 1 and self.learning_rate > 0 and
                self.target_entropy_factor >= 0):
            raise ValueError("invalid SAC optimization settings")

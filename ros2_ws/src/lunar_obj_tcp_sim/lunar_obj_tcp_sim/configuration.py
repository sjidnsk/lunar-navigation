"""Validated configuration shared by launch and operator entry points."""

from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class SimulationConfig:
    host: str
    port: int
    map_directory: str
    mode: str
    prefix: str
    sensor_range_m: float
    sensor_fov_deg: float
    near_field_radius_m: float
    sensor_offset_xyz_m: tuple[float, float, float]
    observation_window_m: float
    observation_rate_hz: float
    coarse_resolution_m: float
    local_window_size_m: float
    bridge_send_rate_hz: float
    command_timeout_s: float
    feedback_timeout_s: float
    connect_timeout_s: float
    max_wheel_speed_radps: float
    max_linear_mps: float
    max_angular_radps: float
    auto_start: bool
    initial_scan: bool
    start_rviz: bool
    start_local_rviz: bool
    wheel_order: tuple[int, ...] = (0,1,2,3)
    steering_order: tuple[int, ...] = (0,1,2,3)
    wheel_signs: tuple[float, ...] = (1.,1.,1.,1.)
    steering_signs: tuple[float, ...] = (-1.,-1.,-1.,-1.)
    steering_zero_rad: tuple[float, ...] = (0.,0.,0.,0.)
    max_steering_angle_rad: float = math.pi/2
    steering_tolerance_rad: float = .1
    goal_position_tolerance_m: float = .1
    goal_yaw_tolerance_rad: float = .05
    max_angular_accel_radps2: float = .1
    scan_max_angular_radps: float = .15
    scan_max_angular_accel_radps2: float = .1
    scan_step_deg: float = 30.
    scan_reanchor_distance_m: float = .1
    bootstrap_timeout_s: float = 45.
    known_chunk_cells: int = 512
    vehicle_id: int = 0
    exploration_size_m: float = 300.0
    coverage_target: float = 1.0
    navigation_map_wait_timeout_s: float = 30.0


def _finite_positive(values: Mapping[str, float]) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")


def load_simulation_config(
    path: str | Path, overrides: Mapping[str, Any] | None = None
) -> SimulationConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("simulation"), dict):
        raise ValueError("configuration requires a simulation mapping")
    values = dict(document["simulation"])
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    allowed = {field.name for field in fields(SimulationConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown simulation parameters: {', '.join(unknown)}")
    try:
        values["port"] = int(values["port"])
        for name in (
            "sensor_range_m", "sensor_fov_deg", "near_field_radius_m",
            "observation_window_m", "observation_rate_hz", "coarse_resolution_m",
            "local_window_size_m", "bridge_send_rate_hz", "command_timeout_s",
            "feedback_timeout_s", "connect_timeout_s", "max_wheel_speed_radps",
            "goal_position_tolerance_m", "goal_yaw_tolerance_rad", "max_linear_mps", "max_angular_radps", "max_angular_accel_radps2", "coverage_target",
            "scan_max_angular_radps", "scan_max_angular_accel_radps2",
            "scan_step_deg", "scan_reanchor_distance_m", "bootstrap_timeout_s",
            "navigation_map_wait_timeout_s", "exploration_size_m",
        ):
            if name in values:
                values[name] = float(values[name])
        offset = tuple(float(value) for value in values["sensor_offset_xyz_m"])
        values["sensor_offset_xyz_m"] = offset
        for name in ('wheel_signs','steering_signs','steering_zero_rad'):
            if name in values:
                values[name] = tuple(float(value) for value in values[name])
        for name in ('wheel_order','steering_order'):
            if name in values:
                values[name] = tuple(values[name])
        for name in ('max_steering_angle_rad','steering_tolerance_rad'):
            if name in values:
                values[name] = float(values[name])
        config = SimulationConfig(**values)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid simulation configuration: {error}") from error
    if not isinstance(config.vehicle_id, int) or not 0 <= config.vehicle_id <= 0xffffffff:
        raise ValueError("vehicle_id must be a uint32")
    if not isinstance(config.known_chunk_cells, int) or config.known_chunk_cells < 1:
        raise ValueError('known_chunk_cells must be a positive integer')
    if config.mode not in {"explore", "nav"}:
        raise ValueError("mode must be explore or nav")
    if not config.host.strip():
        raise ValueError("host must not be empty")
    if not 1 <= config.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if not config.prefix.startswith("/") or config.prefix.endswith("/"):
        raise ValueError("prefix must be an absolute topic prefix without a trailing slash")
    if len(config.sensor_offset_xyz_m) != 3 or not all(
        math.isfinite(value) for value in config.sensor_offset_xyz_m
    ):
        raise ValueError("sensor_offset_xyz_m must contain three finite values")
    _finite_positive({
        "exploration_size_m": config.exploration_size_m,
        "sensor_range_m": config.sensor_range_m,
        "observation_window_m": config.observation_window_m,
        "observation_rate_hz": config.observation_rate_hz,
        "coarse_resolution_m": config.coarse_resolution_m,
        "local_window_size_m": config.local_window_size_m,
        "bridge_send_rate_hz": config.bridge_send_rate_hz,
        "command_timeout_s": config.command_timeout_s,
        "feedback_timeout_s": config.feedback_timeout_s,
        "connect_timeout_s": config.connect_timeout_s,
        "max_wheel_speed_radps": config.max_wheel_speed_radps,
        "goal_position_tolerance_m": config.goal_position_tolerance_m,
        "goal_yaw_tolerance_rad": config.goal_yaw_tolerance_rad,
        "max_linear_mps": config.max_linear_mps,
        "max_angular_radps": config.max_angular_radps,
        "max_angular_accel_radps2": config.max_angular_accel_radps2,
    })
    if not math.isfinite(config.near_field_radius_m) or config.near_field_radius_m < 0.0:
        raise ValueError("near_field_radius_m must be finite and nonnegative")
    if not 0.0 < config.sensor_fov_deg <= 360.0:
        raise ValueError("sensor_fov_deg must be in (0, 360]")
    if config.observation_window_m < 2.0 * config.sensor_range_m:
        raise ValueError("observation_window_m must contain the sensor_range_m diameter")
    if not 0.0 < config.coverage_target <= 1.0:
        raise ValueError("coverage_target must be in (0, 1]")
    if config.navigation_map_wait_timeout_s < 0.0:
        raise ValueError("navigation_map_wait_timeout_s must be nonnegative")
    from .kinematics import validate_steering_options, steering_options
    validate_steering_options(**steering_options(config))
    _finite_positive({name:getattr(config,name) for name in (
        'scan_max_angular_radps','scan_max_angular_accel_radps2','scan_reanchor_distance_m','bootstrap_timeout_s')})
    if not 5<=config.scan_step_deg<=90:
        raise ValueError('scan_step_deg must be between 5 and 90 degrees')
    return config

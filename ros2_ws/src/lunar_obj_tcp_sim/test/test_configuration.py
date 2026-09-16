from pathlib import Path

import pytest

from lunar_obj_tcp_sim.configuration import load_simulation_config


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "simulation.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_arrival_tolerances_accept_overrides_and_reject_invalid_values():
    path = Path(__file__).parents[1] / 'config' / 'simulation.yaml'
    config = load_simulation_config(path, {
        'goal_position_tolerance_m': 0.1, 'goal_yaw_tolerance_rad': 0.05})
    assert config.goal_position_tolerance_m == 0.1
    assert config.goal_yaw_tolerance_rad == 0.05
    for name in ('goal_position_tolerance_m', 'goal_yaw_tolerance_rad'):
        for value in (0., -1., float('nan')):
            with pytest.raises(ValueError, match=name):
                load_simulation_config(path, {name: value})


def test_load_simulation_config_accepts_runtime_overrides(tmp_path):
    path = write_config(
        tmp_path,
        """
simulation:
  host: 127.0.0.1
  port: 6668
  map_directory: /maps/default
  mode: explore
  prefix: /lunar_sim
  sensor_range_m: 12.0
  sensor_fov_deg: 120.0
  near_field_radius_m: 1.4
  sensor_offset_xyz_m: [0.0, 0.0, 1.5]
  observation_window_m: 28.0
  observation_rate_hz: 5.0
  coarse_resolution_m: 1.0
  local_window_size_m: 64.0
  bridge_send_rate_hz: 20.0
  command_timeout_s: 0.5
  feedback_timeout_s: 0.5
  connect_timeout_s: 5.0
  max_wheel_speed_radps: 10.0
  max_linear_mps: 0.2
  max_angular_radps: 0.5
  auto_start: true
  initial_scan: true
  start_rviz: true
  start_local_rviz: false
""",
    )

    config = load_simulation_config(
        path, overrides={"host": "10.4.0.8", "port": 7001, "mode": "nav"}
    )

    assert config.host == "10.4.0.8"
    assert config.port == 7001
    assert config.mode == "nav"
    assert config.prefix == "/lunar_sim"
    assert config.sensor_offset_xyz_m == (0.0, 0.0, 1.5)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("mode: invalid", "mode must be explore or nav"),
        ("port: 0", "port must be in"),
        ("sensor_fov_deg: 361.0", "sensor_fov_deg"),
        ("observation_window_m: 10.0", "observation_window_m"),
    ],
)
def test_load_simulation_config_rejects_invalid_relationships(tmp_path, replacement, message):
    body = """
simulation:
  host: 127.0.0.1
  port: 6668
  map_directory: /maps/default
  mode: explore
  prefix: /lunar_sim
  sensor_range_m: 12.0
  sensor_fov_deg: 120.0
  near_field_radius_m: 1.4
  sensor_offset_xyz_m: [0.0, 0.0, 1.5]
  observation_window_m: 28.0
  observation_rate_hz: 5.0
  coarse_resolution_m: 1.0
  local_window_size_m: 64.0
  bridge_send_rate_hz: 20.0
  command_timeout_s: 0.5
  feedback_timeout_s: 0.5
  connect_timeout_s: 5.0
  max_wheel_speed_radps: 10.0
  max_linear_mps: 0.2
  max_angular_radps: 0.5
  auto_start: true
  initial_scan: true
  start_rviz: true
  start_local_rviz: false
"""
    key = replacement.split(":", 1)[0]
    lines = ["  " + replacement if line.strip().startswith(key + ":") else line for line in body.splitlines()]
    path = write_config(tmp_path, "\n".join(lines))

    with pytest.raises(ValueError, match=message):
        load_simulation_config(path)

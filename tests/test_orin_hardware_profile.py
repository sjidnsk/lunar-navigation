"""Hardware profile must not inherit simulation endpoints or change geometry."""
from pathlib import Path
import yaml
import importlib.util

ROOT = Path(__file__).resolve().parents[1]

def test_hardware_platform_path_resolves_independently_of_working_directory():
    spec = importlib.util.spec_from_file_location('stack_launch', ROOT / 'launch/exploration_navigation.launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module._load_config(str(ROOT / 'config/orin_hardware.yaml'))
    assert Path(config['common']['platform_config']) == ROOT / 'config/wheel_orin.yaml'

def test_hardware_profile_keeps_physics_and_sets_navigation_margin():
    base = yaml.safe_load((ROOT / 'config/wheel.yaml').read_text())
    hardware = yaml.safe_load((ROOT / 'config/wheel_orin.yaml').read_text())
    assert hardware.pop('start_blind_zone_margin_m') == 2.0
    for name in ('maximum_forward_speed_mps', 'maximum_reverse_speed_mps'):
        assert hardware['capability'][name] == 0.2
        hardware['capability'][name] = base['capability'][name]
    assert hardware == base

def test_hardware_navigation_uses_real_input_and_keeps_execution_feedback():
    c = yaml.safe_load((ROOT / 'config/orin_hardware.yaml').read_text())
    assert c['stack']['mode'] == 'incremental_v2'
    assert c['common']['odometry_topic'] == '/Car/T3/localization/odometry'
    assert c['common']['local_map_topic'] == '/Car/T3/mapping/grid_map'
    assert c['common']['tf_topic'] == '/P4/input/map_to_odom'
    assert c['navigation']['enable_tracking_feedback'] is True
    assert c['navigation']['goal_position_tolerance_m'] == 0.1
    assert c['navigation']['goal_yaw_tolerance_rad'] == 0.05

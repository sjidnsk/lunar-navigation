"""Operator view follows runtime configuration; no simulator is launched."""
import importlib.util
from pathlib import Path
import yaml
import pytest

pytest.importorskip('launch_ros')
from launch import LaunchContext
from launch.utilities import perform_substitutions
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('navigation_rviz', ROOT / 'launch/navigation_rviz.launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('start_rviz', ['true', 'false'])
def test_custom_operator_config(tmp_path, monkeypatch, start_rviz):
    module = load()
    monkeypatch.setattr(module, 'get_package_share_directory', lambda _: str(ROOT))
    config = yaml.safe_load((ROOT / 'config/exploration_navigation.yaml').read_text())
    config['common']['frames']['map'] = 'world'
    config['common']['navigation_action'] = '/custom/navigation'
    config['common']['exploration_map_topic'] = '/custom/map'
    config['common']['exploration_map_qos'] = dict(reliability='best_effort', durability='volatile', depth=4)
    config['navigation']['local_path_topic'] = '/custom/path'
    path = tmp_path / 'runtime.yaml'
    path.write_text(yaml.safe_dump(config))
    context = LaunchContext()
    context.launch_configurations.update(config_file=str(path), use_sim_time='true',
                                         goal_topic='/custom/goal', start_rviz=start_rviz)
    actions = module.compose(context)
    nodes = [a for a in actions if isinstance(a, Node)]
    assert len(nodes) == (2 if start_rviz == 'true' else 1)
    from launch_ros.utilities import evaluate_parameters
    parameters = evaluate_parameters(context, nodes[0]._Node__parameters)[0]
    assert parameters['expected_frame'] == 'world'
    assert parameters['action_name'] == '/custom/navigation'
    assert parameters['goal_topic'] == '/custom/goal'
    assert parameters['use_sim_time'] is True
    if start_rviz == 'false':
        return
    cmd = [perform_substitutions(context, part) for part in nodes[-1].cmd]
    view = yaml.safe_load(Path(cmd[cmd.index('-d') + 1]).read_text())['Visualization Manager']
    assert view['Global Options']['Fixed Frame'] == 'world'
    displays = {d['Name']: d for d in view['Displays']}
    assert displays['Exploration map']['Topic']['Value'] == '/custom/map'
    assert displays['Exploration map']['Topic']['Reliability Policy'] == 'Best Effort'
    assert displays['Exploration map']['Topic']['Durability Policy'] == 'Volatile'
    assert displays['Exploration map']['Topic']['Depth'] == 4
    assert displays['Local path']['Topic']['Value'] == '/custom/path'
    assert view['Tools'][-1]['Topic'] == '/custom/goal'

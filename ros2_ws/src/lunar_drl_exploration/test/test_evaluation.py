"""Real frozen Actor, reference intersection and saved evaluation report contract."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from lunar_drl_exploration.config import TrainingConfig
from lunar_drl_exploration.contracts import PrivilegedState, RewardParts, SensorSpec, Transition
from lunar_drl_exploration.reference import CoverageReference
from lunar_drl_exploration.evaluation import evaluate
from lunar_drl_exploration.model import Actor
from worker_fixtures import observation


def test_frozen_evaluation_json_keeps_reference_areas_and_failure_measurements(tmp_path, monkeypatch):
    from lunar_drl_exploration import ros_env
    sensor = SensorSpec(range_m=7., fov_deg=75., offset_x_m=.2,
                        offset_y_m=-.1, offset_yaw_rad=np.pi / 3)
    config = replace(TrainingConfig(), sensor=sensor)
    actor = Actor(config.model)
    weights = {key: value.clone() for key, value in actor.state_dict().items()}
    artifact = tmp_path / 'actor.pt'
    torch.save(dict(schema='task_graph_v3', model_config=asdict(config.model),
                    version=17, state_dict=weights), artifact)
    actions = []

    class ExecutionTransport:
        def __init__(self, *args, **kwargs): pass
        def reset(self, seed, family, extent, *, episode_budget):
            self.seed, self.steps = seed, 0
            if seed == 4:
                # Deliberately leave the previous reference attached: failure
                # must not leak the preceding episode's absolute areas.
                raise RuntimeError('reset unavailable')
            count = {0: 2, 1: 4, 2: 4, 3: 0, 5: 2}[seed]
            self.reference = CoverageReference((1, 8), np.array([(1 << count)-1], np.uint8),
                1., float(count), 'reference', np.array([255], np.uint8))
            # The high bit is observed outside the reference in every case.
            bits = {0: 129, 1: 131, 2: 129, 3: 128, 5: 129}[seed]
            self.state = PrivilegedState('scene', np.array([bits], np.uint8))
            self.report = SimpleNamespace(completed=seed == 3,exhausted=seed == 3)
            return observation(), self.state
        def progress(self): return dict(distance_m=float(self.steps))
        def step(self, action):
            assert 0 <= action < len(observation().action_nodes)
            actions.append((self.seed, action))
            if self.seed == 5 or (self.seed == 2 and self.steps == 1):
                raise RuntimeError('execution unavailable')
            self.steps += 1
            next_state = PrivilegedState('scene', np.array([131 if self.seed == 2 else self.state.observed[0]], np.uint8))
            self.last_execution = SimpleNamespace(reason_code='GOAL_REACHED')
            return Transition(observation(), action, 0., observation(), self.state,
                next_state, RewardParts(1. if self.seed == 2 else 0., 0., 0.),
                False, self.seed != 2, 'episode', 17)
        def close(self): pass

    monkeypatch.setattr(ros_env, 'RosExplorationEnv', ExecutionTransport)
    # Evaluation may perform inference only; a gradient/update path is a bug.
    def forbidden_backward(*args, **kwargs): raise AssertionError('evaluation attempted learning')
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden_backward)
    output = tmp_path / 'evaluation.json'
    result = evaluate(config, artifact, seeds=range(6), families=['moon'],
                      extents=[40], budget=4, output=output)
    saved = json.loads(output.read_text())
    assert saved == result
    rows = saved['cases']
    assert [row['final_coverage'] for row in rows] == [.5, .5, .5, 0., 0., .5]
    assert [row['covered_area_m2'] for row in rows] == [1., 2., 2., 0., None, 1.]
    assert [row['coverable_area_m2'] for row in rows] == [2., 4., 4., 0., None, 2.]
    assert rows[2]['decisions'] == 1 and rows[2]['distance_m'] == 1.
    assert rows[5]['decisions'] == 0 and rows[5]['distance_m'] == 0.
    assert rows[3]['exhausted'] and not rows[3]['reached_99']
    assert [row['error'] is not None for row in rows] == [False, False, True, False, True, True]
    assert [seed for seed, _ in actions] == [0, 1, 2, 2, 5]
    assert rows[0]['max_zero_gain_run'] == 1
    assert rows[0]['stationary_decisions'] == 1
    assert rows[0]['zero_gain_two_point_loop_steps'] == 0
    assert rows[4]['geometric_metrics_unavailable']
    assert saved['sensor'] == asdict(sensor)
    restored = torch.load(artifact, weights_only=False)
    for key, value in restored['state_dict'].items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)

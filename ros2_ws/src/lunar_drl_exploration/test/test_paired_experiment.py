from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from lunar_drl_exploration.config import TrainingConfig, config_record, resume_semantics, training_config_from_record
from lunar_drl_exploration.training import Curriculum


def test_paired_curriculum_interleaving_resume_and_namespace():
    config = replace(TrainingConfig(), paired_scene_sequence=True)
    first = Curriculum(config, np.random.default_rng(1))
    other = Curriculum(config, np.random.default_rng(999))
    expected = {}
    for env in [0, 0, 1, 2, 1, 0]:
        expected.setdefault(env, []).append(first.next(env, 0))
    actual = {}
    for env in [2, 1, 0, 1, 0, 0]:
        actual.setdefault(env, []).append(other.next(env, 999999))
    assert actual == expected
    assert all(row['seed'] >= 2**32 and 40 <= row['extent'] <= 80
               and row['episode_budget'] == 512 for rows in actual.values() for row in rows)
    restored = Curriculum(config, np.random.default_rng(7), copy.deepcopy(first.state))
    assert restored.next(1, 10000) == first.next(1, 10000)
    assert len({row['seed'] for rows in actual.values() for row in rows}) == 6


def test_legacy_curriculum_unchanged_and_paired_semantics():
    config = TrainingConfig()
    curriculum = Curriculum(config, np.random.default_rng(17))
    rng = np.random.default_rng(17)
    size = int(rng.choice(3, p=config.curriculum_mixtures[2]))
    extent = float(rng.uniform(*config.curriculum_extents_m[size]))
    assert curriculum.next(1, 999999) == dict(seed=(config.seed+1)<<32, family='cave',
        extent=extent, episode_budget=config.curriculum_budgets[size])
    paired = replace(config, paired_scene_sequence=True)
    assert resume_semantics(paired) != resume_semantics(config)
    assert config_record(training_config_from_record(config_record(paired))) == config_record(paired)


def test_stages_screen_both_before_final_and_skip_completed():
    from lunar_drl_exploration.paired_experiment import stages, pending_stages
    plan = stages([20260919, 20260920, 20260921], 3000, 10000)
    assert [(s['seed'], s['arm'], s['target']) for s in plan[:4]] == [
        (20260919, 'raw', 3000), (20260919, 'bounded10', 3000),
        (20260919, 'raw', 10000), (20260919, 'bounded10', 10000)]
    assert len(plan) == 8
    done = {plan[0]['id']: dict(status='complete')}
    assert pending_stages(plan, done) == plan[1:]
    done[plan[1]['id']] = dict(status='failed')
    assert pending_stages(plan, done)[0] == plan[1]


def test_zero_exit_final_error_is_failure():
    from lunar_drl_exploration.paired_experiment import validate_child_result
    with pytest.raises(RuntimeError, match='broken'):
        validate_child_result(0, dict(status='failed', error='broken'), 3000)
    with pytest.raises(RuntimeError, match='budget'):
        validate_child_result(0, dict(status='complete', transitions=12), 3000)
    with pytest.raises(RuntimeError, match='failed'):
        validate_child_result(1, dict(status='complete', transitions=3001), 3000)
    assert validate_child_result(0, dict(status='complete', transitions=3001, updates=12), 3000)['updates'] == 12


def test_geometric_metrics_use_coordinates_and_zero_gain_runs():
    from lunar_drl_exploration.paired_experiment import GeometryMetrics
    metric = GeometryMetrics((0, 0))
    for xy, gain in [((1, 0), 0), ((0, 0), 0), ((1, 0), 1), ((1, 0), 0)]:
        metric.observe(xy, gain)
    record = metric.record()
    assert record['reverse_edge_count'] == 2
    assert record['max_zero_gain_run'] == 2
    assert record['stationary_decisions'] == 1
    assert record['revisit_ratio'] == .75


def test_sustained_two_point_loops_require_zero_gain_and_four_moves():
    from lunar_drl_exploration.paired_experiment import GeometryMetrics
    metric = GeometryMetrics((0, 0))
    for xy in [(1, 0), (0, 0), (1, 0)]: metric.observe(xy, 0)
    assert metric.record()['zero_gain_two_point_loop_steps'] == 0
    metric.observe((0, 0), 0)
    assert metric.record()['zero_gain_two_point_loop_steps'] == 1
    metric.observe((1, 0), 1)
    metric.observe((0, 0), 0)
    assert metric.record()['zero_gain_two_point_loop_steps'] == 1


@pytest.mark.parametrize('mode', ['success', 'zero_exit_error', 'interrupt'])
def test_runner_subprocess_resume_errors_and_interrupt(tmp_path, monkeypatch, mode):
    import io
    import signal
    from lunar_drl_exploration import paired_experiment as paired
    calls = []

    class Child:
        stdout = None
        def __init__(self, command, **kwargs):
            assert command[2] == 'lunar_drl_exploration.paired_experiment'
            self.stdout = io.BytesIO(b'bounded child log\n')
            self.stage = json.loads(command[command.index('--child-stage') + 1])
            calls.append(self.stage)
            result = dict(stage=self.stage, status='complete', transitions=self.stage['target'] + 3,
                updates=42, initial_model_checksum='same', overshoot=3)
            if mode == 'zero_exit_error': result.update(status='complete', error='fatal worker')
            paired.atomic_json(tmp_path / str(self.stage['seed']) / self.stage['arm'] / 'child-result.json', result)
        def wait(self):
            if mode == 'interrupt': signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return 0
        def poll(self): return 0
        def send_signal(self, sig): pass

    monkeypatch.setattr(paired.subprocess, 'Popen', Child)
    args = paired.parser().parse_args(['--output-dir', str(tmp_path), '--seeds', '42'])
    if mode == 'zero_exit_error':
        with pytest.raises(RuntimeError, match='fatal worker'): paired.run_experiment(args)
        assert len(calls) == 1
    elif mode == 'interrupt':
        assert paired.run_experiment(args) == 130
        assert len(calls) == 1
    else:
        assert paired.run_experiment(args) == 0
        assert len(calls) == 4
        args.resume = True
        assert paired.run_experiment(args) == 0
        assert len(calls) == 4, 'completed arms must not rerun'
    assert (tmp_path / 'paired-report.md').exists()


def test_second_runner_cannot_own_live_output(tmp_path):
    import fcntl
    from lunar_drl_exploration import paired_experiment as paired
    with (tmp_path / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args = paired.parser().parse_args(['--output-dir', str(tmp_path)])
        with pytest.raises(RuntimeError, match='another paired runner'):
            paired.run_experiment(args)


def test_initial_parameter_checksum_bound_independent_seed_sensitive():
    torch = pytest.importorskip('torch')
    from lunar_drl_exploration.config import ModelConfig
    from lunar_drl_exploration.sac import SACLearner
    from lunar_drl_exploration.training import initial_model_checksum
    checksums = []
    for seed, bound in [(17, 0.), (17, 10.), (18, 10.)]:
        torch.manual_seed(seed)
        learner = SACLearner(ModelConfig(width=16, heads=4, layers=1, actor_score_bound=bound))
        checksums.append(initial_model_checksum(learner))
    assert checksums[0] == checksums[1] != checksums[2]

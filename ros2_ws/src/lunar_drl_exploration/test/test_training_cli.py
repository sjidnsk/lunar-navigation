import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import pytest


def test_cli_help_is_stdlib_only_and_all_commands_exist():
    code = '''import sys, importlib.abc
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','numpy','rclpy'}: raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from lunar_drl_exploration.cli import main
main(['--help'])
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get('PYTHONPATH', '')))
    assert result.returncode == 0, result.stderr
    assert all(command in result.stdout for command in ('train', 'evaluate', 'infer', 'export'))


def test_default_config_and_explicit_probe_preserve_network_and_batch():
    assert importlib.util.find_spec('lunar_drl_exploration.cli') is not None
    from lunar_drl_exploration.cli import parser, load_config
    config = load_config(parser().parse_args(['train']))
    assert (config.environments, config.target_rtf, config.warmup) == (8, 30., 1024)
    args = parser().parse_args(['train', '--probe', '--probe-warmup', '4', '--probe-extent', '40', '--probe-budget', '2'])
    config = load_config(args)
    assert config.model.width == 128 and config.model.layers == 6 and config.model.heads == 8
    assert config.learning.batch_size == 64 and config.learning.microbatch_size == 16
    assert config.warmup == 4
    with pytest.raises(ValueError, match='probe'):
        load_config(parser().parse_args(['train', '--probe-warmup', '4']))


def test_curriculum_uses_cumulative_admissions_and_retains_small_scenes():
    assert importlib.util.find_spec('lunar_drl_exploration.training') is not None
    import numpy as np
    from lunar_drl_exploration.training import Curriculum
    from lunar_drl_exploration.config import TrainingConfig
    c = Curriculum(TrainingConfig(), np.random.default_rng(19))
    early = [c.next(i, 0) for i in range(8)]
    assert [s['family'] for s in early].count('moon') == 4
    assert all(40 <= s['extent'] <= 80 and s['episode_budget'] == 512 for s in early)
    late = [c.next(i % 8, 200000) for i in range(200)]
    assert {s['episode_budget'] for s in late} == {512, 2048, 8192}
    assert len({s['seed'] for s in late}) == len(late)


def test_training_real_updates_save_actual_published_policy_and_resume_new_episodes(tmp_path):
    torch = pytest.importorskip('torch')
    import pickle
    from dataclasses import replace
    from lunar_drl_exploration.config import TrainingConfig
    from lunar_drl_exploration.training import run_training
    from lunar_drl_exploration.checkpoint import CheckpointManager
    from lunar_drl_exploration.replay import loads_transport
    config = replace(TrainingConfig(), environments=2, warmup=0, output_dir=str(tmp_path),
        system_reserve_bytes=0, save_interval_s=.2)
    result = run_training(config, device='cpu', max_transitions=8,
        env_factory='worker_fixtures:ControlledEnv', probe_extent=40, probe_budget=3)
    assert result['new_transitions'] >= 8 and result['updates'] >= 1
    saved = CheckpointManager(tmp_path).load(config).record
    published = saved['collector_state']
    assert published['actor_version'] == published['actor_record']['version'] == 0
    assert any(not torch.equal(value, saved['learner']['actor'][key])
        for key, value in published['actor_record']['state_dict'].items())
    assert saved['schedule']['unfinished_reservations'] == 0
    assert loads_transport(saved['counters'])['transitions'] == saved['schedule']['transitions']
    old_episodes = set(result['episode_ids'])
    resumed = run_training(config, resume=True, device='cpu', max_transitions=4,
        env_factory='worker_fixtures:ControlledEnv', probe_extent=40, probe_budget=3)
    assert resumed['transitions'] >= result['transitions'] + 4
    assert set(resumed['episode_ids']).isdisjoint(old_episodes)
    assert resumed['episodes_issued'] > result['episodes_issued']
    assert sorted(p.name for p in tmp_path.iterdir()) == ['metrics.jsonl', 'resume.pt', 'run.json']
    import threading
    assert not [thread for thread in threading.enumerate() if thread.name.startswith('drl-ipc-')]


def test_frozen_evaluation_metrics_keep_failed_cases_and_late_path():
    assert importlib.util.find_spec('lunar_drl_exploration.evaluation') is not None
    from lunar_drl_exploration.evaluation import EpisodeMetrics, summarize
    a = EpisodeMetrics('moon', 40, -1)
    a.observe(.2, 0., 0., 'INITIAL', False, False)
    a.observe(.8, 10., 1., 'GOAL_REACHED', False, False)
    a.observe(.99, 25., 2., 'GOAL_REACHED', False, True)
    b = EpisodeMetrics('moon', 40, -2)
    b.observe(.5, 5., 0., 'NO_PATH', True, False, exhausted=True)
    rows = [a.record(), b.record()]
    assert rows[0]['path_80_to_99_m'] == 15.
    assert rows[1]['path_to_99_m'] is None
    summary = summarize(rows)[0]
    assert summary['rate_99'] == .5 and summary['mean_final_coverage'] == .745
    assert summary['exhaustion_rate'] == .5 and summary['cases'] == 2


def test_fraction_completion_is_not_reported_as_strict_exhaustion():
    from lunar_drl_exploration.evaluation import EpisodeMetrics,summarize
    metric=EpisodeMetrics('moon',40,1)
    metric.observe(.995,10.,1.,'GOAL_REACHED',True,False,exhausted=False,
                   coverage_lower_bound=.991,remaining_area_upper_m2=.9)
    row=metric.record()
    assert row['completed'] and not row['exhausted']
    assert row['coverage_lower_bound']==.991 and row['exhaustion_coverage'] is None
    summary=summarize([row])[0]
    assert summary['completion_rate']==1 and summary['exhaustion_rate']==0


def test_collision_with_new_coverage_is_not_a_successful_completion():
    from lunar_drl_exploration.evaluation import EpisodeMetrics,summarize
    metric=EpisodeMetrics('moon',40,1)
    metric.observe(.995,10.,1.,'COLLISION',True,False,exhausted=False,
                   coverage_lower_bound=.991,remaining_area_upper_m2=.9)
    row=metric.record()
    assert row['reached_99'] and row['collisions']==1
    assert not row['completed'] and summarize([row])[0]['completion_rate']==0


def test_export_actor_uses_existing_schema_and_infer_import_is_observed_only(tmp_path):
    torch = pytest.importorskip('torch')
    from lunar_drl_exploration.sac import SACLearner
    from lunar_drl_exploration.evaluation import export_actor
    from lunar_drl_exploration.runtime import ActorPolicy
    # Existing Actor artifacts can be copied/exported without inventing a schema.
    source, target = tmp_path / 'source.pt', tmp_path / 'actor.pt'
    torch.save(SACLearner().actor_state(), source)
    export_actor(source, target)
    assert ActorPolicy.load(target).actor is not None
    code = '''import sys, importlib.abc
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname in {'lunar_drl_exploration.ros_env','lunar_drl_exploration.scene','lunar_drl_exploration.reference','lunar_drl_exploration.sac'}: raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from lunar_drl_exploration.evaluation import infer
from lunar_drl_exploration.runtime import InferenceRuntime
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get('PYTHONPATH', '')))
    assert result.returncode == 0, result.stderr


def test_launcher_clears_old_overlays_and_preserves_user_device_selection(tmp_path):
    root = Path(__file__).resolve().parents[4]
    ros = tmp_path / 'ros.bash'; cache = tmp_path / 'cache'
    (cache / 'install').mkdir(parents=True)
    ros.write_text('export ROS_DISTRO=jazzy\nexport PYTHONPATH=/new/ros\n')
    (cache / 'install/local_setup.bash').write_text('export PYTHONPATH=/new/drl:$PYTHONPATH\n')
    python = tmp_path / 'python'
    python.write_text('#!/bin/bash\n/usr/bin/env\n')
    python.chmod(0o755)
    env = dict(os.environ, DRL_ROS_SETUP=str(ros), DRL_CACHE=str(cache), DRL_PYTHON=str(python),
        AMENT_PREFIX_PATH='/old/ament', COLCON_PREFIX_PATH='/old/colcon', CMAKE_PREFIX_PATH='/old/cmake',
        ROS_PACKAGE_PATH='/old/pkg', PYTHONPATH='/old/python', LD_LIBRARY_PATH='/old/lib', CUDA_VISIBLE_DEVICES='2')
    result = subprocess.run([str(root / 'scripts/drl/train.sh'), '--help'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    assert values['PYTHONPATH'] == '/new/drl:/new/ros'
    assert values['CUDA_VISIBLE_DEVICES'] == '2'
    assert not any('/old/' in value for key, value in values.items() if key in
        ('AMENT_PREFIX_PATH', 'COLCON_PREFIX_PATH', 'CMAKE_PREFIX_PATH', 'ROS_PACKAGE_PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH'))


def test_resource_stop_saves_an_acknowledged_boundary_and_reason(tmp_path):
    pytest.importorskip('torch')
    from dataclasses import replace
    from lunar_drl_exploration.config import TrainingConfig
    from lunar_drl_exploration.training import run_training
    from lunar_drl_exploration.checkpoint import CheckpointManager
    config = replace(TrainingConfig(), environments=1, output_dir=str(tmp_path),
        system_reserve_bytes=2**60)
    result = run_training(config, device='cpu', max_transitions=10000,
        env_factory='worker_fixtures:ControlledEnv', probe_extent=40, probe_budget=3)
    assert 'system reserve' in result['stop_reason']
    saved = CheckpointManager(tmp_path).load(config).record
    assert saved['schedule']['unfinished_reservations'] == 0
    assert result['transitions'] == saved['schedule']['transitions']


def test_partial_optimizer_failure_keeps_last_good_checkpoint_and_closes_children(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import multiprocessing as mp
    from dataclasses import replace
    from lunar_drl_exploration.config import TrainingConfig
    from lunar_drl_exploration.training import run_training
    from lunar_drl_exploration.checkpoint import CheckpointManager
    from lunar_drl_exploration.sac import SACLearner
    config = replace(TrainingConfig(), environments=2, warmup=0, output_dir=str(tmp_path), system_reserve_bytes=0)
    before = {p.pid for p in mp.active_children()}
    def fail_update(self, *args):
        with torch.no_grad(): next(self.actor.parameters()).add_(10)
        raise RuntimeError('injected mid optimizer failure')
    monkeypatch.setattr(SACLearner, 'update', fail_update)
    import time
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='mid optimizer'):
        run_training(config, device='cpu', max_transitions=12,
            env_factory='worker_fixtures:ControlledEnv', probe_extent=40, probe_budget=3)
    saved = CheckpointManager(tmp_path).load(config).record
    assert saved['schedule']['updates'] == 0 and saved['schedule']['transitions'] == 0
    assert saved['learner']['updates'] == 0
    assert {p.pid for p in mp.active_children()} == before
    import threading
    assert not [thread for thread in threading.enumerate() if thread.name.startswith('drl-ipc-')]
    assert time.monotonic() - started < 8, 'finished messages need ACK draining even after update failure'


def test_default_sixteen_update_publication_captures_the_published_weights(tmp_path):
    torch = pytest.importorskip('torch')
    from dataclasses import replace
    from lunar_drl_exploration.config import TrainingConfig
    from lunar_drl_exploration.training import run_training
    from lunar_drl_exploration.checkpoint import CheckpointManager
    config = replace(TrainingConfig(), environments=2, warmup=0, output_dir=str(tmp_path), system_reserve_bytes=0)
    result = run_training(config, device='cpu', max_transitions=72,
        env_factory='worker_fixtures:ControlledEnv', probe_extent=40, probe_budget=100)
    state = CheckpointManager(tmp_path).load(config).record
    assert 16 <= result['updates'] < 32
    assert state['collector_state']['actor_version'] == 16
    assert state['collector_state']['actor_record']['version'] == 16
    if result['updates'] > 16:
        assert any(not torch.equal(value, state['learner']['actor'][key])
            for key, value in state['collector_state']['actor_record']['state_dict'].items())


@pytest.mark.parametrize('count,seed,budget', [
    (19999, 0, 512), (20000, 0, 2048), (59999, 0, 2048),
    (60000, 0, 8192), (20000, 25, 512), (60000, 25, 2048),
    (60000, 19, 8192),
])
def test_curriculum_exact_admission_boundaries_and_accepted_mixtures(count, seed, budget):
    import numpy as np
    from lunar_drl_exploration.config import TrainingConfig
    from lunar_drl_exploration.training import Curriculum
    # Independent fixed RNG quantiles: seeds0/25/19 draw .63696/.16072/.42038.
    # They distinguish both boundaries and both late-stage CDF cut points.
    spec = Curriculum(TrainingConfig(), np.random.default_rng(seed)).next(0, count)
    assert spec['episode_budget'] == budget


def test_curriculum_yaml_override_roundtrip_and_restored_counter_apply_on_next_reset(tmp_path):
    import numpy as np
    import json
    from lunar_drl_exploration.cli import load_config, parser
    from lunar_drl_exploration.config import config_record, training_config_from_record
    from lunar_drl_exploration.schedule import UpdateSchedule
    from lunar_drl_exploration.training import Curriculum
    path = tmp_path / 'curriculum.yaml'
    path.write_text('curriculum_transition_boundaries: [1, 3]\n'
        'curriculum_mixtures: [[0, 1, 0], [0, 0, 1], [1, 0, 0]]\n')
    config = load_config(parser().parse_args(['train', '--config', str(path)]))
    config = training_config_from_record(json.loads(json.dumps(config_record(config))))
    c = Curriculum(config, np.random.default_rng(0), {'episodes_issued': 21})
    current = c.next(1, 0)
    retained = dict(current)
    assert current['episode_budget'] == 2048
    assert c.next(1, 1)['episode_budget'] == 8192
    schedule = UpdateSchedule()
    for _ in range(3): schedule.collected()
    restored = UpdateSchedule.from_state_dict(schedule.state_dict())
    next_spec = c.next(1, restored.transitions)
    assert next_spec['episode_budget'] == 512
    assert current == retained, 'advancing admissions must not mutate an active reset specification'
    assert next_spec['seed'] == current['seed'] + 2
    assert c.state['episodes_issued'] == 24 and restored.transitions == 3


@pytest.mark.parametrize('field,value', [
    ('curriculum_transition_boundaries', (3, 3)),
    ('curriculum_transition_boundaries', (-1, 3)),
    ('curriculum_transition_boundaries', (1.5, 3)),
    ('curriculum_transition_boundaries', (1,)),
    ('curriculum_mixtures', ((1., 0., 0.),)),
    ('curriculum_mixtures', ((1., 0., 0.), (.5, .6, 0.), (.15, .25, .6))),
    ('curriculum_mixtures', ((1., 0., 0.), (-.1, 1.1, 0.), (.15, .25, .6))),
    ('curriculum_mixtures', ((1., 0., 0.), (float('nan'), 1., 0.), (.15, .25, .6))),
])
def test_curriculum_configuration_rejects_invalid_scalar_or_probability_shapes(field, value):
    from dataclasses import replace
    from lunar_drl_exploration.config import TrainingConfig
    with pytest.raises(ValueError, match='curriculum'):
        replace(TrainingConfig(), **{field: value})


@pytest.mark.parametrize('missing', [False, True])
def test_training_final_transport_handoff_after_collector_exit_saves_or_reports_missing_stop(
        tmp_path, monkeypatch, missing):
    torch = pytest.importorskip('torch')
    import multiprocessing as mp
    import pickle
    import threading
    import time
    from dataclasses import replace
    from lunar_drl_exploration import ipc
    from lunar_drl_exploration.config import TrainingConfig
    from lunar_drl_exploration.training import run_training
    from lunar_drl_exploration.checkpoint import CheckpointManager
    from lunar_drl_exploration.replay import loads_transport
    decoded, exited, second_empty_poll, release = [threading.Event() for _ in range(4)]
    final = {}
    real_duplex = ipc.Duplex
    class DelayedFinal:
        def __init__(self, raw): self.raw = raw
        def recv(self):
            message = self.raw.recv()
            if message['kind'] == 'STOPPED':
                final.update(pickle.loads(message['payload']))
                decoded.set()
                if missing: return self.raw.recv()  # actual ordered EOF, no STOPPED
                assert release.wait(5), 'test failed to release final reader handoff'
            return message
        def __getattr__(self, name): return getattr(self.raw, name)
    class DelayedDuplex(real_duplex):
        empty_polls = 0
        def __init__(self, raw, *args, **kwargs):
            super().__init__(DelayedFinal(raw), *args, **kwargs)
        def poll(self, timeout=0):
            ready = super().poll(timeout)
            if timeout and not ready and exited.is_set():
                self.empty_polls += 1
                if self.empty_polls >= 2: second_empty_poll.set()
            return ready
        def close(self, **kwargs):
            release.set()  # failure-only cleanup also joins the blocked reader
            return super().close(**kwargs)
    monkeypatch.setattr(ipc, 'Duplex', DelayedDuplex)
    def deliver_after_exit():
        if not decoded.wait(8): return
        if missing: return
        collectors = [p for p in mp.active_children() if p.name == 'drl-cpu-collector']
        assert len(collectors) == 1
        collectors[0].join(3)
        assert collectors[0].exitcode == 0
        exited.set()
        if not missing and second_empty_poll.wait(3): release.set()
    coordinator = threading.Thread(target=deliver_after_exit)
    coordinator.start()
    config = replace(TrainingConfig(), environments=2, warmup=0, output_dir=str(tmp_path),
        system_reserve_bytes=0, save_interval_s=1800)
    started = time.monotonic()
    try:
        if missing:
            with pytest.raises(RuntimeError, match='transport ended before acknowledged stop'):
                run_training(config, device='cpu', max_transitions=8,
                    env_factory='worker_fixtures:ControlledEnv', probe_extent=40, probe_budget=3)
            saved = CheckpointManager(tmp_path).load(config).record
            assert saved['schedule']['transitions'] == saved['schedule']['updates'] == 0
        else:
            result = run_training(config, device='cpu', max_transitions=8,
                env_factory='worker_fixtures:ControlledEnv', probe_extent=40, probe_budget=3)
            saved = CheckpointManager(tmp_path).load(config).record
            assert second_empty_poll.is_set() and exited.is_set()
            assert result['transitions'] >= 8
            assert loads_transport(saved['counters'])['transitions'] == saved['schedule']['transitions'] == result['transitions']
            assert saved['schedule']['unfinished_reservations'] == 0
            from fractions import Fraction
            assert saved['schedule']['updates'] == saved['learner']['updates'] == result['updates']
            assert Fraction(saved['schedule']['credit']) == Fraction(result['transitions'], 4) - result['updates']
            published = saved['collector_state']
            assert published['actor_version'] == final['actor_version'] == 0
            assert published['sequences'] == final['sequences']
            assert published['acknowledged'] == final['acknowledged']
            torch.testing.assert_close(published['policy_rng'], final['policy_rng'], atol=0, rtol=0)
            for key, value in final['actor_record']['state_dict'].items():
                torch.testing.assert_close(published['actor_record']['state_dict'][key], value, atol=0, rtol=0)
    finally:
        release.set(); coordinator.join(8)
    assert time.monotonic() - started < 12
    assert not coordinator.is_alive() and not mp.active_children()
    assert not [t for t in threading.enumerate() if t.name.startswith('drl-ipc-')]


def test_installed_console_success_exits_zero_and_python_api_keeps_result(tmp_path):
    torch = pytest.importorskip('torch')
    ament = pytest.importorskip('ament_index_python.packages')
    from lunar_drl_exploration.cli import main
    from lunar_drl_exploration.sac import SACLearner
    from lunar_drl_exploration.runtime import ActorPolicy
    console = Path(ament.get_package_prefix('lunar_drl_exploration')) / 'lib/lunar_drl_exploration/lunar-drl'
    assert console.is_file(), 'build the affected package before installed-console validation'
    source, api_target, cli_target = [tmp_path / name for name in ('source.pt', 'api.pt', 'cli.pt')]
    torch.save(SACLearner().actor_state(), source)
    assert main(['export', '--checkpoint', str(source), '--output', str(api_target)]) == str(api_target)
    result = subprocess.run([sys.executable, str(console), 'export', '--checkpoint', str(source),
        '--output', str(cli_target)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert ActorPolicy.load(cli_target).actor is not None


def test_episode_progress_does_not_mix_old_reward_and_new_reset_pose():
    from lunar_drl_exploration.training import _merge_progress
    old = dict(episode_id='old', state='READY', seed=7, reference_coverage=.8,
        new_area_m2=48., reason_code='GOAL_REACHED', distance_m=30.)
    reset_progress = dict(episode_id='new', state='READY', steps=0, known_area_m2=0., distance_m=0.)
    fresh = _merge_progress(old, reset_progress)
    assert fresh == reset_progress
    assert old['new_area_m2'] == 48.
    resetting = dict(state='RESETTING', seed=8, family='moon', extent_m=40., budget=8)
    assert _merge_progress(resetting, dict(old, steps=8)) == resetting
    assert _merge_progress(fresh, dict(episode_id='new', steps=1))['known_area_m2'] == 0.

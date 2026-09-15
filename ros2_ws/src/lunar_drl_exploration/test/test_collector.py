import importlib.util
import multiprocessing as mp
import time
import pytest
from lunar_drl_exploration.config import TrainingConfig, config_record
from lunar_drl_exploration.replay import loads_transport


def receive(pipe, kind, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pipe.poll(.1):
            message = pipe.recv()
            if message['kind'] == kind:
                return message
    raise AssertionError('no ' + kind)


def test_worker_retains_completion_until_ack_and_cancels_active_step():
    assert importlib.util.find_spec('lunar_drl_exploration.worker') is not None
    from lunar_drl_exploration.worker import worker_main
    parent, child = mp.get_context('spawn').Pipe()
    process = mp.get_context('spawn').Process(target=worker_main,
        args=(child, config_record(TrainingConfig()), 0),
        kwargs={'env_factory': 'worker_fixtures:ControlledEnv'})
    process.start(); child.close()
    try:
        hello = receive(parent, 'HELLO')
        assert not hello['torch_loaded'] and hello['numeric_threads'] == '1'
        parent.send(dict(kind='RESET', spec=dict(seed=1, family='moon', extent=40, episode_budget=3)))
        scene = receive(parent, 'SCENE')
        assert len(loads_transport(scene['payload'])) == 1
        receive(parent, 'READY')
        parent.send(dict(kind='STEP', token=1, action=0, actor_version=7))
        result = receive(parent, 'DONE')
        transition = loads_transport(result['payload'])
        assert transition.actor_version == 7 and not transition.observation.features.flags.writeable
        assert not parent.poll(.2), 'worker must retain DONE, not send READY before admission ACK'
        retry = receive(parent, 'DONE', timeout=3)
        assert retry['token'] == 1 and retry['payload'] == result['payload']
        parent.send(dict(kind='ACK', token=1))
        receive(parent, 'READY')
        parent.send(dict(kind='RESET', spec=dict(seed=-1, family='moon', extent=40, episode_budget=3)))
        receive(parent, 'SCENE'); receive(parent, 'READY')
        parent.send(dict(kind='STEP', token=2, action=0, actor_version=7))
        time.sleep(.1)
        started = time.monotonic()
        parent.send(dict(kind='STOP'))
        canceled = receive(parent, 'ABORTED', timeout=3)
        assert canceled['token'] == 2
        receive(parent, 'CLOSED', timeout=3)
        process.join(3)
        assert process.exitcode == 0 and time.monotonic() - started < 3
    finally:
        if process.is_alive(): process.terminate()
        process.join(3); parent.close()


def test_admission_deduplicates_lost_ack_and_releases_replaced_scenes():
    assert importlib.util.find_spec('lunar_drl_exploration.training') is not None
    from lunar_drl_exploration.training import AdmissionLedger
    from lunar_drl_exploration.replay import ReplayBuffer
    from lunar_drl_exploration.schedule import UpdateSchedule
    from worker_fixtures import ControlledEnv
    replay, schedule = ReplayBuffer(), UpdateSchedule(warmup=0)
    ledger = AdmissionLedger(replay, schedule)
    env = ControlledEnv(None, 1)
    for token in (1, 2):
        env.reset(token, 'moon', 40, episode_budget=1)
        ledger.scene(1, env.static_scenes)
        schedule.reserve_dispatch()
        t = env.step(0, actor_version=0)
        assert ledger.admit(1, token, t)
        assert not ledger.admit(1, token, t)
    assert schedule.transitions == len(replay) == 2 and schedule.credit == .5
    assert len(ledger.active_scenes) == 1 and len(replay.scenes) == 2
    replay.evict_oldest()
    assert len(replay.scenes) == 1


def test_async_collector_fast_slot_progress_barrier_publication_and_stop():
    assert importlib.util.find_spec('lunar_drl_exploration.collector') is not None
    torch = pytest.importorskip('torch')
    import pickle
    from dataclasses import replace
    from lunar_drl_exploration.collector import collector_main
    from lunar_drl_exploration.sac import SACLearner
    from lunar_drl_exploration.training import AdmissionLedger
    from lunar_drl_exploration.replay import ReplayBuffer
    from lunar_drl_exploration.schedule import UpdateSchedule
    actor = SACLearner().actor_state()
    config = replace(TrainingConfig(), environments=2)
    parent, child = mp.get_context('spawn').Pipe()
    process = mp.get_context('spawn').Process(target=collector_main,
        args=(child, config_record(config), pickle.dumps(dict(actor_record=actor,
            actor_version=0, policy_rng=torch.get_rng_state(), sequences={}))),
        kwargs={'env_factory': 'worker_fixtures:ControlledEnv'})
    process.start(); child.close()
    ledger = AdmissionLedger(ReplayBuffer(), UpdateSchedule(warmup=0, max_credit=1))
    completed = []
    barrier = None
    try:
        deadline = time.monotonic() + 20
        requested_barrier = False
        while time.monotonic() < deadline and barrier is None:
            if not parent.poll(.1): continue
            msg = parent.recv(); kind = msg['kind']; env_id = msg.get('env')
            if kind == 'NEED_RESET':
                parent.send(dict(kind='RESET', env=env_id,
                    spec=dict(seed=1, family='moon', extent=40, episode_budget=100)))
            elif kind == 'SCENE': ledger.scene(env_id, loads_transport(msg['payload']))
            elif kind == 'RESERVE':
                if ledger.schedule.can_dispatch():
                    ledger.schedule.reserve_dispatch()
                    parent.send(dict(kind='GRANT', env=env_id, token=msg['token']))
            elif kind == 'DONE':
                transition = loads_transport(msg['payload'])
                ledger.admit(env_id, msg['token'], transition)
                completed.append(env_id)
                parent.send(dict(kind='ACK', env=env_id, token=msg['token']))
                if len(completed) >= 3 and not requested_barrier:
                    parent.send(dict(kind='BARRIER'))
                    requested_barrier = True
            elif kind == 'BARRIER': barrier = pickle.loads(msg['payload'])
            elif kind == 'FATAL': pytest.fail(msg['reason'])
        assert barrier is not None
        assert completed[:2] == [1, 1], 'fast worker must proceed independently of slow env0'
        assert ledger.schedule.transitions == len(completed) == len(ledger.replay)
        assert ledger.schedule.transitions + ledger.schedule.inflight <= 4
        assert barrier['actor_version'] == barrier['actor_record']['version'] == 0
        assert torch.equal(barrier['policy_rng'], torch.get_rng_state()) is False
        parent.send(dict(kind='STOP'))
        while True:
            msg = receive_any(parent, 5)
            if msg['kind'] == 'DONE':
                ledger.admit(msg['env'], msg['token'], loads_transport(msg['payload']))
                parent.send(dict(kind='ACK', env=msg['env'], token=msg['token']))
            elif msg['kind'] == 'ABORTED': ledger.schedule.cancel_dispatch()
            elif msg['kind'] == 'STOPPED': break
        process.join(5)
        assert process.exitcode == 0 and ledger.schedule.inflight == 0
    finally:
        if process.is_alive(): process.terminate()
        process.join(5); parent.close()


def receive_any(pipe, timeout):
    assert pipe.poll(timeout), 'collector did not respond'
    return pipe.recv()


@pytest.mark.parametrize('seed', [-2, -1])
def test_collector_recovery_is_bounded_and_reports_every_unfinished_reservation(seed):
    torch = pytest.importorskip('torch')
    import pickle
    from dataclasses import replace
    from lunar_drl_exploration.collector import collector_main
    from lunar_drl_exploration.model import Actor
    from lunar_drl_exploration.config import config_record, ModelConfig
    actor = dict(schema='task_graph_v1', model_config=config_record(ModelConfig()),
        version=0, state_dict=Actor().state_dict())
    parent, child = mp.get_context('spawn').Pipe()
    process = mp.get_context('spawn').Process(target=collector_main,
        args=(child, config_record(replace(TrainingConfig(), environments=1)),
              pickle.dumps(dict(actor_record=actor, actor_version=0, policy_rng=torch.get_rng_state()))),
        kwargs=dict(env_factory='worker_fixtures:ControlledEnv', recovery_limit=1, worker_timeout_s=1.))
    process.start(); child.close()
    aborts, retries, fatal, reasons = [], [], None, []
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            msg = receive_any(parent, 5)
            if msg['kind'] == 'NEED_RESET':
                parent.send(dict(kind='RESET', env=0,
                    spec=dict(seed=seed, family='moon', extent=40, episode_budget=2)))
            elif msg['kind'] == 'RESERVE': parent.send(dict(kind='GRANT', env=0, token=msg['token']))
            elif msg['kind'] == 'ABORTED':
                aborts.append(msg['token']); reasons.append(msg['reason'])
            elif msg['kind'] == 'RECOVERY': retries.append(msg['attempt'])
            elif msg['kind'] == 'FATAL': fatal = msg['reason']
            elif msg['kind'] == 'STOPPED': break
        process.join(5)
        assert process.exitcode == 0
        assert len(aborts) == len(set(aborts)) == 2
        assert retries == [1, 2] and 'recovery exhausted' in fatal
        if seed == -1: assert all('controlled cancellation' in reason for reason in reasons)
    finally:
        if process.is_alive(): process.terminate()
        process.join(5); parent.close()


def test_worker_keeps_native_spawning_thread_alive_between_reset_and_step():
    from lunar_drl_exploration.worker import worker_main
    parent, child = mp.get_context('spawn').Pipe()
    process = mp.get_context('spawn').Process(target=worker_main,
        args=(child, config_record(TrainingConfig()), 0),
        kwargs={'env_factory': 'worker_fixtures:OwnedChildEnv'})
    process.start(); child.close()
    try:
        receive(parent, 'HELLO')
        parent.send(dict(kind='RESET', spec=dict(seed=1, family='moon', extent=40, episode_budget=2)))
        receive(parent, 'SCENE'); receive(parent, 'READY')
        time.sleep(.25)
        parent.send(dict(kind='STEP', token=1, action=0, actor_version=0))
        result = receive_any(parent, 3)
        assert result['kind'] == 'DONE', result
        parent.send(dict(kind='ACK', token=1)); parent.send(dict(kind='STOP'))
        receive(parent, 'CLOSED'); process.join(3)
        assert process.exitcode == 0
    finally:
        if process.is_alive(): process.terminate()
        process.join(3); parent.close()

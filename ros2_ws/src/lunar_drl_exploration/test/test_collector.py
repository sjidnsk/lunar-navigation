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
            elif msg['kind'] == 'RECOVERY':
                retries.append(msg['attempt'])
                assert msg['reset_spec'] == dict(seed=seed, family='moon', extent=40, episode_budget=2)
                assert msg['stage'] in ('ACTIVE', 'WAITING')
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


@pytest.mark.parametrize('large_kind', ['SCENE', 'DONE'])
def test_large_bidirectional_actor_publication_and_stop_drain_without_external_reader(large_kind):
    """Reproduce review I1 with real collector/Actor and a >socket-sized payload."""
    torch = pytest.importorskip('torch')
    import os
    import pickle
    import threading
    from dataclasses import replace
    from lunar_drl_exploration.collector import collector_main
    from lunar_drl_exploration.config import ModelConfig
    from lunar_drl_exploration.model import Actor
    from lunar_drl_exploration.replay import ReplayBuffer
    from lunar_drl_exploration.schedule import UpdateSchedule
    from lunar_drl_exploration.training import AdmissionLedger
    torch.set_num_threads(1)
    actor = dict(schema='task_graph_v1', model_config=config_record(ModelConfig()),
        version=0, state_dict=Actor().state_dict())
    state = pickle.dumps(dict(actor_record=actor, actor_version=0, policy_rng=torch.get_rng_state()))
    publication = dict(actor, version=16)
    publication['state_dict'] = {key: value + .01 for key, value in actor['state_dict'].items()}
    from worker_fixtures import large_transport_collector
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    entered, sent = context.Event(), context.Event()
    process = context.Process(target=large_transport_collector, args=(child,
        config_record(replace(TrainingConfig(), environments=1)), state, large_kind, entered, sent))
    process.start(); child.close()
    ledger = AdmissionLedger(ReplayBuffer(), UpdateSchedule(warmup=0))
    sender = None
    publication_done = threading.Event()
    worker_pids = []
    stopped = None
    try:
        deadline = time.monotonic() + 10
        while not entered.is_set() and time.monotonic() < deadline:
            if large_kind == 'SCENE' and entered.wait(.01): break
            if not parent.poll(.05): continue
            message = parent.recv()
            if message['kind'] == 'WORKER_HELLO': worker_pids.append(message['pid'])
            elif message['kind'] == 'NEED_RESET':
                parent.send(dict(kind='RESET', env=0,
                    spec=dict(seed=1, family='moon', extent=40, episode_budget=10)))
                if large_kind == 'SCENE': assert entered.wait(5); break
            elif message['kind'] == 'SCENE': ledger.scene(0, loads_transport(message['payload']))
            elif message['kind'] == 'RESERVE':
                ledger.schedule.reserve_dispatch()
                parent.send(dict(kind='GRANT', env=0, token=message['token']))
                if large_kind == 'DONE': assert entered.wait(5); break
        assert entered.is_set()
        def publish():
            parent.send(dict(kind='PUBLISH', payload=pickle.dumps(publication, protocol=5)))
            publication_done.set()
        sender = threading.Thread(target=publish, daemon=True)
        sender.start()
        # No learner-side reader intervention: the production receiver must keep
        # draining while its independent writer uploads the large scene/result.
        assert publication_done.wait(2), 'bidirectional large sends deadlocked'
        parent.send(dict(kind='BARRIER')); parent.send(dict(kind='STOP'))
        while stopped is None:
            message = receive_any(parent, 10)
            if message['kind'] == 'SCENE': ledger.scene(0, loads_transport(message['payload']))
            elif message['kind'] == 'DONE':
                ledger.admit(0, message['token'], loads_transport(message['payload']))
                parent.send(dict(kind='ACK', env=0, token=message['token']))
            elif message['kind'] == 'ABORTED': ledger.schedule.cancel_dispatch()
            elif message['kind'] == 'STOPPED': stopped = pickle.loads(message['payload'])
            elif message['kind'] == 'FATAL': pytest.fail(message['reason'])
        process.join(5)
        assert process.exitcode == 0 and sent.is_set()
        assert ledger.schedule.inflight == 0
        assert ledger.schedule.transitions == len(ledger.replay) == (large_kind == 'DONE')
        assert ledger.schedule.credit == (.25 if large_kind == 'DONE' else 0)
        assert stopped['actor_version'] == stopped['actor_record']['version'] == 16
        for key, value in publication['state_dict'].items():
            torch.testing.assert_close(stopped['actor_record']['state_dict'][key], value, atol=0, rtol=0)
        assert all(not os.path.exists(f'/proc/{pid}') for pid in worker_pids)
    finally:
        # Failure-only dismantling of the old deadlock; never part of PASS criteria.
        if sender is not None and sender.is_alive():
            if parent.poll(1): parent.recv()
            sender.join(3)
        if process.is_alive():
            try: parent.send(dict(kind='STOP'))
            except (EOFError, OSError): pass
            deadline = time.monotonic() + 3
            while process.is_alive() and time.monotonic() < deadline:
                if parent.poll(.05):
                    try:
                        message = parent.recv()
                        if message['kind'] == 'DONE': parent.send(dict(kind='ACK', env=0, token=message['token']))
                    except (EOFError, OSError): break
            process.join(2)
        if process.is_alive(): process.terminate()
        process.join(3); parent.close()


@pytest.mark.parametrize('control', [None, 'BARRIER', 'STOP'])
def test_currently_queued_grants_batch_and_queued_stop_or_barrier_prevents_inference(control):
    """Real Actor.forward instrumentation; no probability/action substitution."""
    torch = pytest.importorskip('torch')
    import os
    import pickle
    import signal
    from dataclasses import replace
    from lunar_drl_exploration.collector import collector_main
    from lunar_drl_exploration.config import ModelConfig
    from lunar_drl_exploration.model import Actor
    torch.set_num_threads(1)
    actor = dict(schema='task_graph_v1', model_config=config_record(ModelConfig()), version=0,
        state_dict=Actor().state_dict())
    rng_state = torch.get_rng_state()
    state = pickle.dumps(dict(actor_record=actor, actor_version=0, policy_rng=rng_state))
    from worker_fixtures import instrumented_batch_collector
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    sizes, count = context.Array('i', 8), context.Value('i', 0)
    process = context.Process(target=instrumented_batch_collector, args=(child,
        config_record(replace(TrainingConfig(), environments=2)), state, sizes, count))
    process.start(); child.close()
    requests, completed, canceled = {}, 0, 0
    stopped = None
    try:
        while stopped is None:
            message = receive_any(parent, 10); kind = message['kind']; env = message.get('env')
            if kind == 'NEED_RESET':
                parent.send(dict(kind='RESET', env=env,
                    spec=dict(seed=1, family='moon', extent=40, episode_budget=10)))
            elif kind == 'RESERVE' and len(requests) < 2:
                requests[env] = message['token']
                if len(requests) == 2:
                    # Queue all controls while this exact owned process is paused.
                    # This proves already-readable batching, not an artificial wait
                    # in production to collect a slow environment.
                    os.kill(process.pid, signal.SIGSTOP)
                    for env_id, token in requests.items(): parent.send(dict(kind='GRANT', env=env_id, token=token))
                    if control: parent.send(dict(kind=control))
                    os.kill(process.pid, signal.SIGCONT)
            elif kind == 'DONE':
                completed += 1
                parent.send(dict(kind='ACK', env=env, token=message['token']))
                if completed == 2: parent.send(dict(kind='STOP'))
            elif kind == 'ABORTED': canceled += 1
            elif kind == 'BARRIER':
                snapshot = pickle.loads(message['payload'])
                torch.testing.assert_close(snapshot['policy_rng'], rng_state, atol=0, rtol=0)
                parent.send(dict(kind='STOP'))
            elif kind == 'STOPPED': stopped = message
            elif kind == 'FATAL': pytest.fail(message['reason'])
        process.join(5)
        assert process.exitcode == 0
        assert list(sizes[:count.value]) == ([2] if control is None else [])
        assert completed + canceled == 2
    finally:
        if process.is_alive():
            os.kill(process.pid, signal.SIGCONT)
            process.terminate()
        process.join(3); parent.close()


def test_duplex_shutdown_joins_blocked_large_writer_and_reader():
    import threading
    from lunar_drl_exploration.ipc import Duplex
    raw, peer = mp.Pipe()
    entered = threading.Event()
    class ObservedPipe:
        def send(self, message):
            entered.set()
            raw.send(message)
        def __getattr__(self, name): return getattr(raw, name)
    endpoint = Duplex(ObservedPipe())
    endpoint.send(dict(kind='SCENE', payload=b'x' * 1024**2))
    assert entered.wait(1)
    # The peer deliberately never reads. A flush failure must be explicit and
    # shutdown must interrupt both native syscalls, joining non-daemon owners.
    started = time.monotonic()
    with pytest.raises(BrokenPipeError, match='before reliable messages drained'):
        endpoint.close(timeout=.1)
    assert time.monotonic() - started < 1
    assert all(not thread.is_alive() and not thread.daemon for thread in endpoint.threads)
    peer.close()


def test_duplex_mailbox_preserves_reliable_order_and_distinct_transition_tokens():
    from collections import deque
    from lunar_drl_exploration.ipc import Duplex
    endpoint = object.__new__(Duplex)
    endpoint.capacity = 20
    messages = deque()
    for message in [dict(kind='SCENE'), dict(kind='DONE', env=0, token=1),
            dict(kind='STATUS', env=0, sample=1), dict(kind='ACK', token=1),
            dict(kind='DONE', env=0, token=1), dict(kind='STATUS', env=0, sample=2),
            dict(kind='DONE', env=0, token=2), dict(kind='PUBLISH'), dict(kind='BARRIER')]:
        endpoint._append(messages, message)
    assert [(m['kind'], m.get('token')) for m in messages] == [
        ('SCENE', None), ('DONE', 1), ('ACK', 1), ('STATUS', None),
        ('DONE', 2), ('PUBLISH', None), ('BARRIER', None)]
    assert messages[3]['sample'] == 2


def test_duplex_delivers_final_large_completion_before_peer_eof():
    import threading
    from lunar_drl_exploration.ipc import Duplex
    raw, peer = mp.Pipe()
    endpoint = Duplex(raw)
    payload = b'final-transition' * 100000
    def finish():
        peer.send(dict(kind='DONE', env=0, token=7, payload=payload))
        peer.close()
    sender = threading.Thread(target=finish)
    sender.start()
    try:
        assert endpoint.poll(2)
        assert endpoint.recv() == dict(kind='DONE', env=0, token=7, payload=payload)
        assert endpoint.poll(2)
        with pytest.raises(EOFError): endpoint.recv()
    finally:
        endpoint.close(flush=False)
        sender.join(2)
    assert not sender.is_alive()

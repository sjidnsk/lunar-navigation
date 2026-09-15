"""CPU Actor owner; independent bounded worker pipes and admission ACK barrier.

Module imports are stdlib only. Each slot retains at most one observation and one
finished transition. Static scene payloads are forwarded once and not archived.
"""
import multiprocessing as mp
import threading
import os
import pickle
import signal
import time


def collector_main(pipe, config_record, initial_state, *, domain_base=210,
                   env_factory='lunar_drl_exploration.ros_env:RosExplorationEnv',
                   recovery_limit=3, worker_timeout_s=180., stop_timeout_s=10.):
    from .worker import numerical_threads, worker_main
    from .ipc import Duplex
    wake = threading.Event()
    numerical_threads(config_record['collector_threads'])
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch
    from .config import ModelConfig
    from .model import Actor
    from .replay import loads_transport
    torch.set_num_threads(config_record['collector_threads'])
    torch.set_num_interop_threads(1)
    state = pickle.loads(initial_state)
    actor_record = state['actor_record']
    actor = Actor(ModelConfig(**actor_record['model_config'])).eval()
    actor.load_state_dict(actor_record['state_dict'])
    generator = torch.Generator(device='cpu')
    generator.set_state(state['policy_rng'])
    sequences = dict(state.get('sequences', {}))
    acknowledged = dict(state.get('acknowledged', {}))
    context = mp.get_context('spawn')
    slots = {}
    paused = False
    barrier_requested = False
    frozen = False
    stopping = False
    stop_started = None

    def emit(kind, **values):
        pipe.send(dict(kind=kind, **values))

    def spawn(env_id, failures=0):
        parent, child = context.Pipe()
        process = context.Process(target=worker_main, args=(child, config_record, env_id),
            kwargs=dict(domain_base=domain_base, env_factory=env_factory), name=f'drl-env-{env_id}')
        process.start(); child.close()
        parent = Duplex(parent, wake=wake)
        slots[env_id] = dict(pipe=parent, process=process, mode='STARTING', observation=None,
            pending=None, token=None, failures=failures, since=time.monotonic(), closed=False)
        emit('OWNERSHIP', env=env_id, worker_pid=process.pid, child_pids=[])

    def snapshot():
        return pickle.dumps(dict(actor_record=actor_record, actor_version=actor_record['version'],
            policy_rng=generator.get_state().clone(), sequences=sequences, acknowledged=acknowledged), protocol=5)

    def request_stop():
        nonlocal stopping, paused, frozen, stop_started, barrier_requested
        if stopping: return
        stopping = paused = True
        frozen = False
        barrier_requested = False
        stop_started = time.monotonic()
        for env_id, slot in slots.items():
            if slot['mode'] == 'GRANTED':
                emit('ABORTED', env=env_id, token=slot['token'], reason='reserved but not dispatched at stop')
                slot['mode'] = 'READY'
            if not slot['closed']:
                try: slot['pipe'].send(dict(kind='STOP'))
                except (EOFError, BrokenPipeError): pass

    def recover(env_id, reason, stage=None):
        slot = slots[env_id]
        if slot['pending'] is not None:
            # A constructed transition is already retained in this process and
            # must be admitted first, even if the worker itself has died.
            slot['recover_after_ack'] = reason
            return
        if slot['mode'] == 'ACTIVE':
            emit('ABORTED', env=env_id, token=slot['token'], reason=reason)
        process = slot['process']
        if process.is_alive(): process.terminate()
        process.join(3)
        if process.is_alive(): process.kill(); process.join(3)
        slot['pipe'].close(flush=False); slot['closed'] = True
        emit('RECOVERY', env=env_id, reason=reason, attempt=slot['failures'] + 1,
            stage=stage or slot['mode'], reset_spec=slot.get('reset_spec'))
        if stopping: return
        if slot['failures'] >= recovery_limit:
            emit('FATAL', env=env_id, reason=f'infrastructure recovery exhausted: {reason}')
            request_stop()
            return
        spawn(env_id, slot['failures'] + 1)

    interval = config_record['actor_publish_updates']
    pipe = Duplex(pipe, config_record['environments'], wake,
        publication_window=(config_record['max_update_credit'] + interval - 1) // interval)
    try:
        emit('COLLECTOR_HELLO', pid=os.getpid(), device='cpu', numeric_threads=torch.get_num_threads(),
            cuda_initialized=torch.cuda.is_initialized(), package_path=__file__)
        for env_id in range(config_record['environments']): spawn(env_id)
        while True:
            sources = [pipe] + ([] if frozen else [s['pipe'] for s in slots.values() if not s['closed']])
            wake.clear()
            if not any(source.poll() for source in sources): wake.wait(.02)
            # A bounded currently-readable control pass precedes each Actor call.
            # No wait for other workers or for future grants.
            for _ in range(pipe.capacity):
                if not pipe.poll(): break
                command = pipe.recv(); kind = command['kind']
                if kind == 'STOP': request_stop()
                elif kind == 'BARRIER':
                    paused = barrier_requested = True
                elif kind == 'RESUME':
                    paused = frozen = False
                elif kind == 'PUBLISH':
                    actor_record = pickle.loads(command['payload'])
                    actor.load_state_dict(actor_record['state_dict'])
                    emit('PUBLISHED', version=actor_record['version'])
                elif kind in ('RESET', 'GRANT', 'ACK'):
                    env_id = command['env']; slot = slots[env_id]
                    if kind == 'ACK':
                        pending = slot['pending']
                        if pending is not None and pending['token'] == command['token']:
                            acknowledged[env_id] = command['token']
                            slot['pending'] = None
                            slot['mode'] = 'WAITING'
                            if 'recover_after_ack' in slot:
                                recover(env_id, slot.pop('recover_after_ack'))
                            else:
                                try: slot['pipe'].send(dict(kind='ACK', token=command['token']))
                                except (EOFError, BrokenPipeError): recover(env_id, 'worker died after completion')
                    elif kind == 'RESET':
                        if not stopping:
                            slot['reset_spec'] = dict(command['spec'])
                            slot['pipe'].send(dict(kind='RESET', spec=command['spec']))
                            slot['mode'] = 'RESETTING'; slot['since'] = time.monotonic()
                    else:
                        if slot['mode'] != 'REQUESTED' or command['token'] != slot['token']:
                            raise RuntimeError('grant does not match outstanding request')
                        if paused:
                            emit('ABORTED', env=env_id, token=slot['token'], reason='reserved but not dispatched at pause')
                            slot['mode'] = 'READY'
                        else:
                            slot['mode'] = 'GRANTED'
                else: raise ValueError('unknown collector command ' + kind)
            # Drain only currently available messages; never wait for all workers.
            if not frozen:
                for env_id, slot in list(slots.items()):
                    if slot['closed']: continue
                    connection = slot['pipe']
                    while slots.get(env_id) is slot and not slot['closed'] and connection.poll():
                        try: message = connection.recv()
                        except (EOFError, OSError):
                            recover(env_id, 'worker pipe closed'); break
                        kind = message['kind']
                        if kind == 'READY':
                            slot['observation'] = loads_transport(message['payload']); slot['mode'] = 'READY'
                        elif kind == 'DONE':
                            if message['token'] <= acknowledged.get(env_id, -1):
                                connection.send(dict(kind='ACK', token=message['token']))
                                continue
                            if slot['pending'] is not None and slot['pending']['token'] != message['token']:
                                raise RuntimeError('worker completed a second action without ACK')
                            slot['pending'] = message; slot['mode'] = 'PENDING'; slot['failures'] = 0
                            slot['observation'] = None
                            emit('DONE', **{k: v for k, v in message.items() if k != 'kind'})
                        elif kind in ('ABORTED', 'ERROR'):
                            failed_stage = slot['mode']
                            # Retire exactly the reservation associated with this action.
                            if slot['mode'] == 'ACTIVE':
                                emit('ABORTED', env=env_id, token=slot['token'], reason=message['reason'])
                            slot['mode'] = 'WAITING'
                            if not stopping and 'restart_reason' not in slot:
                                recover(env_id, message['reason'], stage=failed_stage)
                        elif kind == 'HELLO':
                            emit('WORKER_HELLO', **{k: v for k, v in message.items() if k != 'kind'})
                            if not stopping: emit('NEED_RESET', env=env_id)
                        elif kind == 'CLOSED':
                            slot['closed'] = True
                            slot['process'].join(3); connection.close(flush=False)
                            emit('OWNERSHIP', env=env_id, worker_pid=None, child_pids=[])
                            if 'restart_reason' in slot and not stopping:
                                recover(env_id, slot['restart_reason'])
                            break
                        elif kind == 'STATUS':
                            emit('STATUS', **{k: v for k, v in message.items() if k != 'kind'})
                        elif kind in ('SCENE', 'EPISODE_END', 'NEED_RESET'):
                            if kind == 'NEED_RESET': slot['mode'] = 'WAIT_RESET'
                            emit(kind, **{k: v for k, v in message.items() if k != 'kind'})
                        else: raise ValueError('unknown worker message ' + kind)
                    if slots.get(env_id) is not slot or slot['closed']: continue
                    if not slot['process'].is_alive():
                        # The reader may still be decoding a final large DONE.
                        # Ordered EOF follows its inbox; do not discard that
                        # frame merely because process exit won the race.
                        deadline = slot.setdefault('exit_deadline', time.monotonic() + stop_timeout_s)
                        if time.monotonic() > deadline:
                            recover(env_id, 'worker exit transport drain timed out')
                    elif 'restart_reason' in slot:
                        if time.monotonic() > slot['restart_deadline'] and slot['pending'] is None:
                            recover(env_id, slot['restart_reason'] + '; cancellation join timed out')
                    elif slot['mode'] in ('STARTING', 'RESETTING', 'ACTIVE') and time.monotonic() - slot['since'] > worker_timeout_s:
                        # First ask the responsive control loop to cancel and
                        # flush any constructed completion. Hard termination is
                        # reserved for an explicit bounded infrastructure hang.
                        slot['restart_reason'] = 'bounded worker infrastructure watchdog'
                        slot['restart_deadline'] = time.monotonic() + stop_timeout_s
                        connection.send(dict(kind='STOP'))
            if not paused and not pipe.poll():
                for env_id, slot in slots.items():
                    if slot['mode'] == 'READY':
                        sequences[env_id] = sequences.get(env_id, 0) + 1
                        slot['token'] = sequences[env_id]; slot['mode'] = 'REQUESTED'
                        emit('RESERVE', env=env_id, token=slot['token'])
                batch = [(i, s) for i, s in slots.items() if s['mode'] == 'GRANTED']
                if batch:
                    with torch.inference_mode():
                        outputs = actor([slot['observation'] for _, slot in batch])
                        for (env_id, slot), output in zip(batch, outputs):
                            action = int(torch.multinomial(output.probs, 1, generator=generator))
                            slot['pipe'].send(dict(kind='STEP', token=slot['token'], action=action,
                                actor_version=actor_record['version']))
                            slot['mode'] = 'ACTIVE'; slot['since'] = time.monotonic()
            if barrier_requested and all(slot['pending'] is None for slot in slots.values()):
                emit('BARRIER', payload=snapshot())
                frozen = True; barrier_requested = False
            if stopping:
                if time.monotonic() - stop_started > stop_timeout_s:
                    for env_id, slot in list(slots.items()):
                        if not slot['closed'] and slot['pending'] is None:
                            recover(env_id, 'bounded stop timeout; unfinished action excluded')
                if all(slot['closed'] for slot in slots.values()) and all(slot['pending'] is None for slot in slots.values()):
                    emit('STOPPED', payload=snapshot())
                    return
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try: emit('FATAL', reason=f'{type(exc).__name__}: {exc}')
        except (EOFError, BrokenPipeError): pass
        raise
    finally:
        for slot in slots.values():
            if slot['process'].is_alive():
                try: slot['pipe'].send(dict(kind='STOP'))
                except (EOFError, BrokenPipeError, OSError): pass
        for slot in slots.values():
            slot['process'].join(3)
            if slot['process'].is_alive(): slot['process'].terminate(); slot['process'].join(3)
            if slot['process'].is_alive(): slot['process'].kill(); slot['process'].join(3)
            slot['pipe'].close(flush=False)
        pipe.close()

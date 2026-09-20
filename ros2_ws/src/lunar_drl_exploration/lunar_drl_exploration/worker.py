"""Torch-free spawn entry and responsive control for one synchronous environment."""
import importlib
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import asdict, is_dataclass


def numerical_threads(count):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(count)


def worker_main(pipe, config_record, env_id, *, domain_base=210,
                env_factory='lunar_drl_exploration.ros_env:RosExplorationEnv'):
    numerical_threads(config_record['worker_threads'])
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from .ipc import Duplex
    from .config import training_config_from_record
    from .replay import dumps_transport
    module, name = env_factory.split(':')
    factory = getattr(importlib.import_module(module), name)
    env = factory(training_config_from_record(config_record), env_id, domain_base=domain_base)
    pipe = Duplex(pipe)
    pipe.send(dict(kind='HELLO', env=env_id, pid=os.getpid(),
        torch_loaded='torch' in sys.modules, numeric_threads=os.environ['OMP_NUM_THREADS'],
        package_path=__file__, native_path=getattr(sys.modules.get('lunar_drl_terrain_native'), '__file__', None)))
    completed = queue.Queue(maxsize=1)
    commands = queue.Queue(maxsize=1)
    active = False
    pending = None
    pending_sent = 0.
    stopping = False
    token = None
    last_status = time.monotonic()

    def execute(command):
        try:
            if command['kind'] == 'RESET':
                obs, initial_state = env.reset(**command['spec'])
                result = dict(kind='RESET_DONE', observation=obs,
                    scenes=env.static_scenes, metadata=dict(env.episode_metadata,
                        initial_reference_coverage=env.reference.coverage_ratio(initial_state.observed)),
                    completed=bool(env.report.completed),exhausted=bool(env.report.exhausted))
            else:
                transition = env.step(command['action'], actor_version=command['actor_version'])
                execution = env.last_execution
                result = dict(kind='DONE', token=command['token'],
                    payload=dumps_transport(transition),
                    execution=asdict(execution) if is_dataclass(execution) else {},
                    progress=env.progress(),
                    reset=bool(transition.terminated or transition.truncated or
                        getattr(execution, 'geometry_failure', False)))
        except BaseException as exc:
            result = dict(kind='ABORTED' if type(exc).__name__ == 'TaskCanceled' else 'ERROR',
                token=command.get('token'), reason=f'{type(exc).__name__}: {exc}'[:2048])
        completed.put(result)

    def execution_loop():
        try:
            while True:
                command = commands.get()
                if command is None: return
                execute(command)
        finally:
            env.close()

    # Linux PDEATHSIG binds native children to the spawning thread. Keep this
    # single executor alive across resets, actions, ACK waits and final close.
    thread = threading.Thread(target=execution_loop, daemon=True)
    thread.start()

    def send(message):
        pipe.send(dict(message, env=env_id))

    def ready():
        send(dict(kind='READY', payload=dumps_transport(env.observation)))

    try:
        while True:
            if pipe.poll(.02):
                command = pipe.recv()
                kind = command['kind']
                if kind in ('CANCEL', 'STOP'):
                    stopping = kind == 'STOP' or stopping
                    env.cancel()
                elif kind == 'ACK':
                    if pending is not None and command['token'] == pending['token']:
                        reset = pending['reset']
                        pending = None
                        if not stopping:
                            send(dict(kind='NEED_RESET')) if reset else ready()
                elif kind in ('RESET', 'STEP'):
                    if active or pending is not None or stopping:
                        raise RuntimeError('command while active, unacknowledged or stopping')
                    token = command.get('token')
                    active = True
                    commands.put_nowait(command)
                else:
                    raise ValueError('unknown worker command ' + kind)
            if active and not completed.empty():
                result = completed.get_nowait()
                active = False
                if result['kind'] == 'RESET_DONE':
                    send(dict(kind='SCENE', payload=dumps_transport(result['scenes']), metadata=result['metadata']))
                    if result['completed']:
                        send(dict(kind='EPISODE_END', reason='EXHAUSTED', zero_decisions=True))
                        if not stopping: send(dict(kind='NEED_RESET'))
                    elif not stopping:
                        send(dict(kind='READY', payload=dumps_transport(result['observation'])))
                else:
                    if result['kind'] == 'DONE':
                        pending = result; pending_sent = time.monotonic()
                    send(result)
            if pending is not None and time.monotonic() - pending_sent >= 2:
                send(pending)
                pending_sent = time.monotonic()
            if stopping and not active and pending is None:
                commands.put_nowait(None)
                thread.join(5)
                if thread.is_alive(): raise RuntimeError('environment close exceeded bounded join')
                send(dict(kind='CLOSED'))
                return
            if time.monotonic() - last_status >= 1 and pending is None:
                # reset owns construction; only inspect fully established episodes.
                try: progress = env.progress()
                except (AttributeError, RuntimeError): progress = {'state': 'INITIALIZING'}
                send(dict(kind='STATUS', progress=progress))
                last_status = time.monotonic()
    except (EOFError, BrokenPipeError):
        env.cancel()
    finally:
        env.cancel()
        if thread.is_alive():
            try: commands.put_nowait(None)
            except queue.Full: pass
            thread.join(5)
        pipe.close()
        # Parent handles bounded termination if an infrastructure initialization
        # cannot join. Never call close concurrently with active reset/step.

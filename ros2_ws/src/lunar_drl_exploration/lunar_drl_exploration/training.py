"""Single learner owner of replay, exact update credit and continuation state."""


class AdmissionLedger:
    def __init__(self, replay, schedule):
        self.replay, self.schedule = replay, schedule
        self.active_scenes = {}
        self.last_admitted = {}

    def scene(self, env_id, scenes):
        if len(scenes) != 1:
            raise ValueError('exactly one active scene per environment required')
        self.active_scenes[env_id] = scenes

    def admit(self, env_id, token, transition):
        previous = self.last_admitted.get(env_id, -1)
        if token <= previous:
            return False
        self.replay.add(transition, self.active_scenes[env_id])
        self.schedule.collected(reserved=True)
        self.last_admitted[env_id] = token
        return True


class Curriculum:
    """Admission-count schedule: early small; middle 25/75; late 15/25/60.

    Configurable initial stage boundaries are 20k and 60k valid transitions.
    Distribution changes only when an environment requests a new episode.
    """
    def __init__(self, config, rng, state=None, *, probe_extent=None, probe_budget=None):
        self.config, self.rng = config, rng
        self.state = dict(state or {'episodes_issued': 0})
        self.probe_extent, self.probe_budget = probe_extent, probe_budget

    def next(self, env_id, transitions):
        stage = sum(transitions >= boundary for boundary in self.config.curriculum_transition_boundaries)
        probabilities = self.config.curriculum_mixtures[stage]
        size = int(self.rng.choice(3, p=probabilities))
        low, high = self.config.curriculum_extents_m[size]
        extent = self.probe_extent if self.probe_extent is not None else float(self.rng.uniform(low, high))
        # Separate disjoint evaluation seed namespace; a monotonic issued counter
        # avoids accidental reuse on resume while RNG restores terrain selection.
        serial = self.state['episodes_issued']
        self.state['episodes_issued'] = serial + 1
        seed = ((int(self.config.seed) + 1) << 32) + serial
        return dict(seed=seed, family='moon' if env_id % 2 == 0 else 'cave', extent=extent,
            episode_budget=self.probe_budget or self.config.curriculum_budgets[size])


def run_training(config, *, resume=False, device='cuda', max_transitions=None,
                 domain_base=210, probe_extent=None, probe_budget=None,
                 env_factory='lunar_drl_exploration.ros_env:RosExplorationEnv'):
    """Run finite or continuous training; max_transitions counts this-run admissions.

    The only replay/optimizer owner is this process. Periodic saves ACK-drain at
    a completed update boundary; unfinished goals are retained live until resume
    of dispatch. A restored run deliberately creates new environment episodes.
    """
    import multiprocessing as mp
    import os
    import pickle
    from pathlib import Path
    import random
    import signal
    import subprocess
    import sys
    import time
    from .worker import numerical_threads
    numerical_threads(config.learner_threads)
    import numpy as np
    import torch
    from .checkpoint import CheckpointManager, TrainingState, UpdateBoundary
    from .collector import collector_main
    from .config import config_record
    from .metrics import MetricsWriter, ResourceMonitor, ResourceLimitError
    from .replay import ReplayBuffer, loads_transport
    from .sac import SACLearner
    from .schedule import UpdateSchedule
    if config.seed < 0: raise ValueError('training seed must be nonnegative')
    if max_transitions is not None and max_transitions <= 0:
        raise ValueError('max_transitions must be positive')
    if not 0 <= domain_base <= 233 - config.environments:
        raise ValueError('isolated worker DDS domains must be within 0..232')
    if config.model.width != 128 or config.model.heads != 8 or config.model.layers != 6:
        raise ValueError('training requires full 128/8/6 graph model')
    torch.set_num_threads(config.learner_threads)
    random.seed(config.seed); np.random.seed(config.seed % 2**32); torch.manual_seed(config.seed)
    manager = CheckpointManager(config.output_dir, config.save_interval_s,
        total_max_bytes=config.total_output_max_bytes, snapshot_max_bytes=config.snapshot_max_bytes)
    if not resume and (Path(config.output_dir) / 'resume.pt').exists():
        raise ValueError('resume.pt already exists; use --resume or a new output directory')
    learner = SACLearner(config.model, config.learning, device=device)
    replay = ReplayBuffer(config.replay_max_bytes)
    schedule = UpdateSchedule(config.warmup, config.update_ratio, config.max_update_credit)
    replay_rng, curriculum_rng = np.random.default_rng(config.seed), np.random.default_rng(config.seed + 1)
    curriculum_state = None
    collector_state = dict(actor_record=learner.actor_state(), actor_version=0,
        policy_rng=torch.Generator().manual_seed(config.seed + 2).get_state(), sequences={})
    counters = dict(transitions=0, aborted=0, recoveries=0)
    if resume:
        restored = manager.load(config).restore(learner, config)
        replay, schedule = restored.replay, restored.schedule
        replay_rng, curriculum_rng = restored.replay_rng, restored.curriculum_rng
        curriculum_state, collector_state, counters = restored.curriculum, restored.collector_state, restored.counters
    if collector_state['actor_record']['version'] != collector_state['actor_version']:
        raise ValueError('saved collector weights/publication identity disagree')
    curriculum = Curriculum(config, curriculum_rng, curriculum_state,
        probe_extent=probe_extent, probe_budget=probe_budget)
    ledger = AdmissionLedger(replay, schedule)
    ledger.last_admitted = dict(counters.get('last_admitted', {}))
    boundary = UpdateBoundary()
    metrics = MetricsWriter(Path(config.output_dir) / 'metrics.jsonl', config.metrics_max_bytes,
        total_max_bytes=config.total_output_max_bytes)
    context = mp.get_context('spawn')
    connection, child = context.Pipe()
    process = context.Process(target=collector_main,
        args=(child, config_record(config), pickle.dumps(collector_state, protocol=5)),
        kwargs=dict(domain_base=domain_base, env_factory=env_factory), name='drl-cpu-collector')
    previous_signals = {sig: signal.signal(sig, manager.request_sigint) for sig in (signal.SIGINT, signal.SIGTERM)}
    baseline = schedule.transitions
    started = time.monotonic()
    pending_requests = {}
    reservations = {}
    env_status = {}
    env_owners = {}
    observations = dict(learner=dict(pid=os.getpid(), device=str(learner.device),
        numeric_threads=torch.get_num_threads(), package_path=__file__,
        torch_path=torch.__file__, cuda_initialized=torch.cuda.is_initialized()), workers={},
        probe=dict(extent_m=probe_extent, budget=probe_budget, max_new_transitions=max_transitions),
        resume=resume)
    # Bounded recent episode identities suffice for run auditing; counters remain cumulative.
    episode_ids = []
    barrier_pending = False
    stopping = False
    stopped = False
    failure = None
    stop_reason = None
    last_metrics = started
    last_update = {}
    observed_publication = collector_state['actor_version']
    try:
        code_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): code_revision = 'unavailable'

    def capture():
        counters['last_admitted'] = dict(ledger.last_admitted)
        return TrainingState.capture(learner=learner, config=config, replay=replay,
            schedule=schedule, boundary=boundary, replay_rng=replay_rng, curriculum_rng=curriculum_rng,
            curriculum=curriculum.state, collector_state=collector_state, counters=counters)

    def save():
        path = manager.save(capture())
        metrics.append(dict(event='checkpoint', transitions=schedule.transitions,
            updates=schedule.updates, actor_version=collector_state['actor_version'],
            unfinished_reservations=schedule.inflight, path=str(path)))

    def stop(reason):
        nonlocal stopping, stop_reason
        if stop_reason is None: stop_reason = reason
        manager.request_save(reason, stop=True)
        if not stopping:
            stopping = True
            pending_requests.clear()
            connection.send(dict(kind='STOP'))

    def record_run():
        manager.write_run(config, owned_pids=[os.getpid(), process.pid] +
            [pid for pids in env_owners.values() for pid in pids], code_revision=code_revision, extra=observations)

    def status_record():
        elapsed = max(time.monotonic() - started, 1e-9)
        return dict(event='progress', wall_s=elapsed, transitions=schedule.transitions,
            new_transitions=schedule.transitions-baseline, updates=schedule.updates,
            transitions_per_s=(schedule.transitions-baseline)/elapsed,
            updates_per_s=(schedule.updates-start_updates)/elapsed, credit=float(schedule.credit),
            inflight=schedule.inflight, actor_version=observed_publication,
            replay_count=len(replay), replay_bytes=replay.bytes_used,
            save_in_s=max(0., config.save_interval_s-(time.monotonic()-manager.last_save)),
            environments=env_status, learning=last_update,
            gpu_memory=(dict(allocated_bytes=torch.cuda.memory_allocated(learner.device),
                reserved_bytes=torch.cuda.memory_reserved(learner.device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(learner.device))
                if learner.device.type == 'cuda' else None))

    start_updates = schedule.updates
    try:
        # An initial good checkpoint also protects a first-update infrastructure failure.
        if not resume: save()
        process.start(); child.close()
        record_run()
        while not stopped:
            if connection.poll(.01):
                message = connection.recv(); kind = message['kind']; env_id = message.get('env')
                if kind == 'NEED_RESET':
                    if not stopping:
                        spec = curriculum.next(env_id, schedule.transitions)
                        connection.send(dict(kind='RESET', env=env_id, spec=spec))
                elif kind == 'SCENE':
                    scenes = loads_transport(message['payload'])
                    ledger.scene(env_id, scenes)
                    metadata = message['metadata']
                    env_status[env_id] = dict(metadata, state='READY')
                    episode_ids.append(metadata['episode_id'])
                    del episode_ids[:-config.environments * 2]
                    observations['workers'].setdefault(str(env_id), {})['scene'] = dict(metadata,
                        truth_nodes=sum(len(scene.positions) for scene in scenes.values()),
                        truth_edges=sum(len(scene.edges) for scene in scenes.values()),
                        reference_cells=sum(len(scene.reference_indices) for scene in scenes.values()))
                elif kind == 'RESERVE':
                    if not stopping: pending_requests[env_id] = message['token']
                elif kind == 'DONE':
                    transition = loads_transport(message['payload'])
                    is_new = message['token'] > ledger.last_admitted.get(env_id, -1)
                    if is_new and reservations.get(env_id) != message['token']:
                        raise RuntimeError('completion lacks its learner-owned dispatch reservation')
                    admitted = ledger.admit(env_id, message['token'], transition)
                    if admitted:
                        reservations.pop(env_id)
                        counters['transitions'] = schedule.transitions
                    # Order is transactional replay admission -> collected -> ACK.
                    connection.send(dict(kind='ACK', env=env_id, token=message['token']))
                    status = env_status.setdefault(env_id, {})
                    status.update(message.get('progress', {}))
                    execution = message.get('execution', {})
                    status.update({key: value for key, value in execution.items()
                        if key not in ('distance_m', 'turn_rad', 'simulation_s', 'wall_s')})
                    status['last_execution'] = execution
                    scene = ledger.active_scenes[env_id][transition.next_privileged.scene_id]
                    visible_count = int(np.unpackbits(transition.next_privileged.observed & scene.packed_reference).sum())
                    status['reference_coverage'] = visible_count / len(scene.reference_indices) if len(scene.reference_indices) else 0.
                    status.update(new_area_m2=transition.parts.new_area_m2,
                        reward=transition.reward, terminated=transition.terminated, truncated=transition.truncated,
                        actor_version=transition.actor_version, observed_nodes=len(transition.observation.node_ids),
                        observed_edges=len(transition.observation.edges))
                    wall = message.get('execution', {}).get('wall_s', 0)
                    status['actual_rtf'] = message.get('execution', {}).get('simulation_s', 0) / wall if wall else None
                    if admitted:
                        metrics.append(dict(status, event='transition', env=env_id, token=message['token'],
                            transitions=schedule.transitions, episode_id=transition.episode_id))
                    if max_transitions is not None and schedule.transitions-baseline >= max_transitions:
                        stop('max new transitions reached; draining finished messages')
                elif kind == 'ABORTED':
                    if reservations.get(env_id) == message['token']:
                        schedule.cancel_dispatch(); reservations.pop(env_id)
                        counters['aborted'] = counters.get('aborted', 0) + 1
                    metrics.append(dict(message, event='unfinished_excluded'))
                elif kind in ('BARRIER', 'STOPPED'):
                    collector_state = pickle.loads(message['payload'])
                    if counters['transitions'] != schedule.transitions:
                        raise RuntimeError('admission counter differs at barrier')
                    save()
                    barrier_pending = False
                    if kind == 'STOPPED':
                        if schedule.inflight or reservations: raise RuntimeError('stop left unaccounted reservations')
                        stopped = True
                    elif not stopping:
                        connection.send(dict(kind='RESUME'))
                elif kind == 'PUBLISHED': observed_publication = message['version']
                elif kind == 'FATAL':
                    failure = message['reason']
                    print(f'Collector fatal: {failure}', file=sys.stderr, flush=True)
                    stop(failure)
                elif kind == 'RECOVERY':
                    counters['recoveries'] = counters.get('recoveries', 0) + 1
                    print(f"E{env_id} infrastructure recovery attempt={message['attempt']}: {message['reason']}", flush=True)
                    metrics.append(dict(message, event='recovery'))
                elif kind == 'OWNERSHIP':
                    env_owners[env_id] = ([message['worker_pid']] if message['worker_pid'] else []) + message['child_pids']
                elif kind == 'WORKER_HELLO':
                    observations['workers'].setdefault(str(env_id), {}).update(message)
                    if message['torch_loaded']: raise RuntimeError('Torch imported in ROS worker')
                elif kind == 'COLLECTOR_HELLO': observations['collector'] = message
                elif kind == 'STATUS':
                    progress = message['progress']; env_status.setdefault(env_id, {}).update(progress)
                    worker_pid = observations['workers'].get(str(env_id), {}).get('pid')
                    env_owners[env_id] = ([worker_pid] if worker_pid else []) + list(progress.get('owned_pids', []))
                elif kind == 'EPISODE_END': metrics.append(dict(message, event='episode_end'))
                else: raise RuntimeError('unexpected learner message ' + kind)
            if not process.is_alive() and not stopped and not connection.poll():
                raise RuntimeError(f'collector exited before acknowledged stop: {process.exitcode}')
            if not stopping and not barrier_pending:
                for env_id, token in list(pending_requests.items()):
                    if schedule.can_dispatch():
                        if env_id in reservations: raise RuntimeError('second reservation for environment')
                        schedule.reserve_dispatch(); reservations[env_id] = token
                        connection.send(dict(kind='GRANT', env=env_id, token=token))
                        del pending_requests[env_id]
            # Exactly one complete update; poll completions again before the next.
            if not stopping and not barrier_pending and schedule.can_update:
                with boundary.update():
                    before_update = time.monotonic()
                    last_update = learner.update(replay.sample(config.learning.batch_size, replay_rng), replay.scenes)
                    schedule.updated()
                    last_update['wall_s'] = time.monotonic() - before_update
                if schedule.updates % config.actor_publish_updates == 0:
                    connection.send(dict(kind='PUBLISH', payload=pickle.dumps(learner.actor_state(), protocol=5)))
            if manager.stop_requested and not stopping: stop(manager.requested_reason or 'stop requested')
            if manager.save_requested and not stopping and not barrier_pending:
                connection.send(dict(kind='BARRIER')); barrier_pending = True
            if not stopped and time.monotonic() - last_metrics >= 2:
                record = status_record()
                owned = [os.getpid(), process.pid] + [pid for pids in env_owners.values() for pid in pids]
                monitor = ResourceMonitor(owned, system_reserve_bytes=config.system_reserve_bytes,
                    owned_pss_limit_bytes=config.owned_pss_limit_bytes)
                try:
                    resource = monitor.observe(config.output_dir); record['resources'] = resource
                    monitor.check(resource)
                except (FileNotFoundError, ProcessLookupError) as exc:
                    # Lifecycle messages may be queued behind a completed update.
                    # Report unknown PID data; never substitute zero PSS.
                    record['resource_unavailable'] = str(exc)
                except ResourceLimitError as exc: stop(str(exc))
                metrics.append(record)
                target = '' if max_transitions is None else f'/{max_transitions}'
                bar = '' if max_transitions is None else '[' + '=' * min(20, int(20 * (schedule.transitions-baseline) / max_transitions)) + ' ' * max(0, 20-int(20 * (schedule.transitions-baseline) / max_transitions)) + '] '
                print(f"DRL {bar}transitions={record['new_transitions']}{target} total={schedule.transitions} "
                    f"updates={schedule.updates} credit={float(schedule.credit):.2f} actor={observed_publication} "
                    f"sample/s={record['transitions_per_s']:.2f} update/s={record['updates_per_s']:.2f} "
                    f"replay={replay.bytes_used/1024**2:.1f}MiB save={record['save_in_s']:.0f}s", flush=True)
                for env_id, status in sorted(env_status.items()):
                    area, reference = status.get('known_area_m2'), status.get('reference_area_m2')
                    ratio = status.get('reference_coverage', status.get('initial_reference_coverage'))
                    coverage = f'{ratio:.1%}' if ratio is not None else 'unavailable'
                    print(f"  E{env_id} {status.get('family','?')} {status.get('extent_m',0):.0f}m "
                        f"decisions={status.get('steps',0)}/{status.get('budget','?')} coverage={coverage} "
                        f"area={area} gain={status.get('new_area_m2')} distance={status.get('distance_m',0):.2f} "
                        f"RTF={status.get('actual_rtf')} reason={status.get('reason_code',status.get('state','?'))}", flush=True)
                last_metrics = time.monotonic()
        process.join(10)
        observations['final'] = status_record()
        observations['final']['overshoot'] = max(0, schedule.transitions-baseline-(max_transitions or schedule.transitions))
        observations['final']['stop_reason'] = failure or stop_reason or 'orderly stop'
        observations['final']['owned_children_closed'] = not process.is_alive()
        record_run()
        if failure: raise RuntimeError(failure)
        return dict(transitions=schedule.transitions, new_transitions=schedule.transitions-baseline,
            updates=schedule.updates, episode_ids=episode_ids, episodes_issued=curriculum.state['episodes_issued'],
            stop_reason=stop_reason)
    except BaseException as exc:
        # Never overwrite the last good checkpoint after a partly applied update
        # or an unadmitted resource-damaged completion. The reason is explicit.
        print(f'DRL stopped with error; last completed checkpoint preserved: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        # A poisoned optimizer boundary forbids saving, but not receiving/ACKing
        # already finished valid work. Drain before joining the collector so it
        # can cancel active goals and close workers rather than wait on our ACK.
        if process.pid is not None and process.is_alive():
            try:
                connection.send(dict(kind='STOP'))
                deadline = time.monotonic() + 12
                while time.monotonic() < deadline:
                    if not connection.poll(.05):
                        if not process.is_alive(): break
                        continue
                    message = connection.recv(); kind = message['kind']; env_id = message.get('env')
                    if kind == 'SCENE': ledger.scene(env_id, loads_transport(message['payload']))
                    elif kind == 'DONE':
                        ledger.admit(env_id, message['token'], loads_transport(message['payload']))
                        reservations.pop(env_id, None)
                        connection.send(dict(kind='ACK', env=env_id, token=message['token']))
                    elif kind == 'ABORTED' and reservations.get(env_id) == message['token']:
                        schedule.cancel_dispatch(); reservations.pop(env_id)
                    elif kind == 'STOPPED': break
                process.join(2)
            except (EOFError, OSError, ResourceLimitError, ValueError) as drain_error:
                print(f'Fatal drain incomplete; unadmitted completion/owned reservations retained until shutdown: {drain_error}',
                    file=sys.stderr, flush=True)
        print(f'Fatal recovery uses last good resume; current admitted={schedule.transitions}, '
            f'updates={schedule.updates}, remaining reservations={schedule.inflight}', file=sys.stderr, flush=True)
        raise
    finally:
        for sig, handler in previous_signals.items(): signal.signal(sig, handler)
        if process.pid is not None:
            if process.is_alive():
                try: connection.send(dict(kind='STOP'))
                except (EOFError, BrokenPipeError): pass
                process.join(12)
            if process.is_alive(): process.terminate(); process.join(5)
            if process.is_alive(): process.kill(); process.join(5)
        child.close(); connection.close()

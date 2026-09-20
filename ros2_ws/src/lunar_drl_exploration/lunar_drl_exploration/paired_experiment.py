"""Serial, resumable C=0/C=10 experiment; children own GPU and ROS lifetimes.

Run with ``python -m lunar_drl_exploration.paired_experiment --help`` in the
isolated Jazzy environment. The 3k screen is an engineering gate, never selection.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ARMS = {'raw': 0., 'bounded10': 10.}
EVALUATION_SEEDS = (2026091901, 2026091902)


def stages(seeds, screen, total):
    result = []
    for index, seed in enumerate(seeds):
        for phase, target in ([('screen', screen), ('final', total)] if index == 0 else [('final', total)]):
            for arm in ARMS:
                result.append(dict(id=f'{seed}-{arm}-{phase}', seed=seed, arm=arm,
                    phase=phase, target=target))
    return result


def pending_stages(plan, records):
    return [stage for stage in plan if records.get(stage['id'], {}).get('status') != 'complete']


def validate_child_result(returncode, record, target):
    if returncode != 0:
        raise RuntimeError(f'child failed with exit code {returncode}: {record.get("error", "no result")}')
    error = record.get('error') or record.get('final', {}).get('error')
    if error or record.get('status') != 'complete':
        raise RuntimeError(str(error or f'child status {record.get("status", "missing")}'))
    if record.get('transitions', -1) < target:
        raise RuntimeError('child stopped before requested transition budget')
    return record


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    if len(payload.encode()) > 8 * 1024**2:
        raise ValueError('paired report exceeds 8 MiB bound')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else {}


def experiment_config(seed, arm, output):
    from .config import TrainingConfig
    config = TrainingConfig()
    return replace(config, seed=seed, paired_scene_sequence=True,
        model=replace(config.model, actor_score_bound=ARMS[arm]),
        environments=8, curriculum_mixtures=((1., 0., 0.),) * 3,
        curriculum_extents_m=((40, 80), (80, 150), (100, 300)),
        curriculum_budgets=(512, 2048, 8192), output_dir=str(output))


from .evaluation import GeometryMetrics


def evaluate_stage(config, actor, phase, domain_base):
    """Keep every fixed case, including reset failures and physical collisions."""
    from .evaluation import EpisodeMetrics, summarize
    from .runtime import ActorPolicy
    from .ros_env import RosExplorationEnv
    policy = ActorPolicy.load(actor)
    rows = []
    for family in ('moon', 'cave'):
        for extent in ((40.,) if phase == 'screen' else (40., 80.)):
            for seed in (EVALUATION_SEEDS[:1] if phase == 'screen' else EVALUATION_SEEDS):
                metric = EpisodeMetrics(family, extent, seed, config.success_coverage)
                geometry = None
                env = None
                try:
                    env = RosExplorationEnv(config, 0, domain_base=domain_base)
                    obs, state = env.reset(seed, family, extent, episode_budget=512)
                    geometry = GeometryMetrics((env.plant.pose.x, env.plant.pose.y), config.goal_position_tolerance_m)
                    metric.observe(env.reference.coverage_ratio(state.observed),
                        env.progress()['distance_m'], 0., 'INITIAL', env.terminated, False,
                        exhausted=env.report.exhausted,
                        covered_area_m2=env.reference.covered_area(state.observed),
                        coverable_area_m2=env.reference.area_m2)
                    while not metric.terminated and not metric.truncated and not metric.collisions:
                        transition = env.step(policy(obs))
                        geometry.observe((env.plant.pose.x, env.plant.pose.y), transition.parts.new_area_m2)
                        metric.observe(env.reference.coverage_ratio(transition.next_privileged.observed),
                            env.progress()['distance_m'], transition.parts.new_area_m2,
                            env.last_execution.reason_code, transition.terminated, transition.truncated,
                            exhausted=env.report.exhausted,
                            covered_area_m2=env.reference.covered_area(transition.next_privileged.observed),
                            coverable_area_m2=env.reference.area_m2)
                        obs = transition.next_observation
                except Exception as exc:
                    metric.error = f'{type(exc).__name__}: {exc}'
                finally:
                    if env is not None:
                        try: env.close()
                        except Exception as exc:
                            metric.error = (metric.error or '') + f' close: {type(exc).__name__}: {exc}'
                row = metric.record()
                row.update(geometry.record() if geometry else dict(geometric_metrics_unavailable=True))
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    return dict(policy='frozen_joint_argmax', budget=512,
        cases=rows, groups=summarize(rows),
        infrastructure_failures=sum(row['error'] is not None for row in rows))


def child_stage(args):
    from .worker import numerical_threads
    numerical_threads(4)
    import torch
    from .config import config_record
    from .training import run_training
    from .evaluation import export_actor
    stage = json.loads(args.child_stage)
    directory = args.output_dir / str(stage['seed']) / stage['arm']
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'child-result.json'
    result = dict(status='running', stage=stage)
    atomic_json(result_path, result)
    config = experiment_config(stage['seed'], stage['arm'], directory)
    atomic_json(directory / 'config.json', config_record(config))
    checkpoint = directory / 'resume.pt'
    actor_path = directory / '.paired-evaluation-actor.pt'
    started = time.monotonic()
    try:
        before = 0
        updates = 0
        if checkpoint.exists():
            record = torch.load(checkpoint, map_location='cpu', weights_only=False)
            from .checkpoint import TrainingState
            TrainingState(record).validate(config)
            before, updates = record['schedule']['transitions'], record['schedule']['updates']
            del record
        result['before_transitions'] = before
        if before < stage['target']:
            trained = run_training(config, resume=checkpoint.exists(), device=args.device,
                max_transitions=stage['target'] - before, domain_base=args.domain_base)
            result.update(trained)
            reason = trained.get('stop_reason') or ''
            if not reason.startswith('max new transitions reached'):
                raise RuntimeError(f'training stopped unexpectedly: {reason}')
        else:
            result.update(transitions=before, new_transitions=0, updates=updates)
        run = read_json(directory / 'run.json')
        final = run.get('observations', {}).get('final', {})
        if final.get('error'):
            raise RuntimeError(str(final['error']))
        result['overshoot'] = result['transitions'] - stage['target']
        result['training_final'] = final
        result['initial_model_checksum'] = run.get('observations', {}).get('initial_model_checksum')
        if not result['initial_model_checksum']:
            raise RuntimeError('training omitted paired initial parameter checksum')
        export_actor(checkpoint, actor_path)
        evaluation = evaluate_stage(config, actor_path, stage['phase'], args.domain_base)
        atomic_json(directory / f"{stage['phase']}-evaluation.json", evaluation)
        result['evaluation'] = evaluation
        if all(row['error'] is not None and row['decisions'] == 0 for row in evaluation['cases']):
            raise RuntimeError('all fixed evaluation cases failed before any decision; all cases retained')
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
            error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['wall_s'] = time.monotonic() - started
        atomic_json(result_path, result)
        # This exact ephemeral export is owned by this invocation, never resume.pt.
        if actor_path.exists(): actor_path.unlink()
    return result


def write_report(output, state):
    lines = ['# Bounded Actor paired experiment', '',
        'Local Jazzy experiment only. Humble/Orin/vehicle and convergence: NOT_RUN.', '',
        'The 3000-transition screen is an engineering gate, not winner selection. ',
        'Training maps use per-slot/per-episode identities; evaluation maps use a disjoint seed namespace.',
        'Checkpoint continuation starts new episodes and records actual admissions and updates.', '',
        '| Stage | Status | Transitions | Updates | Overshoot | Error |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for stage in state['plan']:
        row = state['records'].get(stage['id'], {})
        lines.append('| ' + ' | '.join(str(v).replace('|', '/').replace('\n', ' ') for v in (
            stage['id'], row.get('status', 'NOT_RUN'), row.get('transitions', ''),
            row.get('updates', ''), row.get('overshoot', ''), row.get('error', ''))) + ' |')
    lines.extend(['', '## Fixed-map evaluation', '',
        '| Stage | Cases | Mean coverage | 80% rate | 99% rate | Exhaustion rate | Mean path m | Zero-gain ratio | Two-point loop steps | Collisions | Errors |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for stage in state['plan']:
        record = state['records'].get(stage['id'], {})
        rows = record.get('evaluation', {}).get('cases', [])
        if not rows: continue
        count = len(rows)
        mean = lambda key: sum(row.get(key, 0) for row in rows) / count
        values = [stage['id'], count, f"{mean('final_coverage'):.4f}", f"{mean('reached_80'):.3f}",
            f"{mean('reached_99'):.3f}", f"{mean('exhausted'):.3f}", f"{mean('distance_m'):.2f}",
            f"{mean('zero_gain_ratio'):.3f}", sum(row.get('zero_gain_two_point_loop_steps', 0) for row in rows),
            sum(row.get('collisions', 0) for row in rows), sum(row.get('error') is not None for row in rows)]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines.extend(['', '## Initial parameter identity', ''])
    for seed in dict.fromkeys(stage['seed'] for stage in state['plan']):
        checksums = {stage['arm']: state['records'].get(stage['id'], {}).get('initial_model_checksum')
            for stage in state['plan'] if stage['seed'] == seed and
            state['records'].get(stage['id'], {}).get('initial_model_checksum')}
        lines.append(f"- {seed}: " + (', '.join(f'{arm} `{value}`' for arm, value in checksums.items()) or 'NOT_RUN'))
    lines.extend(['', 'Per-stage evaluation JSON retains every fixed case: coverage, 80/99% path lengths,',
        'exhaustion, failures, zero-gain runs and 0.5 m geometric reversals/revisits.',
        'Unreached thresholds stay null; failed cases remain in denominators.',
        'Evaluation stops the experiment only if every fixed case errors before any decision; individual failures remain in the comparison.',
        'Reverse-edge/revisit counts alone are geometric indicators, not proof of a harmful loop.', '',
        f"Experiment status: {state.get('status', 'pending')}", ''])
    temporary = output / 'paired-report.md.tmp'
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    os.replace(temporary, output / 'paired-report.md')


def run_experiment(args):
    import fcntl
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / '.runner.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another paired runner owns this output directory') from exc
        return _run_experiment_locked(args)


def _run_experiment_locked(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = stages(args.seeds, args.screen_transitions, args.total_transitions)
    state_path = output / 'paired-state.json'
    state = read_json(state_path)
    if not state and (any(output.glob('*/raw/resume.pt')) or any(output.glob('*/bounded10/resume.pt'))):
        raise ValueError('untracked checkpoints already exist; refusing to adopt another experiment')
    if state and not args.resume:
        raise ValueError('experiment already exists; use --resume')
    if state and state['plan'] != plan:
        raise ValueError('resume requires identical seed and budget plan')
    if not state:
        state = dict(schema='paired_actor_v1', plan=plan, records={}, status='pending')
    active = None
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        # Only learner receives the signal, then drains its own workers orderly.
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGINT)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for stage in pending_stages(plan, state['records']):
            if interrupted: break
            state['status'] = 'running'
            state['runner_pid'] = os.getpid()
            state['records'][stage['id']] = dict(status='running')
            atomic_json(state_path, state); write_report(output, state)
            command = [sys.executable, '-m', 'lunar_drl_exploration.paired_experiment', '--output-dir', str(output),
                '--device', args.device, '--domain-base', str(args.domain_base),
                '--child-stage', json.dumps(stage)]
            # Fresh session prevents terminal Ctrl-C from bypassing orderly forwarding.
            directory = output / str(stage['seed']) / stage['arm']
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Starting {stage['id']} target={stage['target']} log={directory / 'current-stage.log'}", flush=True)
            active = subprocess.Popen(command, start_new_session=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            with (directory / 'current-stage.log').open('wb') as log:
                while True:
                    block = active.stdout.read1(65536)
                    if not block: break
                    if log.tell() + len(block) > 8 * 1024**2:
                        log.seek(0); log.truncate()
                    log.write(block); log.flush()
            active.stdout.close()
            code = active.wait()
            active = None
            record = read_json(output / str(stage['seed']) / stage['arm'] / 'child-result.json')
            if record.get('stage') != stage:
                record = dict(status='failed', error='child result missing or belongs to another stage')
            state['records'][stage['id']] = record
            if interrupted:
                record['status'] = 'interrupted'
                state['status'] = 'interrupted'
                break
            try:
                validate_child_result(code, record, stage['target'])
                for earlier in state['records'].values():
                    if earlier.get('stage', {}).get('seed') == stage['seed']:
                        left, right = earlier.get('initial_model_checksum'), record.get('initial_model_checksum')
                        if left is not None and right is not None and left != right:
                            raise RuntimeError('paired initial parameter checksums differ')
            except RuntimeError as exc:
                record.update(status='failed', error=str(exc))
                state['status'] = 'failed'
                raise
            finally:
                atomic_json(state_path, state); write_report(output, state)
        else:
            state['status'] = 'complete'
        if interrupted: state['status'] = 'interrupted'
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)
        atomic_json(state_path, state); write_report(output, state)
    return 130 if interrupted else 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output-dir', type=Path, default=Path('training-output/actor-bounded-20260919'))
    result.add_argument('--seeds', nargs='+', type=int, default=[20260919, 20260920, 20260921])
    result.add_argument('--screen-transitions', type=int, default=3000)
    result.add_argument('--total-transitions', type=int, default=10000)
    result.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    result.add_argument('--domain-base', type=int, default=210)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--child-stage', help=argparse.SUPPRESS)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.child_stage:
        child_stage(args)
        return 0
    if not 0 < args.screen_transitions < args.total_transitions:
        raise ValueError('require 0 < screen transitions < total transitions')
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError('require unique nonnegative paired seeds')
    return run_experiment(args)


if __name__ == '__main__':
    sys.exit(main())

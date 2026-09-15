"""Frozen evaluation/export and deployment entrypoints with lazy dependencies."""
from dataclasses import asdict
import json
from pathlib import Path


class EpisodeMetrics:
    def __init__(self, family, extent, seed):
        self.family, self.extent, self.seed = family, extent, seed
        self.coverage = self.distance = 0.
        self.covered_area_m2 = self.coverable_area_m2 = None
        self.path80 = self.path99 = None
        self.exhausted = self.truncated = False
        self.steps = self.zero_gain = self.failures = self.collisions = 0
        self.reasons = {}
        self.error = None

    def observe(self, coverage, distance, new_area, reason, terminated, truncated,
                *, covered_area_m2=None, coverable_area_m2=None):
        self.coverage, self.distance = float(coverage), float(distance)
        self.covered_area_m2 = None if covered_area_m2 is None else float(covered_area_m2)
        self.coverable_area_m2 = None if coverable_area_m2 is None else float(coverable_area_m2)
        self.exhausted, self.truncated = bool(terminated), bool(truncated)
        if coverage >= .8 and self.path80 is None: self.path80 = float(distance)
        if coverage >= .99 and self.path99 is None: self.path99 = float(distance)
        if reason != 'INITIAL':
            self.steps += 1; self.zero_gain += new_area <= 0
            self.failures += reason != 'GOAL_REACHED'
            self.collisions += reason == 'COLLISION'
            self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def record(self):
        return dict(family=self.family, extent_m=self.extent, seed=self.seed,
            covered_area_m2=self.covered_area_m2, coverable_area_m2=self.coverable_area_m2,
            final_coverage=self.coverage, reached_80=self.path80 is not None,
            reached_99=self.path99 is not None, exhausted=self.exhausted,
            exhaustion_coverage=self.coverage if self.exhausted else None,
            path_to_80_m=self.path80, path_to_99_m=self.path99,
            path_80_to_99_m=self.path99-self.path80 if self.path99 is not None else None,
            distance_m=self.distance, decisions=self.steps, zero_gain_ratio=self.zero_gain/self.steps if self.steps else 0.,
            truncated=self.truncated, navigation_failures=self.failures, collisions=self.collisions,
            reasons=self.reasons, error=self.error)


def summarize(rows):
    groups = {}
    for row in rows: groups.setdefault((row['family'], row['extent_m']), []).append(row)
    result = []
    for (family, extent), cases in sorted(groups.items()):
        count = len(cases)
        result.append(dict(family=family, extent_m=extent, cases=count,
            rate_80=sum(row['reached_80'] for row in cases)/count,
            rate_99=sum(row['reached_99'] for row in cases)/count,
            exhaustion_rate=sum(row['exhausted'] for row in cases)/count,
            mean_final_coverage=sum(row['final_coverage'] for row in cases)/count,
            infrastructure_failures=sum(row['error'] is not None for row in cases)))
    return result


def evaluate(config, actor_path, *, seeds, families, extents, budget, output=None, domain_base=220):
    import torch
    from .runtime import ActorPolicy
    from .ros_env import RosExplorationEnv
    if budget <= 0 or any(extent <= 0 for extent in extents): raise ValueError('positive evaluation extent/budget required')
    if any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('evaluation seeds must be in [0,2**32); training uses a separate higher namespace')
    torch.set_num_threads(config.collector_threads)
    policy = ActorPolicy.load(actor_path)
    env = RosExplorationEnv(config, 0, domain_base=domain_base)
    rows = []
    try:
        for family in families:
            for extent in extents:
                for seed in seeds:
                    metric = EpisodeMetrics(family, extent, seed)
                    try:
                        obs, state = env.reset(seed, family, extent, episode_budget=budget)
                        progress = env.progress()
                        metric.observe(env.reference.coverage_ratio(state.observed),
                            progress['distance_m'], 0., 'INITIAL', env.report.exhausted, False,
                            covered_area_m2=env.reference.covered_area(state.observed),
                            coverable_area_m2=env.reference.area_m2)
                        while not metric.exhausted and not metric.truncated and not metric.collisions:
                            transition = env.step(policy(obs))
                            execution = env.last_execution
                            metric.observe(env.reference.coverage_ratio(transition.next_privileged.observed),
                                env.progress()['distance_m'], transition.parts.new_area_m2,
                                execution.reason_code, transition.terminated, transition.truncated,
                                covered_area_m2=env.reference.covered_area(transition.next_privileged.observed),
                                coverable_area_m2=env.reference.area_m2)
                            obs = transition.next_observation
                    except Exception as exc:
                        metric.error = f'{type(exc).__name__}: {exc}'
                    finally: env.close()
                    rows.append(metric.record())
                    print(json.dumps(rows[-1], ensure_ascii=False, allow_nan=False), flush=True)
    finally: env.close()
    result = dict(actor=str(Path(actor_path).resolve()), policy='frozen_joint_argmax',
        sensor=asdict(config.sensor),
        budget=budget, groups=summarize(rows), cases=rows)
    if output is not None:
        payload = (json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()
        if len(payload) > config.metrics_max_bytes: raise ValueError('evaluation report exceeds configured metrics bound')
        _atomic_file(output, lambda stream: stream.write(payload))
    print(json.dumps(result['groups'], ensure_ascii=False, allow_nan=False), flush=True)
    return result


def _atomic_file(output, write):
    # Reuse Task6's write-time old+temporary budget enforcement and fsync.
    from .checkpoint import CheckpointManager
    output = Path(output)
    manager = CheckpointManager(output.parent)
    manager._atomic(output.name, write)


def export_actor(checkpoint, output):
    import torch
    # Full checkpoints are explicitly trusted local continuation artifacts.
    record = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if record.get('schema') == 'bounded_training_v1':
        from .checkpoint import TrainingState
        TrainingState(record).validate()
        source = record['learner']
        actor = dict(schema='task_graph_v1', model_config=source['model_config'],
            version=source['updates'], state_dict=source['actor'])
    elif record.get('schema') == 'task_graph_v1': actor = record
    else: raise ValueError('unsupported checkpoint/Actor schema; no old GRU migration')
    # Same loader's exact shape/schema checks, before replacing an existing file.
    from .model import Actor
    from .config import ModelConfig
    if set(actor) != {'schema', 'model_config', 'version', 'state_dict'}:
        raise ValueError('invalid actor_state artifact keys')
    Actor(ModelConfig(**actor['model_config'])).load_state_dict(actor['state_dict'], strict=True)
    _atomic_file(output, lambda stream: torch.save(actor, stream))
    print(json.dumps(dict(actor=str(output), source=str(checkpoint), version=actor['version'],
        evaluated_best=False)), flush=True)
    return str(output)


def infer(config, actor_path, *, task_topic='/Car/T4/exploration/task'):
    import time
    import torch
    import rclpy
    from .runtime import ActorPolicy, InferenceRuntime
    torch.set_num_threads(config.collector_threads)
    policy = ActorPolicy.load(actor_path)
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('drl_exploration_inference')
    runtime = InferenceRuntime.attach(node, policy, sensor=config.sensor, task_topic=task_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        runtime.cancel()
        deadline = time.monotonic() + 10
        while runtime.core is not None and runtime.state != 'CANCELED' and time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.05)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

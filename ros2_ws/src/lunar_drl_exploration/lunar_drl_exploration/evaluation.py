"""Frozen evaluation/export and deployment entrypoints with lazy dependencies."""
from dataclasses import asdict
import json
from pathlib import Path


class GeometryMetrics:
    """Fixed 0.5 m world cells, independent of changing graph node identities."""
    def __init__(self, xy, tolerance_m=.05):
        self.tolerance_m = tolerance_m
        self.pose_history = [tuple(map(float, xy[:2]))]
        self.two_point_steps = 0
        self.previous = self.cell(xy)
        self.seen = {self.previous}
        self.last_edge = None
        self.steps = self.revisits = self.reversals = self.stationary = 0
        self.zero_run = self.max_zero_run = 0

    @staticmethod
    def cell(xy):
        import math
        return tuple(math.floor(float(v) / .5) for v in xy[:2])

    def observe(self, xy, gain):
        current = self.cell(xy)
        self.steps += 1
        if gain > 0:
            self.pose_history.clear()
        self.pose_history.append(tuple(map(float, xy[:2])))
        del self.pose_history[:-5]
        if len(self.pose_history) == 5:
            import math
            a, b, c, d, e = self.pose_history
            if (math.dist(a, b) > self.tolerance_m and
                    max(math.dist(a, c), math.dist(a, e), math.dist(b, d)) <= self.tolerance_m):
                self.two_point_steps += 1
        self.revisits += current in self.seen
        self.stationary += current == self.previous
        if current != self.previous:
            edge = (self.previous, current)
            self.reversals += self.last_edge == (current, self.previous)
            self.last_edge = edge
        self.seen.add(current)
        self.previous = current
        self.zero_run = self.zero_run + 1 if gain <= 0 else 0
        self.max_zero_run = max(self.max_zero_run, self.zero_run)

    def record(self):
        return dict(geometric_cell_m=.5, two_point_tolerance_m=self.tolerance_m,
            zero_gain_two_point_loop_steps=self.two_point_steps, reverse_edge_count=self.reversals,
            stationary_decisions=self.stationary, max_zero_gain_run=self.max_zero_run,
            revisit_ratio=self.revisits / self.steps if self.steps else 0.)


class EpisodeMetrics:
    def __init__(self, family, extent, seed):
        self.family, self.extent, self.seed = family, extent, seed
        self.coverage = self.distance = 0.
        self.covered_area_m2 = self.coverable_area_m2 = None
        self.path80 = self.path99 = None
        self.completed = self.exhausted = self.terminated = self.truncated = False
        self.coverage_lower_bound=self.remaining_area_upper_m2=None
        self.steps = self.zero_gain = self.failures = self.collisions = 0
        self.reasons = {}
        self.error = None

    def observe(self, coverage, distance, new_area, reason, terminated, truncated,
                *, covered_area_m2=None, coverable_area_m2=None, exhausted=False,
                coverage_lower_bound=None,remaining_area_upper_m2=None):
        self.coverage, self.distance = float(coverage), float(distance)
        self.covered_area_m2 = None if covered_area_m2 is None else float(covered_area_m2)
        self.coverable_area_m2 = None if coverable_area_m2 is None else float(coverable_area_m2)
        self.terminated,self.exhausted,self.truncated=bool(terminated),bool(exhausted),bool(truncated)
        self.coverage_lower_bound=coverage_lower_bound
        self.remaining_area_upper_m2=remaining_area_upper_m2
        if coverage >= .8 and self.path80 is None: self.path80 = float(distance)
        if coverage >= .99 and self.path99 is None: self.path99 = float(distance)
        if reason != 'INITIAL':
            self.steps += 1; self.zero_gain += new_area <= 0
            self.failures += reason != 'GOAL_REACHED'
            self.collisions += reason == 'COLLISION'
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
        # A space-completion terminal can coincide with a physical failure.
        # Preserve its coverage and RL terminal flag without calling it success.
        self.completed=self.terminated and not self.collisions

    def record(self):
        return dict(family=self.family, extent_m=self.extent, seed=self.seed,
            covered_area_m2=self.covered_area_m2, coverable_area_m2=self.coverable_area_m2,
            final_coverage=self.coverage, reached_80=self.path80 is not None,
            reached_99=self.path99 is not None, completed=self.completed,exhausted=self.exhausted,
            terminated=self.terminated,
            coverage_lower_bound=self.coverage_lower_bound,remaining_area_upper_m2=self.remaining_area_upper_m2,
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
            completion_rate=sum(row['completed'] for row in cases)/count,
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
                    geometry = None
                    try:
                        obs, state = env.reset(seed, family, extent, episode_budget=budget)
                        geometry = GeometryMetrics(obs.positions[obs.current_index], config.goal_position_tolerance_m)
                        progress = env.progress()
                        metric.observe(env.reference.coverage_ratio(state.observed),
                            progress['distance_m'], 0., 'INITIAL', env.report.completed, False,
                            exhausted=env.report.exhausted,coverage_lower_bound=env.report.coverage_lower_bound,
                            remaining_area_upper_m2=env.report.remaining_area_upper_m2,
                            covered_area_m2=env.reference.covered_area(state.observed),
                            coverable_area_m2=env.reference.area_m2)
                        while not metric.completed and not metric.truncated and not metric.collisions:
                            transition = env.step(policy(obs))
                            next_obs = transition.next_observation
                            geometry.observe(next_obs.positions[next_obs.current_index], transition.parts.new_area_m2)
                            execution = env.last_execution
                            metric.observe(env.reference.coverage_ratio(transition.next_privileged.observed),
                                env.progress()['distance_m'], transition.parts.new_area_m2,
                                execution.reason_code, transition.terminated, transition.truncated,
                                exhausted=env.report.exhausted,coverage_lower_bound=env.report.coverage_lower_bound,
                                remaining_area_upper_m2=env.report.remaining_area_upper_m2,
                                covered_area_m2=env.reference.covered_area(transition.next_privileged.observed),
                                coverable_area_m2=env.reference.area_m2)
                            obs = transition.next_observation
                    except Exception as exc:
                        metric.error = f'{type(exc).__name__}: {exc}'
                    finally: env.close()
                    row = metric.record()
                    row.update(geometry.record() if geometry else dict(geometric_metrics_unavailable=True,
                        max_zero_gain_run=0, zero_gain_two_point_loop_steps=0))
                    rows.append(row)
                    print(json.dumps(rows[-1], ensure_ascii=False, allow_nan=False), flush=True)
    finally: env.close()
    result = dict(actor=str(Path(actor_path).resolve()), policy='frozen_joint_argmax',
        sensor=asdict(config.sensor),coverage_target=config.coverage_target,
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
        if record['semantics'].get('action_schema') != 'local_metric_pose_v3':
            raise ValueError('incompatible legacy action semantics; train a new policy')
        source = record['learner']
        actor = dict(schema='task_graph_v3', model_config=source['model_config'],
            version=source['updates'], state_dict=source['actor'])
    elif record.get('schema') == 'task_graph_v3': actor = record
    else: raise ValueError('unsupported checkpoint/Actor schema; no old GRU migration')
    # Same loader's exact shape/schema checks, before replacing an existing file.
    from .model import Actor
    from .config import ModelConfig, canonical_model_config
    if set(actor) != {'schema', 'model_config', 'version', 'state_dict'}:
        raise ValueError('invalid actor_state artifact keys')
    actor = dict(actor, model_config=canonical_model_config(actor['model_config']))
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
    runtime = InferenceRuntime.attach(node, policy, sensor=config.sensor, task_topic=task_topic,
                                     coverage_target=config.coverage_target)
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

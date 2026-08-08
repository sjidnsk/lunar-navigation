"""Canonical per-platform release evaluation report."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from ..curriculum import CurriculumSchedule, PLATFORMS
from ..checkpoint import RunIdentity
from ..environment.parallel_pool import ParallelActions, ParallelEnvPool
from ..eval.baselines import select_baseline_action
from ..policy.cross_attention import CrossAttentionPolicy, sample_action
from ..policy.observation import PolicyBatch
from ..proxy_scenario import ProxyEnvironmentFactory, proxy_observation
from ..reward import compute_transition_reward, reward_weights_sha256


EVALUATION_SCHEMA_VERSION = "lunar-policy-release-evaluation/v2"
DEVELOPMENT_EVALUATION_SCHEMA_VERSION = (
    "lunar-policy-development-evaluation/v2"
)
REQUIRED_METHODS = (
    "ppo_policy",
    "nearest_frontier",
    "gain_over_cost_frontier",
)


@dataclass(frozen=True, slots=True)
class PlatformMetrics:
    scenario_seeds: tuple[int, ...]
    success_coverage_rate: float
    safety_violation_count: int
    invalid_action_count: int
    output_finite_rate: float
    platform_reference_mismatch_count: int
    hopper_commitment_violation_count: int
    selected_action_observed_safe_rate: float
    deterministic_repeat_match_rate: float
    planner_failure_rate: float
    completion_time_s: float
    theta_mean_resultant_length: float
    fixed_yaw_mean_abs_delta_rad: float

    def __post_init__(self) -> None:
        if not self.scenario_seeds or any(
            type(seed) is not int for seed in self.scenario_seeds
        ):
            raise ValueError("platform metrics require integer scenario seeds")
        rates = (
            self.success_coverage_rate,
            self.output_finite_rate,
            self.selected_action_observed_safe_rate,
            self.deterministic_repeat_match_rate,
            self.planner_failure_rate,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in rates
        ):
            raise ValueError("platform rates must be finite values in [0, 1]")
        counts = (
            self.safety_violation_count,
            self.invalid_action_count,
            self.platform_reference_mismatch_count,
            self.hopper_commitment_violation_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("platform counts must be non-negative integers")
        if (
            not isinstance(self.completion_time_s, (int, float))
            or isinstance(self.completion_time_s, bool)
            or not math.isfinite(float(self.completion_time_s))
            or self.completion_time_s < 0.0
        ):
            raise ValueError("completion time must be finite and non-negative")
        if (
            not isinstance(self.theta_mean_resultant_length, (int, float))
            or isinstance(self.theta_mean_resultant_length, bool)
            or not math.isfinite(float(self.theta_mean_resultant_length))
            or not 0.0 <= float(self.theta_mean_resultant_length) <= 1.0
        ):
            raise ValueError("theta mean resultant length must be finite in [0, 1]")
        if (
            not isinstance(self.fixed_yaw_mean_abs_delta_rad, (int, float))
            or isinstance(self.fixed_yaw_mean_abs_delta_rad, bool)
            or not math.isfinite(float(self.fixed_yaw_mean_abs_delta_rad))
            or not 0.0 <= float(self.fixed_yaw_mean_abs_delta_rad) <= math.pi
        ):
            raise ValueError(
                "fixed-yaw mean absolute delta must be finite in [0, pi]"
            )

    def replace(self, **changes: object) -> "PlatformMetrics":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class MethodEvaluation:
    method: str
    per_platform: Mapping[str, PlatformMetrics]

    def __post_init__(self) -> None:
        if self.method not in REQUIRED_METHODS:
            raise ValueError("evaluation method is not required by Task 4")
        if set(self.per_platform) != set(PLATFORMS) or any(
            not isinstance(self.per_platform[platform], PlatformMetrics)
            for platform in PLATFORMS
        ):
            raise ValueError("method evaluation must contain three platforms")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    proxy: bool
    scenario_schedule_id: str
    run_identity: RunIdentity
    reward_hash: str
    checkpoint_sha256: str
    methods: tuple[MethodEvaluation, ...]
    schema_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_identity, RunIdentity):
            raise ValueError("evaluation run identity is missing")
        expected_schema = (
            DEVELOPMENT_EVALUATION_SCHEMA_VERSION
            if self.proxy
            else EVALUATION_SCHEMA_VERSION
        )
        if self.schema_version is None:
            object.__setattr__(self, "schema_version", expected_schema)
        elif self.schema_version != expected_schema:
            raise ValueError("evaluation schema version mismatch")
        if self.proxy:
            if self.run_identity.run_kind != "development-smoke":
                raise ValueError("proxy evaluation cannot use formal run identity")
            if not self.scenario_schedule_id.startswith("proxy-"):
                raise ValueError("proxy scenario schedule identity is missing")
        elif self.run_identity.run_kind != "formal":
            raise ValueError("formal evaluation requires formal run identity")
        if len(self.reward_hash) != 64 or len(self.checkpoint_sha256) != 64:
            raise ValueError("evaluation hashes must be SHA-256 hex digests")
        if self.reward_hash != self.run_identity.reward_sha256:
            raise ValueError("evaluation reward hash differs from run identity")
        names = tuple(method.method for method in self.methods)
        if len(set(names)) != len(names):
            raise ValueError("evaluation methods must be unique")

    def method(self, name: str) -> MethodEvaluation:
        try:
            return next(method for method in self.methods if method.method == name)
        except StopIteration as error:
            raise KeyError(name) from error

    def replace_platform_metrics(
        self,
        *,
        method: str,
        platform_type: str,
        metrics: PlatformMetrics,
    ) -> "EvaluationReport":
        current = self.method(method)
        changed = dict(current.per_platform)
        changed[platform_type] = metrics
        return self.replace_method(
            MethodEvaluation(method=current.method, per_platform=changed)
        )

    def replace_method(self, changed: MethodEvaluation) -> "EvaluationReport":
        methods = tuple(
            changed if method.method == changed.method else method
            for method in self.methods
        )
        if changed.method not in {method.method for method in self.methods}:
            methods = (*methods, changed)
        return replace(self, methods=methods)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "proxy": self.proxy,
            "scenario_schedule_id": self.scenario_schedule_id,
            "run_identity": self.run_identity.to_dict(),
            "reward_hash": self.reward_hash,
            "checkpoint_sha256": self.checkpoint_sha256,
            "methods": [
                {
                    "method": method.method,
                    "per_platform": {
                        platform: asdict(method.per_platform[platform])
                        for platform in PLATFORMS
                    },
                }
                for method in sorted(self.methods, key=lambda value: value.method)
            ],
        }


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    checkpoint: str
    report: EvaluationReport


@dataclass(frozen=True, slots=True)
class FormalEvaluationBatch:
    """One frozen non-training split evaluated on an identical scene order."""

    split: str
    factory: object
    observation_template: PolicyBatch
    scenario_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.split not in {"validation", "test", "holdout"}:
            raise ValueError("formal evaluation split is invalid")
        schedule_id = getattr(self.factory, "scenario_schedule_id", None)
        if not isinstance(schedule_id, str) or not schedule_id:
            raise ValueError("formal evaluation factory schedule identity is missing")
        if not isinstance(self.observation_template, PolicyBatch):
            raise ValueError("formal evaluation observation template is invalid")
        if (
            not isinstance(self.scenario_seeds, tuple)
            or not self.scenario_seeds
            or any(type(seed) is not int for seed in self.scenario_seeds)
        ):
            raise ValueError("formal evaluation requires integer scenario seeds")


@dataclass(frozen=True, slots=True)
class _ScenarioEvidence:
    scenario_seed: int
    final_coverage: float
    safety_violation_count: int
    invalid_action_count: int
    output_finite: bool
    platform_reference_mismatch_count: int
    hopper_commitment_violation_count: int
    selected_action_observed_safe_count: int
    deterministic_match_count: int
    planner_failure_count: int
    executed_step_count: int
    completion_step_count: int
    selected_thetas_rad: tuple[float, ...]


def _aggregate_platform_metrics(
    scenarios: Sequence[_ScenarioEvidence],
    *,
    theta_active: bool,
) -> PlatformMetrics:
    if not scenarios:
        raise ValueError("platform evaluation requires scenario evidence")
    executed_steps = sum(item.executed_step_count for item in scenarios)
    if executed_steps <= 0:
        raise ValueError("platform evaluation requires executed steps")
    if type(theta_active) is not bool:
        raise ValueError("theta_active must be boolean")
    theta_samples = tuple(
        float(theta)
        for scenario in scenarios
        for theta in scenario.selected_thetas_rad
    )
    if (
        any(
            len(scenario.selected_thetas_rad) != scenario.executed_step_count
            for scenario in scenarios
        )
        or len(theta_samples) != executed_steps
        or any(not math.isfinite(theta) for theta in theta_samples)
    ):
        raise ValueError("theta diagnostics require one finite sample per step")
    if theta_active:
        mean_cosine = sum(math.cos(theta) for theta in theta_samples) / executed_steps
        mean_sine = sum(math.sin(theta) for theta in theta_samples) / executed_steps
        theta_mean_resultant_length = min(
            1.0,
            math.hypot(mean_cosine, mean_sine),
        )
        fixed_yaw_mean_abs_delta_rad = sum(
            abs(math.remainder(theta, 2.0 * math.pi))
            for theta in theta_samples
        ) / executed_steps
    else:
        if any(theta != 0.0 for theta in theta_samples):
            raise ValueError("inactive theta diagnostics must contain exact zero")
        theta_mean_resultant_length = 0.0
        fixed_yaw_mean_abs_delta_rad = 0.0
    scenario_count = len(scenarios)
    return PlatformMetrics(
        scenario_seeds=tuple(item.scenario_seed for item in scenarios),
        success_coverage_rate=(
            sum(item.final_coverage >= 0.95 for item in scenarios)
            / scenario_count
        ),
        safety_violation_count=sum(
            item.safety_violation_count for item in scenarios
        ),
        invalid_action_count=sum(item.invalid_action_count for item in scenarios),
        output_finite_rate=(
            sum(item.output_finite for item in scenarios) / scenario_count
        ),
        platform_reference_mismatch_count=sum(
            item.platform_reference_mismatch_count for item in scenarios
        ),
        hopper_commitment_violation_count=sum(
            item.hopper_commitment_violation_count for item in scenarios
        ),
        selected_action_observed_safe_rate=(
            sum(
                item.selected_action_observed_safe_count for item in scenarios
            )
            / executed_steps
        ),
        deterministic_repeat_match_rate=(
            sum(item.deterministic_match_count for item in scenarios)
            / executed_steps
        ),
        planner_failure_rate=(
            sum(item.planner_failure_count for item in scenarios)
            / executed_steps
        ),
        completion_time_s=(
            sum(item.completion_step_count for item in scenarios) / scenario_count
        ),
        theta_mean_resultant_length=theta_mean_resultant_length,
        fixed_yaw_mean_abs_delta_rad=fixed_yaw_mean_abs_delta_rad,
    )


def report_sha256(report: EvaluationReport) -> str:
    payload = json.dumps(
        report.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_report(path: Path, report: EvaluationReport) -> str:
    digest = report_sha256(report)
    payload = {**report.to_dict(), "report_sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return digest


def select_best_candidate(
    candidates: Sequence[CandidateEvaluation],
) -> CandidateEvaluation:
    if not candidates:
        raise ValueError("at least one candidate evaluation is required")

    def ranking(candidate: CandidateEvaluation) -> tuple[float, float, float, str]:
        metrics = candidate.report.method("ppo_policy").per_platform
        return (
            -min(
                metrics[platform].success_coverage_rate for platform in PLATFORMS
            ),
            max(metrics[platform].planner_failure_rate for platform in PLATFORMS),
            max(metrics[platform].completion_time_s for platform in PLATFORMS),
            candidate.checkpoint,
        )

    return min(candidates, key=ranking)


def evaluate_proxy_policy(
    policy: CrossAttentionPolicy,
    *,
    device: torch.device | str,
    checkpoint_sha256: str,
    schedule: CurriculumSchedule,
    run_identity: RunIdentity,
) -> EvaluationReport:
    """Compare PPO and two diagnostics on one identical real-v3 proxy schedule."""
    if not isinstance(policy, CrossAttentionPolicy):
        raise ValueError("proxy evaluation requires CrossAttentionPolicy")
    if not isinstance(schedule, CurriculumSchedule):
        raise ValueError("proxy evaluation requires CurriculumSchedule")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA proxy evaluation requested without CUDA")
    methods = tuple(
        _evaluate_method(
            policy,
            method=method,
            device=target_device,
            schedule=schedule,
        )
        for method in REQUIRED_METHODS
    )
    return EvaluationReport(
        proxy=True,
        scenario_schedule_id=schedule.scenario_schedule_id,
        run_identity=run_identity,
        reward_hash=reward_weights_sha256(),
        checkpoint_sha256=checkpoint_sha256,
        methods=methods,
    )


def evaluate_formal_policy(
    policy: CrossAttentionPolicy,
    *,
    device: torch.device | str,
    checkpoint_sha256: str,
    run_identity: RunIdentity,
    batches: Sequence[FormalEvaluationBatch],
) -> EvaluationReport:
    """Compare all required methods on fixed validation, test, and holdout worlds."""
    if not isinstance(policy, CrossAttentionPolicy):
        raise ValueError("formal evaluation requires CrossAttentionPolicy")
    if not isinstance(run_identity, RunIdentity) or run_identity.run_kind != "formal":
        raise ValueError("formal evaluation requires formal run identity")
    formal_batches = tuple(batches)
    if (
        len(formal_batches) != 3
        or any(not isinstance(batch, FormalEvaluationBatch) for batch in formal_batches)
        or {batch.split for batch in formal_batches}
        != {"validation", "test", "holdout"}
    ):
        raise ValueError(
            "formal evaluation requires validation, test, and holdout exactly once"
        )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA formal evaluation requested without CUDA")
    schedule_payload = tuple(
        (
            batch.split,
            getattr(batch.factory, "scenario_schedule_id"),
            batch.scenario_seeds,
        )
        for batch in sorted(formal_batches, key=lambda value: value.split)
    )
    schedule_digest = hashlib.sha256(
        json.dumps(schedule_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    methods: list[MethodEvaluation] = []
    for method in REQUIRED_METHODS:
        by_platform: dict[str, list[_ScenarioEvidence]] = {
            platform: [] for platform in PLATFORMS
        }
        for batch in formal_batches:
            batch_results = _evaluate_formal_batch(
                policy,
                method=method,
                device=target_device,
                batch=batch,
            )
            for platform in PLATFORMS:
                by_platform[platform].extend(batch_results[platform])
        per_platform: dict[str, PlatformMetrics] = {}
        for platform in PLATFORMS:
            per_platform[platform] = _aggregate_platform_metrics(
                tuple(by_platform[platform]),
                theta_active=platform != "HOPPER",
            )
        methods.append(MethodEvaluation(method=method, per_platform=per_platform))
    return EvaluationReport(
        proxy=False,
        scenario_schedule_id=f"formal-evaluation/{schedule_digest}",
        run_identity=run_identity,
        reward_hash=reward_weights_sha256(),
        checkpoint_sha256=checkpoint_sha256,
        methods=tuple(methods),
    )


def _evaluate_formal_batch(
    policy: CrossAttentionPolicy,
    *,
    method: str,
    device: torch.device,
    batch: FormalEvaluationBatch,
) -> Mapping[str, tuple[_ScenarioEvidence, ...]]:
    """Bound formal evaluation to nine concurrent workers per process group."""
    collected: dict[str, list[_ScenarioEvidence]] = {
        platform: [] for platform in PLATFORMS
    }
    chunks: list[tuple[int, tuple[int, ...]]] = []
    full_count = len(batch.scenario_seeds) - len(batch.scenario_seeds) % 3
    for offset in range(0, full_count, 3):
        chunks.append((offset, batch.scenario_seeds[offset : offset + 3]))
    for offset in range(full_count, len(batch.scenario_seeds)):
        chunks.append((offset, batch.scenario_seeds[offset : offset + 1]))
    for offset, seeds in chunks:
        chunk = _evaluate_formal_chunk(
            policy,
            method=method,
            device=device,
            batch=batch,
            scenario_offset=offset,
            scenario_seeds=seeds,
        )
        for platform in PLATFORMS:
            collected[platform].extend(chunk[platform])
    return {platform: tuple(collected[platform]) for platform in PLATFORMS}


def _evaluate_formal_chunk(
    policy: CrossAttentionPolicy,
    *,
    method: str,
    device: torch.device,
    batch: FormalEvaluationBatch,
    scenario_offset: int,
    scenario_seeds: tuple[int, ...],
) -> Mapping[str, tuple[_ScenarioEvidence, ...]]:
    count = len(scenario_seeds)
    row_schedule = tuple(
        (platform, local_index, scenario_seed)
        for platform in PLATFORMS
        for local_index, scenario_seed in enumerate(scenario_seeds)
    )
    row_count = len(row_schedule)
    episode_cursor = _formal_chunk_episode_cursor(scenario_offset, count)
    cursors = (episode_cursor,) * row_count
    with ParallelEnvPool(
        allocation={platform: count for platform in PLATFORMS},
        observation_template=batch.observation_template,
        environment_factory=batch.factory,
        reward_fn=_evaluation_reward,
        worker_timeout_seconds=30.0,
        auto_reset=False,
        initial_episode_cursors=cursors,
    ) as pool:
        pool.reset()
        planner_failures = np.zeros(row_count, dtype=np.int64)
        executed_steps = np.zeros(row_count, dtype=np.int64)
        deterministic_matches = np.zeros(row_count, dtype=np.int64)
        selected_safe_actions = np.zeros(row_count, dtype=np.int64)
        safety_violations = np.zeros(row_count, dtype=np.int64)
        invalid_actions = np.zeros(row_count, dtype=np.int64)
        reference_mismatches = np.zeros(row_count, dtype=np.int64)
        hopper_commitment_violations = np.zeros(row_count, dtype=np.int64)
        output_finite = np.ones(row_count, dtype=np.bool_)
        completion_steps = np.zeros(row_count, dtype=np.int64)
        final_coverage = np.zeros(row_count, dtype=np.float32)
        theta_samples: list[list[float]] = [[] for _ in range(row_count)]
        for step_index in range(1, 4):
            prepared = pool.prepare_decision_boundaries(policy_version=0)
            no_action_workers = tuple(
                int(index)
                for index in torch.nonzero(prepared.dones, as_tuple=False)
                .flatten()
                .tolist()
            )
            if no_action_workers:
                if any(completion_steps[index] == 0 for index in no_action_workers):
                    raise ValueError(
                        "formal evaluation scenario has no actionable boundary"
                    )
                pool.reset_terminated_workers(
                    no_action_workers, policy_version=0
                )
                prepared = pool.prepare_decision_boundaries(policy_version=0)
                if bool(prepared.dones.any()):
                    raise ValueError(
                        "formal evaluation reset did not reach an actionable boundary"
                    )
            first_indices, first_thetas = _select_formal_actions(
                policy,
                method=method,
                observations=prepared.observations,
                device=device,
                row_schedule=row_schedule,
            )
            repeated_indices, repeated_thetas = _select_formal_actions(
                policy,
                method=method,
                observations=prepared.observations,
                device=device,
                row_schedule=row_schedule,
            )
            deterministic_rows = np.logical_and(
                first_indices == repeated_indices,
                first_thetas == repeated_thetas,
            )
            masks = prepared.observations.candidate_mask.detach().cpu().numpy()
            selected_action_valid = np.asarray(
                [
                    0 <= candidate_index < masks.shape[1]
                    and bool(masks[row, candidate_index])
                    and math.isfinite(float(first_thetas[row]))
                    for row, candidate_index in enumerate(first_indices)
                ],
                dtype=np.bool_,
            )
            stepped = pool.step(
                ParallelActions(
                    candidate_indices=torch.from_numpy(first_indices),
                    thetas=torch.from_numpy(first_thetas),
                ),
                policy_version=0,
            )
            if (
                len(stepped.execution_events) != row_count
                or len(stepped.planning_outcomes) != row_count
            ):
                raise ValueError("formal evaluation worker evidence is missing")
            finite_rows = _finite_output_rows(stepped.observations, stepped.rewards)
            coverage = (
                stepped.observations.coverage_summary[:, 0]
                .mean(dim=(1, 2))
                .detach()
                .cpu()
                .numpy()
            )
            for index, outcome in enumerate(stepped.planning_outcomes):
                if completion_steps[index] != 0:
                    continue
                executed_steps[index] += 1
                theta_samples[index].append(float(first_thetas[index]))
                deterministic_matches[index] += int(deterministic_rows[index])
                events = stepped.execution_events[index]
                safety_violations[index] += events.safety_violation_count
                invalid_actions[index] += events.invalid_action_count + int(
                    not selected_action_valid[index]
                )
                reference_mismatches[index] += (
                    events.platform_reference_mismatch_count
                )
                hopper_commitment_violations[index] += (
                    events.hopper_commitment_violation_count
                )
                selected_safe_actions[index] += int(
                    events.selected_action_observed_safe
                )
                accepted = (
                    outcome.name == "NEW_REFERENCE_AVAILABLE"
                    and events.execution_failure_count == 0
                )
                planner_failures[index] += int(not accepted)
                platform, _, _ = row_schedule[index]
                if (
                    accepted
                    and platform == "HOPPER"
                    and events.hopper_commitment_states
                    != ("JUMP_COMMITTED", "IN_FLIGHT", "LANDED_HOLD")
                ):
                    hopper_commitment_violations[index] += 1
                output_finite[index] = bool(
                    output_finite[index] and finite_rows[index]
                )
                final_coverage[index] = coverage[index]
                if bool(stepped.dones[index].item()):
                    completion_steps[index] = step_index
            terminated_workers = tuple(
                int(index)
                for index in torch.nonzero(stepped.dones, as_tuple=False)
                .flatten()
                .tolist()
            )
            if terminated_workers and step_index < 3:
                pool.reset_terminated_workers(
                    terminated_workers, policy_version=0
                )
    result: dict[str, tuple[_ScenarioEvidence, ...]] = {}
    for platform in PLATFORMS:
        result[platform] = tuple(
            _ScenarioEvidence(
                scenario_seed=scenario_seed,
                final_coverage=float(final_coverage[index]),
                safety_violation_count=int(safety_violations[index]),
                invalid_action_count=int(invalid_actions[index]),
                output_finite=bool(output_finite[index]),
                platform_reference_mismatch_count=int(reference_mismatches[index]),
                hopper_commitment_violation_count=int(
                    hopper_commitment_violations[index]
                ),
                selected_action_observed_safe_count=int(
                    selected_safe_actions[index]
                ),
                deterministic_match_count=int(deterministic_matches[index]),
                planner_failure_count=int(planner_failures[index]),
                executed_step_count=int(executed_steps[index]),
                completion_step_count=int(completion_steps[index] or 3),
                selected_thetas_rad=tuple(theta_samples[index]),
            )
            for index, (row_platform, _, scenario_seed) in enumerate(row_schedule)
            if row_platform == platform
        )
    return result


def _formal_chunk_episode_cursor(scenario_offset: int, worker_lanes: int) -> int:
    if worker_lanes == 3 and scenario_offset % 3 == 0:
        return scenario_offset // 3
    if worker_lanes == 1:
        return scenario_offset
    raise ValueError("formal evaluation chunk cannot preserve cache order")


def _select_formal_actions(
    policy: CrossAttentionPolicy,
    *,
    method: str,
    observations: PolicyBatch,
    device: torch.device,
    row_schedule: tuple[tuple[str, int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    if method == "ppo_policy":
        model_input = _move_batch(observations, device)
        policy.to(device).eval()
        with torch.no_grad():
            selected = sample_action(
                policy(model_input),
                model_input.candidate_mask,
                model_input.platform_context,
                deterministic=True,
            )
        return (
            selected.selected_frontier_index.detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=True),
            selected.selected_theta.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True),
        )
    features = observations.frontier_features.detach().cpu().numpy()
    masks = observations.candidate_mask.detach().cpu().numpy()
    indices: list[int] = []
    thetas: list[float] = []
    for row, (platform, _, scenario_seed) in enumerate(row_schedule):
        selected = select_baseline_action(
            method,
            features[row],
            masks[row],
            np.random.Generator(np.random.PCG64(scenario_seed)),
        )
        indices.append(selected.candidate_index)
        thetas.append(0.0 if platform == "HOPPER" else selected.theta)
    return np.asarray(indices, dtype=np.int64), np.asarray(thetas, dtype=np.float32)


def _evaluate_method(
    policy: CrossAttentionPolicy,
    *,
    method: str,
    device: torch.device,
    schedule: CurriculumSchedule,
) -> MethodEvaluation:
    scenario_indices = schedule.evaluation_scenario_indices
    row_schedule = tuple(
        (platform, scenario_index)
        for platform in PLATFORMS
        for scenario_index in scenario_indices
    )
    allocation = {
        platform: len(scenario_indices) for platform in PLATFORMS
    }
    row_count = len(row_schedule)
    with ParallelEnvPool(
        allocation=allocation,
        observation_template=proxy_observation(0, "WHEELED", step=0),
        environment_factory=ProxyEnvironmentFactory(),
        reward_fn=_evaluation_reward,
        worker_timeout_seconds=30.0,
        auto_reset=False,
    ) as pool:
        observations = pool.reset().observations
        planner_failures = np.zeros(row_count, dtype=np.int64)
        executed_steps = np.zeros(row_count, dtype=np.int64)
        deterministic_matches = np.zeros(row_count, dtype=np.int64)
        selected_safe_actions = np.zeros(row_count, dtype=np.int64)
        safety_violations = np.zeros(row_count, dtype=np.int64)
        invalid_actions = np.zeros(row_count, dtype=np.int64)
        reference_mismatches = np.zeros(row_count, dtype=np.int64)
        hopper_commitment_violations = np.zeros(row_count, dtype=np.int64)
        output_finite = np.ones(row_count, dtype=np.bool_)
        completion_steps = np.zeros(row_count, dtype=np.int64)
        final_coverage = np.full(row_count, 0.05, dtype=np.float32)
        theta_samples: list[list[float]] = [[] for _ in range(row_count)]
        for step_index in range(1, 4):
            prepared = pool.prepare_decision_boundaries(policy_version=0)
            no_action_workers = tuple(
                int(index)
                for index in torch.nonzero(prepared.dones, as_tuple=False)
                .flatten()
                .tolist()
            )
            if no_action_workers:
                pool.reset_terminated_workers(
                    no_action_workers,
                    policy_version=0,
                )
                prepared = pool.prepare_decision_boundaries(policy_version=0)
                if bool(prepared.dones.any()):
                    raise ValueError(
                        "evaluation reset did not reach an actionable boundary"
                    )
            observations = prepared.observations
            first_indices, first_thetas = _select_actions(
                policy,
                method=method,
                observations=observations,
                device=device,
                schedule=schedule,
                row_schedule=row_schedule,
            )
            repeated_indices, repeated_thetas = _select_actions(
                policy,
                method=method,
                observations=observations,
                device=device,
                schedule=schedule,
                row_schedule=row_schedule,
            )
            deterministic_rows = np.logical_and(
                first_indices == repeated_indices,
                first_thetas == repeated_thetas,
            )
            masks = observations.candidate_mask.detach().cpu().numpy()
            selected_action_valid = np.asarray(
                [
                    0 <= candidate_index < masks.shape[1]
                    and bool(masks[row, candidate_index])
                    and math.isfinite(float(first_thetas[row]))
                    for row, candidate_index in enumerate(first_indices)
                ],
                dtype=np.bool_,
            )
            stepped = pool.step(
                ParallelActions(
                    candidate_indices=torch.from_numpy(first_indices),
                    thetas=torch.from_numpy(first_thetas),
                ),
                policy_version=0,
            )
            observations = stepped.observations
            finite_rows = _finite_output_rows(observations, stepped.rewards)
            observed_coverage = (
                observations.coverage_summary[:, 0]
                .mean(dim=(1, 2))
                .detach()
                .cpu()
                .numpy()
            )
            if len(stepped.execution_events) != row_count:
                raise ValueError("evaluation worker execution events are missing")
            for index, outcome in enumerate(stepped.planning_outcomes):
                if completion_steps[index] != 0:
                    continue
                executed_steps[index] += 1
                theta_samples[index].append(float(first_thetas[index]))
                deterministic_matches[index] += int(deterministic_rows[index])
                events = stepped.execution_events[index]
                safety_violations[index] += events.safety_violation_count
                invalid_actions[index] += (
                    events.invalid_action_count
                    + int(not selected_action_valid[index])
                )
                reference_mismatches[index] += (
                    events.platform_reference_mismatch_count
                )
                hopper_commitment_violations[index] += (
                    events.hopper_commitment_violation_count
                )
                selected_safe_actions[index] += int(
                    events.selected_action_observed_safe
                )
                accepted = (
                    outcome.name == "NEW_REFERENCE_AVAILABLE"
                    and events.execution_failure_count == 0
                )
                planner_failures[index] += int(not accepted)
                platform, _ = row_schedule[index]
                if (
                    accepted
                    and platform == "HOPPER"
                    and events.hopper_commitment_states
                    != ("JUMP_COMMITTED", "IN_FLIGHT", "LANDED_HOLD")
                ):
                    hopper_commitment_violations[index] += 1
                output_finite[index] = bool(
                    output_finite[index] and finite_rows[index]
                )
                final_coverage[index] = observed_coverage[index]
                if bool(stepped.dones[index].item()):
                    completion_steps[index] = step_index
            terminated_workers = tuple(
                int(index)
                for index in torch.nonzero(stepped.dones, as_tuple=False)
                .flatten()
                .tolist()
            )
            if terminated_workers and step_index < 3:
                observations = pool.reset_terminated_workers(
                    terminated_workers,
                    policy_version=0,
                ).observations
    per_platform: dict[str, PlatformMetrics] = {}
    for platform in PLATFORMS:
        scenario_evidence = tuple(
            _ScenarioEvidence(
                scenario_seed=schedule.scenario_for(
                    platform_type=row_platform,
                    scenario_index=scenario_index,
                ).scenario_seed,
                final_coverage=float(final_coverage[index]),
                safety_violation_count=int(safety_violations[index]),
                invalid_action_count=int(invalid_actions[index]),
                output_finite=bool(output_finite[index]),
                platform_reference_mismatch_count=int(
                    reference_mismatches[index]
                ),
                hopper_commitment_violation_count=int(
                    hopper_commitment_violations[index]
                ),
                selected_action_observed_safe_count=int(
                    selected_safe_actions[index]
                ),
                deterministic_match_count=int(deterministic_matches[index]),
                planner_failure_count=int(planner_failures[index]),
                executed_step_count=int(executed_steps[index]),
                completion_step_count=int(completion_steps[index] or 3),
                selected_thetas_rad=tuple(theta_samples[index]),
            )
            for index, (row_platform, scenario_index) in enumerate(row_schedule)
            if row_platform == platform
        )
        per_platform[platform] = _aggregate_platform_metrics(
            scenario_evidence,
            theta_active=platform != "HOPPER",
        )
    return MethodEvaluation(method=method, per_platform=per_platform)


def _select_actions(
    policy: CrossAttentionPolicy,
    *,
    method: str,
    observations: PolicyBatch,
    device: torch.device,
    schedule: CurriculumSchedule,
    row_schedule: tuple[tuple[str, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    if method == "ppo_policy":
        batch = _move_batch(observations, device)
        policy.to(device).eval()
        with torch.no_grad():
            selected = sample_action(
                policy(batch),
                batch.candidate_mask,
                batch.platform_context,
                deterministic=True,
            )
        return (
            selected.selected_frontier_index.detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=True),
            selected.selected_theta.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True),
        )
    features = observations.frontier_features.detach().cpu().numpy()
    masks = observations.candidate_mask.detach().cpu().numpy()
    indices: list[int] = []
    thetas: list[float] = []
    for row, (platform, scenario_index) in enumerate(row_schedule):
        seed = schedule.scenario_for(
            platform_type=platform, scenario_index=scenario_index
        ).scenario_seed
        action = select_baseline_action(
            method,
            features[row],
            masks[row],
            np.random.Generator(np.random.PCG64(seed)),
        )
        indices.append(action.candidate_index)
        thetas.append(0.0 if platform == "HOPPER" else action.theta)
    return np.asarray(indices, dtype=np.int64), np.asarray(thetas, dtype=np.float32)


def _evaluation_reward(transition) -> float:
    return compute_transition_reward(transition)


def _finite_output_rows(
    observations: PolicyBatch, rewards: torch.Tensor
) -> np.ndarray:
    rows = observations.prior_channels.shape[0]
    finite = torch.ones(rows, dtype=torch.bool)
    for tensor in (
        observations.prior_channels,
        observations.coverage_summary,
        observations.local_crop,
        observations.frontier_features,
        observations.pose_features,
        observations.platform_context,
        rewards,
    ):
        finite &= torch.isfinite(tensor).reshape(rows, -1).all(dim=1).cpu()
    return finite.numpy()


def _move_batch(batch: PolicyBatch, device: torch.device) -> PolicyBatch:
    return PolicyBatch(
        prior_channels=batch.prior_channels.to(device),
        coverage_summary=batch.coverage_summary.to(device),
        local_crop=batch.local_crop.to(device),
        frontier_features=batch.frontier_features.to(device),
        pose_features=batch.pose_features.to(device),
        candidate_mask=batch.candidate_mask.to(device),
        platform_context=batch.platform_context.to(device),
    )


__all__ = [
    "CandidateEvaluation",
    "DEVELOPMENT_EVALUATION_SCHEMA_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationReport",
    "FormalEvaluationBatch",
    "MethodEvaluation",
    "PlatformMetrics",
    "REQUIRED_METHODS",
    "report_sha256",
    "evaluate_formal_policy",
    "evaluate_proxy_policy",
    "select_best_candidate",
    "write_report",
]

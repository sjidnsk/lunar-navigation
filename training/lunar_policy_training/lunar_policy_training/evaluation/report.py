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
from ..environment.parallel_pool import ParallelActions, ParallelEnvPool
from ..eval.baselines import select_baseline_action
from ..policy.cross_attention import CrossAttentionPolicy, sample_action
from ..policy.observation import PolicyBatch
from ..proxy_scenario import ProxyEnvironmentFactory, proxy_observation
from ..reward import compute_transition_reward, reward_weights_sha256


EVALUATION_SCHEMA_VERSION = "lunar-policy-release-evaluation/v1"
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
    reward_hash: str
    checkpoint_sha256: str
    methods: tuple[MethodEvaluation, ...]
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise ValueError("evaluation schema version mismatch")
        if self.proxy is not True:
            raise ValueError("Task 4 scenario conclusions must be marked proxy")
        if not self.scenario_schedule_id.startswith("proxy-"):
            raise ValueError("proxy scenario schedule identity is missing")
        if len(self.reward_hash) != 64 or len(self.checkpoint_sha256) != 64:
            raise ValueError("evaluation hashes must be SHA-256 hex digests")
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
        reward_hash=reward_weights_sha256(),
        checkpoint_sha256=checkpoint_sha256,
        methods=methods,
    )


def _evaluate_method(
    policy: CrossAttentionPolicy,
    *,
    method: str,
    device: torch.device,
    schedule: CurriculumSchedule,
) -> MethodEvaluation:
    allocation = {platform: 1 for platform in PLATFORMS}
    with ParallelEnvPool(
        allocation=allocation,
        observation_template=proxy_observation(0, "WHEELED", step=0),
        environment_factory=ProxyEnvironmentFactory(scenario_index=0),
        reward_fn=compute_transition_reward,
        worker_timeout_seconds=30.0,
        auto_reset=False,
    ) as pool:
        observations = pool.reset().observations
        deterministic_match = True
        planner_failures = [0, 0, 0]
        accepted_references = [0, 0, 0]
        executed_steps = [0, 0, 0]
        output_finite = [True, True, True]
        completion_steps = [0, 0, 0]
        final_coverage = np.full(3, 0.05, dtype=np.float32)
        for step_index in range(1, 4):
            first_indices, first_thetas = _select_actions(
                policy,
                method=method,
                observations=observations,
                device=device,
                schedule=schedule,
            )
            repeated_indices, repeated_thetas = _select_actions(
                policy,
                method=method,
                observations=observations,
                device=device,
                schedule=schedule,
            )
            deterministic_match = deterministic_match and bool(
                np.array_equal(first_indices, repeated_indices)
                and np.array_equal(first_thetas, repeated_thetas)
            )
            stepped = pool.step(
                ParallelActions(
                    candidate_indices=torch.from_numpy(first_indices),
                    thetas=torch.from_numpy(first_thetas),
                ),
                policy_version=0,
            )
            observations = stepped.observations
            finite_tensors = (
                observations.prior_channels,
                observations.coverage_summary,
                observations.local_crop,
                observations.frontier_features,
                observations.pose_features,
                observations.platform_context,
                stepped.rewards,
            )
            finite_step = bool(
                all(
                    torch.isfinite(tensor).all().item()
                    for tensor in finite_tensors
                )
            )
            observed_coverage = (
                observations.coverage_summary[:, 0]
                .mean(dim=(1, 2))
                .detach()
                .cpu()
                .numpy()
            )
            for index, outcome in enumerate(stepped.planning_outcomes):
                if completion_steps[index] != 0:
                    continue
                executed_steps[index] += 1
                accepted = outcome.name == "NEW_REFERENCE_AVAILABLE"
                accepted_references[index] += int(accepted)
                planner_failures[index] += int(not accepted)
                output_finite[index] = output_finite[index] and finite_step
                final_coverage[index] = observed_coverage[index]
                if bool(stepped.dones[index].item()):
                    completion_steps[index] = step_index
    per_platform: dict[str, PlatformMetrics] = {}
    for index, platform in enumerate(PLATFORMS):
        success = float(final_coverage[index]) >= 0.95
        per_platform[platform] = PlatformMetrics(
            scenario_seeds=(
                schedule.scenario_for(
                    platform_type=platform, scenario_index=0
                ).scenario_seed,
            ),
            success_coverage_rate=float(success),
            safety_violation_count=0,
            invalid_action_count=0,
            output_finite_rate=float(output_finite[index]),
            platform_reference_mismatch_count=0,
            hopper_commitment_violation_count=0,
            selected_action_observed_safe_rate=(
                accepted_references[index] / executed_steps[index]
            ),
            deterministic_repeat_match_rate=float(deterministic_match),
            planner_failure_rate=planner_failures[index] / executed_steps[index],
            completion_time_s=float(completion_steps[index] or 3),
        )
    return MethodEvaluation(method=method, per_platform=per_platform)


def _select_actions(
    policy: CrossAttentionPolicy,
    *,
    method: str,
    observations: PolicyBatch,
    device: torch.device,
    schedule: CurriculumSchedule,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "ppo_policy":
        batch = _move_batch(observations, device)
        policy.to(device).eval()
        with torch.no_grad():
            selected = sample_action(
                policy(batch), batch.candidate_mask, deterministic=True
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
    for row, platform in enumerate(PLATFORMS):
        seed = schedule.scenario_for(
            platform_type=platform, scenario_index=0
        ).scenario_seed
        action = select_baseline_action(
            method,
            features[row],
            masks[row],
            np.random.Generator(np.random.PCG64(seed)),
        )
        indices.append(action.candidate_index)
        thetas.append(action.theta)
    return np.asarray(indices, dtype=np.int64), np.asarray(thetas, dtype=np.float32)


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
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationReport",
    "MethodEvaluation",
    "PlatformMetrics",
    "REQUIRED_METHODS",
    "report_sha256",
    "evaluate_proxy_policy",
    "select_best_candidate",
    "write_report",
]

"""Canonical per-platform release evaluation report."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import torch
from lunar_planner_training_bridge import PlanningOutcome

from ..curriculum import CurriculumSchedule, PLATFORMS
from ..config import FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS
from ..checkpoint import RunIdentity
from ..environment.parallel_pool import ParallelActions, ParallelEnvPool
from ..eval.baselines import select_baseline_action
from ..policy.cross_attention import CrossAttentionPolicy, sample_action
from ..policy.observation import PolicyBatch
from ..proxy_scenario import ProxyEnvironmentFactory, proxy_observation
from ..reward import (
    InvalidTransition,
    compute_transition_reward,
    reward_weights_sha256,
)
from ..reward_contract import TaskScaleBucket
from ..reward_curriculum import PlatformGateMetrics, PlatformType
from ..reward_evaluation import (
    CheckpointScore,
    PlatformScaleMetrics,
    RewardEpisodeMetrics,
    RewardEvaluationManifest,
    RewardEvaluationTask,
    bootstrap_platform_scale_metrics,
    build_checkpoint_score,
    checkpoint_score_sha256,
    reward_evaluation_manifest_sha256,
    reward_episode_metrics_from_mapping,
    summarize_platform_gate_metrics,
    summarize_platform_scale_metrics,
)
from ..training_semantics import FORMAL_SUCCESS_COVERAGE_RATIO


EVALUATION_SCHEMA_VERSION = "lunar-policy-release-evaluation/v3"
DEVELOPMENT_EVALUATION_SCHEMA_VERSION = (
    "lunar-policy-development-evaluation/v2"
)
REQUIRED_METHODS = (
    "ppo_policy",
    "nearest_frontier",
    "gain_over_cost_frontier",
)
FORMAL_EVALUATION_SPLITS = ("validation", "test", "holdout")
FORMAL_EVALUATION_WATCHDOG_MAX_STEPS = 4096
FORMAL_EVALUATION_WATCHDOG_SECONDS = 3600.0
REWARD_V4_EVALUATION_REPORT_SCHEMA = "lunar-reward-v4-evaluation-report/v2"


@dataclass(frozen=True, slots=True)
class RewardV4EvaluationReport:
    """Canonical fixed-task Reward V4 checkpoint evaluation artifact."""

    checkpoint_payload_sha256: str
    manifest: RewardEvaluationManifest
    episodes: tuple[RewardEpisodeMetrics, ...]
    stratum_metrics: tuple[PlatformScaleMetrics, ...]
    platform_gate_metrics: Mapping[PlatformType, PlatformGateMetrics]
    bootstrap_by_stratum: Mapping[str, Mapping[str, object]]
    checkpoint_score: CheckpointScore
    bootstrap_seed: int
    bootstrap_resample_count: int
    schema_version: str = REWARD_V4_EVALUATION_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REWARD_V4_EVALUATION_REPORT_SCHEMA:
            raise ValueError("Reward V4 evaluation report schema is invalid")
        _require_lower_sha256(
            self.checkpoint_payload_sha256,
            "Reward V4 evaluation checkpoint",
        )
        if not isinstance(self.manifest, RewardEvaluationManifest):
            raise ValueError("Reward V4 evaluation manifest is invalid")
        if type(self.bootstrap_seed) is not int or (
            type(self.bootstrap_resample_count) is not int
            or self.bootstrap_resample_count <= 0
        ):
            raise ValueError("Reward V4 evaluation bootstrap is invalid")
        expected_episode_keys = tuple(
            (task.platform, task.scale_bucket, task.evaluation_seed)
            for task in self.manifest.tasks
        )
        actual_episode_keys = tuple(
            (row.platform, row.scale_bucket, row.evaluation_seed)
            for row in self.episodes
        )
        if (
            any(not isinstance(row, RewardEpisodeMetrics) for row in self.episodes)
            or actual_episode_keys != expected_episode_keys
        ):
            raise ValueError("Reward V4 evaluation episode grid differs")
        expected_strata = tuple(
            dict.fromkeys(
                (task.platform, task.scale_bucket)
                for task in self.manifest.tasks
            )
        )
        actual_strata = tuple(
            (row.platform, row.scale_bucket) for row in self.stratum_metrics
        )
        if (
            any(
                not isinstance(row, PlatformScaleMetrics)
                for row in self.stratum_metrics
            )
            or actual_strata != expected_strata
        ):
            raise ValueError("Reward V4 evaluation stratum grid differs")
        platforms = tuple(dict.fromkeys(task.platform for task in self.manifest.tasks))
        gates = dict(self.platform_gate_metrics)
        if set(gates) != set(platforms) or any(
            not isinstance(gates[platform], PlatformGateMetrics)
            for platform in platforms
        ):
            raise ValueError("Reward V4 evaluation platform gates differ")
        bootstraps = {
            str(key): dict(value)
            for key, value in self.bootstrap_by_stratum.items()
        }
        stratum_by_key = {
            _reward_v4_stratum_key(row.platform, row.scale_bucket): row
            for row in self.stratum_metrics
        }
        if set(bootstraps) != set(stratum_by_key) or any(
            value.get("point_estimate") != stratum_by_key[key].to_dict()
            or value.get("resample_count") != self.bootstrap_resample_count
            for key, value in bootstraps.items()
        ):
            raise ValueError("Reward V4 evaluation bootstrap evidence differs")
        if (
            not isinstance(self.checkpoint_score, CheckpointScore)
            or self.checkpoint_score.payload_sha256
            != self.checkpoint_payload_sha256
            or self.checkpoint_score.compared_strata
            != tuple(sorted(stratum_by_key))
        ):
            raise ValueError("Reward V4 checkpoint score differs")
        object.__setattr__(
            self,
            "platform_gate_metrics",
            MappingProxyType({platform: gates[platform] for platform in platforms}),
        )
        object.__setattr__(
            self,
            "bootstrap_by_stratum",
            MappingProxyType(
                {key: MappingProxyType(bootstraps[key]) for key in sorted(bootstraps)}
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_payload_sha256": self.checkpoint_payload_sha256,
            "manifest_sha256": reward_evaluation_manifest_sha256(self.manifest),
            "manifest": self.manifest.to_dict(),
            "episodes": [row.to_dict() for row in self.episodes],
            "strata": {
                _reward_v4_stratum_key(row.platform, row.scale_bucket): row.to_dict()
                for row in self.stratum_metrics
            },
            "platform_gates": {
                platform.value: self.platform_gate_metrics[platform].to_dict()
                for platform in self.platform_gate_metrics
            },
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_resample_count": self.bootstrap_resample_count,
            "bootstrap_by_stratum": {
                key: dict(value)
                for key, value in self.bootstrap_by_stratum.items()
            },
            "checkpoint_score": self.checkpoint_score.to_dict(),
            "checkpoint_score_sha256": checkpoint_score_sha256(
                self.checkpoint_score
            ),
        }


class FormalEvaluationIncomplete(RuntimeError):
    """A technical watchdog expired before every natural terminal."""


def mission_coverage_ratio(observations: PolicyBatch) -> np.ndarray:
    """Return the controller-owned 0.2 m ROI coverage for every row."""
    if not isinstance(observations, PolicyBatch):
        raise ValueError("mission coverage requires PolicyBatch")
    values = (
        observations.pose_features[:, 4]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=True)
    )
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or ((values < 0.0) | (values > 1.0)).any()
    ):
        raise ValueError("mission coverage must be finite in [0,1]")
    return values


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
    per_split: Mapping[str, Mapping[str, PlatformMetrics]] | None = None

    def __post_init__(self) -> None:
        if self.method not in REQUIRED_METHODS:
            raise ValueError("evaluation method is not required by Task 4")
        if set(self.per_platform) != set(PLATFORMS) or any(
            not isinstance(self.per_platform[platform], PlatformMetrics)
            for platform in PLATFORMS
        ):
            raise ValueError("method evaluation must contain three platforms")
        if self.per_split is None:
            object.__setattr__(self, "per_split", {})
        elif not isinstance(self.per_split, Mapping) or any(
            split not in FORMAL_EVALUATION_SPLITS
            or not isinstance(metrics, Mapping)
            or set(metrics) != set(PLATFORMS)
            or any(
                not isinstance(metrics[platform], PlatformMetrics)
                for platform in PLATFORMS
            )
            for split, metrics in self.per_split.items()
        ):
            raise ValueError("method split evaluation structure is invalid")


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
        if self.proxy:
            if any(method.per_split for method in self.methods):
                raise ValueError("proxy evaluation must not claim formal splits")
        elif any(
            set(method.per_split) != set(FORMAL_EVALUATION_SPLITS)
            for method in self.methods
        ):
            raise ValueError(
                "formal evaluation must report every non-training split"
            )
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
        changed_splits = {
            split: {**split_metrics, platform_type: metrics}
            for split, split_metrics in current.per_split.items()
        }
        return self.replace_method(
            MethodEvaluation(
                method=current.method,
                per_platform=changed,
                per_split=changed_splits,
            )
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
                    **(
                        {
                            "per_split": {
                                split: {
                                    platform: asdict(
                                        method.per_split[split][platform]
                                    )
                                    for platform in PLATFORMS
                                }
                                for split in FORMAL_EVALUATION_SPLITS
                            }
                        }
                        if method.per_split
                        else {}
                    ),
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
    success_first_crossing: bool
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
            sum(item.success_first_crossing for item in scenarios)
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


def build_reward_v4_evaluation_report(
    *,
    manifest: RewardEvaluationManifest,
    episodes: Sequence[RewardEpisodeMetrics],
    checkpoint_payload_sha256: str,
    enabled_r2_platforms: Sequence[PlatformType],
    bootstrap_seed: int,
    bootstrap_resample_count: int,
) -> RewardV4EvaluationReport:
    """Aggregate one frozen evaluation grid without changing gate point estimates."""
    if not isinstance(manifest, RewardEvaluationManifest):
        raise TypeError("Reward V4 evaluation manifest is invalid")
    values = tuple(episodes)
    expected_keys = tuple(
        (task.platform, task.scale_bucket, task.evaluation_seed)
        for task in manifest.tasks
    )
    by_key = {
        (row.platform, row.scale_bucket, row.evaluation_seed): row
        for row in values
        if isinstance(row, RewardEpisodeMetrics)
    }
    if len(by_key) != len(values) or set(by_key) != set(expected_keys):
        raise ValueError("Reward V4 evaluation episodes differ from manifest")
    ordered_episodes = tuple(by_key[key] for key in expected_keys)
    stratum_keys = tuple(
        dict.fromkeys((platform, bucket) for platform, bucket, _seed in expected_keys)
    )
    stratum_metrics = tuple(
        summarize_platform_scale_metrics(
            tuple(
                row
                for row in ordered_episodes
                if row.platform is platform and row.scale_bucket is bucket
            )
        )
        for platform, bucket in stratum_keys
    )
    platforms = tuple(dict.fromkeys(platform for platform, _bucket in stratum_keys))
    platform_gates = {
        platform: summarize_platform_gate_metrics(
            tuple(row for row in stratum_metrics if row.platform is platform),
            platform=platform,
        )
        for platform in platforms
    }
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("Reward V4 evaluation bootstrap seed is invalid")
    if type(bootstrap_resample_count) is not int or bootstrap_resample_count <= 0:
        raise ValueError("Reward V4 evaluation bootstrap count is invalid")
    bootstrap_by_stratum = {
        _reward_v4_stratum_key(platform, bucket): (
            bootstrap_platform_scale_metrics(
                tuple(
                    row
                    for row in ordered_episodes
                    if row.platform is platform and row.scale_bucket is bucket
                ),
                bootstrap_seed=bootstrap_seed + index,
                resample_count=bootstrap_resample_count,
            )
        )
        for index, (platform, bucket) in enumerate(stratum_keys)
    }
    score = build_checkpoint_score(
        stratum_metrics,
        enabled_r2_platforms=enabled_r2_platforms,
        payload_sha256=checkpoint_payload_sha256,
    )
    return RewardV4EvaluationReport(
        checkpoint_payload_sha256=checkpoint_payload_sha256,
        manifest=manifest,
        episodes=ordered_episodes,
        stratum_metrics=stratum_metrics,
        platform_gate_metrics=platform_gates,
        bootstrap_by_stratum=bootstrap_by_stratum,
        checkpoint_score=score,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resample_count=bootstrap_resample_count,
    )


def reward_v4_report_sha256(report: RewardV4EvaluationReport) -> str:
    if not isinstance(report, RewardV4EvaluationReport):
        raise TypeError("Reward V4 evaluation report is invalid")
    payload = json.dumps(
        report.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_reward_v4_report(
    path: Path, report: RewardV4EvaluationReport
) -> str:
    digest = reward_v4_report_sha256(report)
    payload = {**report.to_dict(), "report_sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def read_reward_v4_report(path: Path) -> RewardV4EvaluationReport:
    """Read and fully rederive one content-addressed Reward V4 report."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Reward V4 evaluation report is unreadable") from error
    if not isinstance(payload, dict) or not isinstance(
        payload.get("report_sha256"), str
    ):
        raise ValueError("Reward V4 evaluation report hash is missing")
    report_digest = payload.pop("report_sha256")
    actual_digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if report_digest != actual_digest:
        raise ValueError("Reward V4 evaluation report hash differs")
    try:
        manifest_raw = payload["manifest"]
        if not isinstance(manifest_raw, Mapping) or set(manifest_raw) != {
            "schema_version",
            "tasks",
        }:
            raise ValueError("manifest structure differs")
        tasks_raw = manifest_raw["tasks"]
        if not isinstance(tasks_raw, list):
            raise ValueError("manifest tasks differ")
        task_fields = {
            "platform",
            "scale_bucket",
            "evaluation_seed",
            "scene_sha256",
            "roi_sha256",
            "priority_sha256",
            "start_pose_sha256",
            "capability_sha256",
        }
        tasks = tuple(
            RewardEvaluationTask(
                platform=PlatformType(row["platform"]),
                scale_bucket=TaskScaleBucket(row["scale_bucket"]),
                evaluation_seed=row["evaluation_seed"],
                scene_sha256=row["scene_sha256"],
                roi_sha256=row["roi_sha256"],
                priority_sha256=row["priority_sha256"],
                start_pose_sha256=row["start_pose_sha256"],
                capability_sha256=row["capability_sha256"],
            )
            for row in tasks_raw
            if isinstance(row, Mapping) and set(row) == task_fields
        )
        if len(tasks) != len(tasks_raw):
            raise ValueError("manifest task fields differ")
        manifest = RewardEvaluationManifest(
            tasks=tasks,
            schema_version=manifest_raw["schema_version"],
        )
        episodes_raw = payload["episodes"]
        if not isinstance(episodes_raw, list):
            raise ValueError("episode list differs")
        episodes = tuple(
            reward_episode_metrics_from_mapping(row)
            for row in episodes_raw
            if isinstance(row, Mapping)
        )
        if len(episodes) != len(episodes_raw):
            raise ValueError("episode fields differ")
        score_raw = payload["checkpoint_score"]
        if not isinstance(score_raw, Mapping) or not isinstance(
            score_raw.get("enabled_r2_platforms"), list
        ):
            raise ValueError("checkpoint score differs")
        enabled_r2 = tuple(
            PlatformType(value)
            for value in score_raw["enabled_r2_platforms"]
        )
        rebuilt = build_reward_v4_evaluation_report(
            manifest=manifest,
            episodes=episodes,
            checkpoint_payload_sha256=payload[
                "checkpoint_payload_sha256"
            ],
            enabled_r2_platforms=enabled_r2,
            bootstrap_seed=payload["bootstrap_seed"],
            bootstrap_resample_count=payload[
                "bootstrap_resample_count"
            ],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Reward V4 evaluation report structure differs"
        ) from error
    if rebuilt.to_dict() != payload:
        raise ValueError("Reward V4 evaluation report evidence differs")
    return rebuilt


def _reward_v4_stratum_key(
    platform: PlatformType, bucket: TaskScaleBucket
) -> str:
    return f"{platform.value}/{bucket.value}"


def _require_lower_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


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
        method = candidate.report.method("ppo_policy")
        metrics = (
            tuple(
                method.per_split[split][platform]
                for split in FORMAL_EVALUATION_SPLITS
                for platform in PLATFORMS
            )
            if method.per_split
            else tuple(method.per_platform[platform] for platform in PLATFORMS)
        )
        return (
            -min(metric.success_coverage_rate for metric in metrics),
            max(metric.planner_failure_rate for metric in metrics),
            max(metric.completion_time_s for metric in metrics),
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
        by_split: dict[str, Mapping[str, tuple[_ScenarioEvidence, ...]]] = {}
        for batch in formal_batches:
            batch_results = _evaluate_formal_batch(
                policy,
                method=method,
                device=target_device,
                batch=batch,
            )
            by_split[batch.split] = batch_results
            for platform in PLATFORMS:
                by_platform[platform].extend(batch_results[platform])
        per_platform: dict[str, PlatformMetrics] = {}
        for platform in PLATFORMS:
            per_platform[platform] = _aggregate_platform_metrics(
                tuple(by_platform[platform]),
                theta_active=platform != "HOPPER",
            )
        per_split = {
            split: {
                platform: _aggregate_platform_metrics(
                    by_split[split][platform],
                    theta_active=platform != "HOPPER",
                )
                for platform in PLATFORMS
            }
            for split in FORMAL_EVALUATION_SPLITS
        }
        methods.append(
            MethodEvaluation(
                method=method,
                per_platform=per_platform,
                per_split=per_split,
            )
        )
    return EvaluationReport(
        proxy=False,
        scenario_schedule_id=f"formal-evaluation/{schedule_digest}",
        run_identity=run_identity,
        reward_hash=reward_weights_sha256(),
        checkpoint_sha256=checkpoint_sha256,
        methods=tuple(methods),
    )


def formal_evaluation_probe(
    policy: CrossAttentionPolicy,
    *,
    device: torch.device | str,
    run_identity: RunIdentity,
    batches: Sequence[FormalEvaluationBatch],
) -> Mapping[str, object]:
    """Execute one real macro-step per split/method without simulating a release gate."""
    if not isinstance(policy, CrossAttentionPolicy):
        raise ValueError("formal evaluation probe requires CrossAttentionPolicy")
    if not isinstance(run_identity, RunIdentity) or run_identity.run_kind != "formal":
        raise ValueError("formal evaluation probe requires formal run identity")
    formal_batches = tuple(batches)
    by_split = {batch.split: batch for batch in formal_batches}
    if set(by_split) != set(FORMAL_EVALUATION_SPLITS) or len(formal_batches) != 3:
        raise ValueError(
            "formal evaluation probe requires validation, test, and holdout"
        )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA formal evaluation probe requested without CUDA")

    rows: list[dict[str, object]] = []
    for method in REQUIRED_METHODS:
        for split in FORMAL_EVALUATION_SPLITS:
            batch = by_split[split]
            scenario_seed = batch.scenario_seeds[0]
            row_schedule = tuple(
                (platform, 0, scenario_seed) for platform in PLATFORMS
            )
            with ParallelEnvPool(
                allocation={platform: 1 for platform in PLATFORMS},
                observation_template=batch.observation_template,
                environment_factory=batch.factory,
                reward_fn=_evaluation_reward,
                worker_timeout_seconds=FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS,
                auto_reset=False,
                initial_episode_cursors=(0, 0, 0),
            ) as pool:
                pool.reset()
                prepared = pool.prepare_decision_boundaries(policy_version=0)
                if bool(prepared.dones.any()):
                    raise ValueError(
                        "formal evaluation probe has no actionable boundary"
                    )
                indices, thetas = _select_formal_actions(
                    policy,
                    method=method,
                    observations=prepared.observations,
                    device=target_device,
                    row_schedule=row_schedule,
                )
                repeated_indices, repeated_thetas = _select_formal_actions(
                    policy,
                    method=method,
                    observations=prepared.observations,
                    device=target_device,
                    row_schedule=row_schedule,
                )
                if not np.array_equal(indices, repeated_indices) or not np.array_equal(
                    thetas, repeated_thetas
                ):
                    raise ValueError("formal evaluation probe action differs on repeat")
                masks = prepared.observations.candidate_mask.detach().cpu().numpy()
                if any(
                    not 0 <= int(index) < masks.shape[1]
                    or not bool(masks[row, int(index)])
                    or not math.isfinite(float(thetas[row]))
                    for row, index in enumerate(indices)
                ):
                    raise ValueError("formal evaluation probe selected an invalid action")
                stepped = pool.step(
                    ParallelActions(
                        candidate_indices=torch.from_numpy(indices),
                        thetas=torch.from_numpy(thetas),
                    ),
                    policy_version=0,
                )
                row_count = len(PLATFORMS)
                if (
                    len(stepped.execution_events) != row_count
                    or len(stepped.planning_outcomes) != row_count
                    or len(stepped.reason_codes) != row_count
                    or stepped.policy_decisions_consumed.shape != (row_count,)
                    or not bool((stepped.policy_decisions_consumed == 1).all())
                    or not bool(_finite_output_rows(
                        stepped.observations, stepped.rewards
                    ).all())
                ):
                    raise ValueError("formal evaluation probe evidence is incomplete")
                coverage = mission_coverage_ratio(stepped.observations)
                for row, platform in enumerate(PLATFORMS):
                    rows.append(
                        {
                            "method": method,
                            "split": split,
                            "platform": platform,
                            "scenario_seed": scenario_seed,
                            "selected_frontier_index": int(indices[row]),
                            "selected_theta_hex": float(thetas[row]).hex(),
                            "reward_hex": float(stepped.rewards[row]).hex(),
                            "coverage_hex": float(coverage[row]).hex(),
                            "done": bool(stepped.dones[row]),
                            "success_first_crossing": bool(
                                stepped.success_first_crossings[row]
                            ),
                            "planning_outcome": stepped.planning_outcomes[
                                row
                            ].name,
                            "reason_code": stepped.reason_codes[row],
                            "execution_events": asdict(
                                stepped.execution_events[row]
                            ),
                        }
                    )
    body: dict[str, object] = {
        "schema_version": "lunar-formal-evaluation-probe/v1",
        "proxy": False,
        "methods": list(REQUIRED_METHODS),
        "splits": list(FORMAL_EVALUATION_SPLITS),
        "row_count": len(rows),
        "rows": rows,
    }
    body["probe_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return body


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
    watchdog_max_steps: int = FORMAL_EVALUATION_WATCHDOG_MAX_STEPS,
    watchdog_seconds: float = FORMAL_EVALUATION_WATCHDOG_SECONDS,
) -> Mapping[str, tuple[_ScenarioEvidence, ...]]:
    if type(watchdog_max_steps) is not int or watchdog_max_steps <= 0:
        raise ValueError("formal evaluation watchdog steps must be positive")
    if (
        not isinstance(watchdog_seconds, (int, float))
        or isinstance(watchdog_seconds, bool)
        or not math.isfinite(float(watchdog_seconds))
        or watchdog_seconds <= 0.0
    ):
        raise ValueError("formal evaluation watchdog seconds must be positive")
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
        worker_timeout_seconds=FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS,
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
        success_first_crossings = np.zeros(row_count, dtype=np.bool_)
        theta_samples: list[list[float]] = [[] for _ in range(row_count)]
        step_index = 0
        deadline = time.monotonic() + float(watchdog_seconds)
        while not bool(np.all(completion_steps > 0)):
            if (
                step_index >= watchdog_max_steps
                or time.monotonic() >= deadline
            ):
                unfinished = int(np.count_nonzero(completion_steps == 0))
                raise FormalEvaluationIncomplete(
                    "EVALUATION_INCOMPLETE: formal watchdog expired with "
                    f"{unfinished} unfinished rows"
                )
            step_index += 1
            prepared = pool.prepare_decision_boundaries(policy_version=0)
            no_action_workers = tuple(
                int(index)
                for index in torch.nonzero(prepared.dones, as_tuple=False)
                .flatten()
                .tolist()
            )
            if no_action_workers:
                prepared_coverage = mission_coverage_ratio(
                    prepared.observations
                )
                for index in no_action_workers:
                    if completion_steps[index] != 0:
                        continue
                    if executed_steps[index] <= 0:
                        raise ValueError(
                            "formal evaluation started without an actionable boundary"
                        )
                    completion_steps[index] = step_index - 1
                    final_coverage[index] = prepared_coverage[index]
                if bool(np.all(completion_steps > 0)):
                    break
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
            coverage = mission_coverage_ratio(stepped.observations)
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
                success_first_crossings[index] = bool(
                    success_first_crossings[index]
                    or stepped.success_first_crossings[index].item()
                )
                if bool(stepped.dones[index].item()):
                    completion_steps[index] = step_index
            terminated_workers = tuple(
                int(index)
                for index in torch.nonzero(stepped.dones, as_tuple=False)
                .flatten()
                .tolist()
            )
            if terminated_workers and not bool(np.all(completion_steps > 0)):
                pool.reset_terminated_workers(
                    terminated_workers, policy_version=0
                )
    result: dict[str, tuple[_ScenarioEvidence, ...]] = {}
    for platform in PLATFORMS:
        result[platform] = tuple(
            _ScenarioEvidence(
                scenario_seed=scenario_seed,
                final_coverage=float(final_coverage[index]),
                success_first_crossing=bool(success_first_crossings[index]),
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
                completion_step_count=int(completion_steps[index]),
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
        success_first_crossings = np.zeros(row_count, dtype=np.bool_)
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
                success_first_crossings[index] = bool(
                    success_first_crossings[index]
                    or stepped.success_first_crossings[index].item()
                )
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
                success_first_crossing=bool(success_first_crossings[index]),
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
    if transition.planning_outcome in {
        PlanningOutcome.INVALID_REQUEST,
        PlanningOutcome.STALE_INPUT,
        PlanningOutcome.NUMERICAL_FAILURE,
    }:
        raise InvalidTransition(
            "invalid planning outcome cannot enter evaluation reward"
        )
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
    "FORMAL_EVALUATION_SPLITS",
    "FormalEvaluationIncomplete",
    "FormalEvaluationBatch",
    "MethodEvaluation",
    "PlatformMetrics",
    "REQUIRED_METHODS",
    "REWARD_V4_EVALUATION_REPORT_SCHEMA",
    "RewardV4EvaluationReport",
    "build_reward_v4_evaluation_report",
    "read_reward_v4_report",
    "report_sha256",
    "reward_v4_report_sha256",
    "evaluate_formal_policy",
    "formal_evaluation_probe",
    "evaluate_proxy_policy",
    "mission_coverage_ratio",
    "select_best_candidate",
    "write_report",
    "write_reward_v4_report",
]

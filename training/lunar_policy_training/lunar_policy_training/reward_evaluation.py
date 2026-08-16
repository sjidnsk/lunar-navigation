"""Fixed-task Reward V4 evaluation and weakest-stratum checkpoint ranking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

from .eval.metrics_core import priority_coverage_auc_over_macro_actions
from .reward_contract import DEFAULT_REWARD_CONFIG, TaskScaleBucket
from .reward_curriculum import PlatformGateMetrics, PlatformType


REWARD_EVALUATION_MANIFEST_SCHEMA = "lunar-reward-evaluation-manifest/v1"
REWARD_EVALUATION_BOOTSTRAP_SCHEMA = "lunar-reward-evaluation-bootstrap/v1"
_POSITIVE_INFINITY = "POSITIVE_INFINITY"
_NEGATIVE_INFINITY = "NEGATIVE_INFINITY"
_PLATFORM_ORDER = tuple(PlatformType)
_SCALE_ORDER = tuple(TaskScaleBucket)


@dataclass(frozen=True, slots=True)
class RewardEvaluationTask:
    platform: PlatformType
    scale_bucket: TaskScaleBucket
    evaluation_seed: int
    scene_sha256: str
    roi_sha256: str
    priority_sha256: str
    start_pose_sha256: str
    capability_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.platform, PlatformType):
            raise ValueError("evaluation task platform is invalid")
        if not isinstance(self.scale_bucket, TaskScaleBucket):
            raise ValueError("evaluation task scale bucket is invalid")
        if type(self.evaluation_seed) is not int or self.evaluation_seed < 0:
            raise ValueError("evaluation task seed is invalid")
        for name in (
            "scene_sha256",
            "roi_sha256",
            "priority_sha256",
            "start_pose_sha256",
            "capability_sha256",
        ):
            _require_sha256(getattr(self, name), f"evaluation task {name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "platform": self.platform.value,
            "scale_bucket": self.scale_bucket.value,
            "evaluation_seed": self.evaluation_seed,
            "scene_sha256": self.scene_sha256,
            "roi_sha256": self.roi_sha256,
            "priority_sha256": self.priority_sha256,
            "start_pose_sha256": self.start_pose_sha256,
            "capability_sha256": self.capability_sha256,
        }


@dataclass(frozen=True, slots=True)
class RewardEvaluationManifest:
    tasks: tuple[RewardEvaluationTask, ...]
    schema_version: str = REWARD_EVALUATION_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REWARD_EVALUATION_MANIFEST_SCHEMA:
            raise ValueError("Reward V4 evaluation manifest schema is invalid")
        if (
            not isinstance(self.tasks, tuple)
            or not self.tasks
            or any(not isinstance(task, RewardEvaluationTask) for task in self.tasks)
        ):
            raise ValueError("Reward V4 evaluation tasks are invalid")
        keys = tuple(
            (task.platform, task.scale_bucket, task.evaluation_seed)
            for task in self.tasks
        )
        if len(set(keys)) != len(keys):
            raise ValueError("Reward V4 evaluation task identity is duplicated")
        if self.tasks != tuple(sorted(self.tasks, key=_task_order_key)):
            raise ValueError("Reward V4 evaluation tasks are not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tasks": [task.to_dict() for task in self.tasks],
        }


@dataclass(frozen=True, slots=True)
class RewardEpisodeMetrics:
    platform: PlatformType
    scale_bucket: TaskScaleBucket
    evaluation_seed: int
    success_at_0_95: bool
    final_coverage: float
    priority_coverage_auc_over_macro_actions: float | None
    steps_to_success: int | None
    normalized_executed_path_to_success: float | None
    hard_error_count: int
    macro_action_count: int
    priority_denominator_present: bool

    def __post_init__(self) -> None:
        if not isinstance(self.platform, PlatformType) or not isinstance(
            self.scale_bucket, TaskScaleBucket
        ):
            raise ValueError("Reward V4 episode stratum is invalid")
        if type(self.evaluation_seed) is not int or self.evaluation_seed < 0:
            raise ValueError("Reward V4 episode seed is invalid")
        if type(self.success_at_0_95) is not bool:
            raise ValueError("Reward V4 episode success is invalid")
        _require_ratio(self.final_coverage, "episode final coverage")
        if type(self.hard_error_count) is not int or self.hard_error_count < 0:
            raise ValueError("Reward V4 episode hard error count is invalid")
        if type(self.macro_action_count) is not int or self.macro_action_count < 0:
            raise ValueError("Reward V4 episode macro action count is invalid")
        if type(self.priority_denominator_present) is not bool:
            raise ValueError("Reward V4 priority denominator fact is invalid")
        auc = self.priority_coverage_auc_over_macro_actions
        if auc is not None:
            _require_ratio(auc, "episode priority AUC")
            if not self.priority_denominator_present or self.macro_action_count == 0:
                raise ValueError("Reward V4 priority AUC has no valid sample")
        for name in (
            "steps_to_success",
            "normalized_executed_path_to_success",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"Reward V4 episode {name} is invalid")
        if self.steps_to_success is not None and type(self.steps_to_success) is not int:
            raise ValueError("Reward V4 steps to success must be an integer")
        if (self.steps_to_success is None) != (
            self.normalized_executed_path_to_success is None
        ):
            raise ValueError("Reward V4 episode efficiency samples differ")
        if self.steps_to_success is not None and (
            not self.success_at_0_95
            or self.steps_to_success <= 0
            or self.steps_to_success > self.macro_action_count
        ):
            raise ValueError("Reward V4 success step is inconsistent")
        if not self.success_at_0_95 and self.final_coverage >= 0.95:
            raise ValueError("Reward V4 episode success fact differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "platform": self.platform.value,
            "scale_bucket": self.scale_bucket.value,
            "evaluation_seed": self.evaluation_seed,
            "success_at_0_95": self.success_at_0_95,
            "final_coverage": self.final_coverage,
            "priority_coverage_auc_over_macro_actions": (
                self.priority_coverage_auc_over_macro_actions
            ),
            "steps_to_success": self.steps_to_success,
            "normalized_executed_path_to_success": (
                self.normalized_executed_path_to_success
            ),
            "hard_error_count": self.hard_error_count,
            "macro_action_count": self.macro_action_count,
            "priority_denominator_present": (
                self.priority_denominator_present
            ),
        }


@dataclass(frozen=True, slots=True)
class PlatformScaleMetrics:
    platform: PlatformType
    scale_bucket: TaskScaleBucket
    evaluation_seeds: tuple[int, ...]
    episode_count: int
    success_rate: float
    mean_final_coverage: float
    priority_coverage_auc_over_macro_actions: float | None
    priority_eligible_episode_count: int
    priority_auc_sample_count: int
    mean_steps_to_success: float
    mean_normalized_executed_path_to_success: float
    success_sample_count: int
    hard_error_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.platform, PlatformType) or not isinstance(
            self.scale_bucket, TaskScaleBucket
        ):
            raise ValueError("platform-scale metric stratum is invalid")
        if (
            not isinstance(self.evaluation_seeds, tuple)
            or not self.evaluation_seeds
            or any(type(seed) is not int or seed < 0 for seed in self.evaluation_seeds)
            or len(set(self.evaluation_seeds)) != len(self.evaluation_seeds)
        ):
            raise ValueError("platform-scale evaluation seeds are invalid")
        if self.episode_count != len(self.evaluation_seeds):
            raise ValueError("platform-scale episode count differs")
        _require_ratio(self.success_rate, "platform-scale success rate")
        _require_ratio(
            self.mean_final_coverage, "platform-scale mean final coverage"
        )
        for name in (
            "priority_eligible_episode_count",
            "priority_auc_sample_count",
            "success_sample_count",
            "hard_error_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= self.episode_count:
                raise ValueError(f"platform-scale {name} is invalid")
        if not math.isclose(
            self.success_rate,
            self.success_sample_count / self.episode_count,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("platform-scale success count differs")
        if self.priority_auc_sample_count > self.priority_eligible_episode_count:
            raise ValueError("platform-scale priority sample count differs")
        priority_auc = self.priority_coverage_auc_over_macro_actions
        if self.priority_auc_sample_count == 0:
            if priority_auc is not None:
                raise ValueError("platform-scale priority AUC must be absent")
        elif priority_auc is None:
            raise ValueError("platform-scale priority AUC is missing")
        else:
            _require_ratio(priority_auc, "platform-scale priority AUC")
        for name in (
            "mean_steps_to_success",
            "mean_normalized_executed_path_to_success",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or math.isnan(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"platform-scale {name} is invalid")
        if self.success_sample_count == 0 and not (
            math.isinf(self.mean_steps_to_success)
            and math.isinf(self.mean_normalized_executed_path_to_success)
        ):
            raise ValueError("platform-scale missing-success sentinel is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "platform": self.platform.value,
            "scale_bucket": self.scale_bucket.value,
            "evaluation_seeds": list(self.evaluation_seeds),
            "episode_count": self.episode_count,
            "success_rate": self.success_rate,
            "mean_final_coverage": self.mean_final_coverage,
            "priority_coverage_auc_over_macro_actions": (
                self.priority_coverage_auc_over_macro_actions
            ),
            "priority_eligible_episode_count": (
                self.priority_eligible_episode_count
            ),
            "priority_auc_sample_count": self.priority_auc_sample_count,
            "mean_steps_to_success": _encode_infinity(
                self.mean_steps_to_success
            ),
            "mean_normalized_executed_path_to_success": _encode_infinity(
                self.mean_normalized_executed_path_to_success
            ),
            "success_sample_count": self.success_sample_count,
            "hard_error_count": self.hard_error_count,
        }


@dataclass(frozen=True, slots=True)
class CheckpointScore:
    payload_sha256: str
    compared_strata: tuple[str, ...]
    enabled_r2_platforms: tuple[PlatformType, ...]
    priority_comparison_strata: tuple[str, ...]
    hard_error_count: int
    minimum_success_rate: float
    minimum_mean_final_coverage: float
    minimum_priority_auc: float | None
    maximum_steps_to_success: float | None
    maximum_normalized_path_to_success: float | None

    def __post_init__(self) -> None:
        _require_sha256(self.payload_sha256, "checkpoint payload")
        if (
            not isinstance(self.compared_strata, tuple)
            or not self.compared_strata
            or len(set(self.compared_strata)) != len(self.compared_strata)
            or self.compared_strata != tuple(sorted(self.compared_strata))
        ):
            raise ValueError("checkpoint compared strata are invalid")
        if self.enabled_r2_platforms != tuple(
            platform
            for platform in _PLATFORM_ORDER
            if platform in self.enabled_r2_platforms
        ) or len(set(self.enabled_r2_platforms)) != len(self.enabled_r2_platforms):
            raise ValueError("checkpoint R2 platforms are invalid")
        if (
            not isinstance(self.priority_comparison_strata, tuple)
            or len(set(self.priority_comparison_strata))
            != len(self.priority_comparison_strata)
            or self.priority_comparison_strata
            != tuple(sorted(self.priority_comparison_strata))
        ):
            raise ValueError("checkpoint priority comparison strata are invalid")
        if type(self.hard_error_count) is not int or self.hard_error_count < 0:
            raise ValueError("checkpoint hard error count is invalid")
        _require_ratio(self.minimum_success_rate, "checkpoint minimum success")
        _require_ratio(
            self.minimum_mean_final_coverage,
            "checkpoint minimum final coverage",
        )
        if self.enabled_r2_platforms:
            if self.priority_comparison_strata:
                if not _valid_order_metric(self.minimum_priority_auc):
                    raise ValueError("checkpoint minimum priority AUC is invalid")
            elif self.minimum_priority_auc is not None:
                raise ValueError("checkpoint priority AUC must be omitted")
            for value, name in (
                (self.maximum_steps_to_success, "maximum success steps"),
                (
                    self.maximum_normalized_path_to_success,
                    "maximum normalized path",
                ),
            ):
                if not _valid_order_metric(value):
                    raise ValueError(f"checkpoint {name} is invalid")
        elif any(
            value is not None
            for value in (
                self.minimum_priority_auc,
                self.maximum_steps_to_success,
                self.maximum_normalized_path_to_success,
            )
        ):
            raise ValueError("R1 checkpoint must omit efficiency metrics")

    def to_dict(self) -> dict[str, object]:
        return {
            "payload_sha256": self.payload_sha256,
            "compared_strata": list(self.compared_strata),
            "enabled_r2_platforms": [
                platform.value for platform in self.enabled_r2_platforms
            ],
            "priority_comparison_strata": list(
                self.priority_comparison_strata
            ),
            "hard_error_count": self.hard_error_count,
            "minimum_success_rate": self.minimum_success_rate,
            "minimum_mean_final_coverage": self.minimum_mean_final_coverage,
            "minimum_priority_auc": _encode_optional_infinity(
                self.minimum_priority_auc
            ),
            "maximum_steps_to_success": _encode_optional_infinity(
                self.maximum_steps_to_success
            ),
            "maximum_normalized_path_to_success": (
                _encode_optional_infinity(
                    self.maximum_normalized_path_to_success
                )
            ),
        }


def build_reward_v4_evaluation_manifest(
    *,
    platforms: Sequence[PlatformType],
    scale_buckets: Sequence[TaskScaleBucket],
    evaluation_seeds: Sequence[int],
) -> RewardEvaluationManifest:
    """Build the canonical fixed evaluation task grid."""
    platform_values = _ordered_unique_platforms(platforms)
    bucket_values = _ordered_unique_buckets(scale_buckets)
    seeds = tuple(evaluation_seeds)
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Reward V4 evaluation seeds are invalid")
    tasks: list[RewardEvaluationTask] = []
    for platform in platform_values:
        for bucket in bucket_values:
            for seed in sorted(seeds):
                namespace = f"{platform.value}:{bucket.value}:{seed}"
                tasks.append(
                    RewardEvaluationTask(
                        platform=platform,
                        scale_bucket=bucket,
                        evaluation_seed=seed,
                        scene_sha256=_domain_sha256("scene", namespace),
                        roi_sha256=_domain_sha256("roi", namespace),
                        priority_sha256=_domain_sha256("priority", namespace),
                        start_pose_sha256=_domain_sha256("start", namespace),
                        capability_sha256=_domain_sha256("capability", namespace),
                    )
                )
    return RewardEvaluationManifest(tasks=tuple(tasks))


def reward_evaluation_manifest_sha256(
    manifest: RewardEvaluationManifest,
) -> str:
    if not isinstance(manifest, RewardEvaluationManifest):
        raise TypeError("Reward V4 evaluation manifest is invalid")
    return _json_sha256(manifest.to_dict())


def checkpoint_score_sha256(score: CheckpointScore) -> str:
    """Hash the full comparison set and ranking facts for one checkpoint."""
    if not isinstance(score, CheckpointScore):
        raise TypeError("Reward V4 checkpoint score is invalid")
    return _json_sha256(score.to_dict())


def build_reward_episode_metrics(
    *,
    platform: PlatformType,
    scale_bucket: TaskScaleBucket,
    evaluation_seed: int,
    coverage_curve: Sequence[float],
    priority_coverage_curve: Sequence[float] | None,
    cumulative_executed_path_m: Sequence[float],
    task_scale_m: float,
    priority_denominator_present: bool,
    hard_error_count: int,
) -> RewardEpisodeMetrics:
    coverage = _ratio_curve(coverage_curve, "coverage")
    path = _path_curve(cumulative_executed_path_m)
    if len(coverage) != len(path):
        raise ValueError("Reward V4 episode curve lengths differ")
    if (
        not isinstance(task_scale_m, (int, float))
        or isinstance(task_scale_m, bool)
        or not math.isfinite(float(task_scale_m))
        or float(task_scale_m) <= 0.0
    ):
        raise ValueError("Reward V4 task scale is invalid")
    if type(priority_denominator_present) is not bool:
        raise ValueError("Reward V4 priority denominator fact is invalid")
    if priority_denominator_present:
        if priority_coverage_curve is None:
            raise ValueError("Reward V4 priority coverage curve is missing")
        priority = _ratio_curve(priority_coverage_curve, "priority coverage")
        if len(priority) != len(coverage):
            raise ValueError("Reward V4 priority curve length differs")
    else:
        if priority_coverage_curve is not None:
            raise ValueError("Reward V4 priority curve has no denominator")
        priority = None
    first_success = next(
        (
            index
            for index, value in enumerate(coverage)
            if value >= DEFAULT_REWARD_CONFIG.success_threshold
        ),
        None,
    )
    macro_count = len(coverage) - 1
    if first_success == 0 and macro_count != 0:
        raise ValueError("initially successful evaluation consumed an action")
    has_efficiency_sample = first_success is not None and first_success > 0
    priority_auc = priority_coverage_auc_over_macro_actions(
        priority,
        priority_denominator_present=priority_denominator_present,
    )
    return RewardEpisodeMetrics(
        platform=platform,
        scale_bucket=scale_bucket,
        evaluation_seed=evaluation_seed,
        success_at_0_95=first_success is not None,
        final_coverage=coverage[-1],
        priority_coverage_auc_over_macro_actions=priority_auc,
        steps_to_success=(first_success if has_efficiency_sample else None),
        normalized_executed_path_to_success=(
            path[first_success] / float(task_scale_m)
            if has_efficiency_sample
            else None
        ),
        hard_error_count=hard_error_count,
        macro_action_count=macro_count,
        priority_denominator_present=priority_denominator_present,
    )


def summarize_platform_scale_metrics(
    episodes: Sequence[RewardEpisodeMetrics],
) -> PlatformScaleMetrics:
    rows = tuple(episodes)
    if not rows or any(not isinstance(row, RewardEpisodeMetrics) for row in rows):
        raise ValueError("Reward V4 stratum episodes are invalid")
    seeds = tuple(row.evaluation_seed for row in rows)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Reward V4 stratum evaluation seeds are duplicated")
    return _summarize_rows(tuple(sorted(rows, key=lambda row: row.evaluation_seed)))


def summarize_platform_gate_metrics(
    metrics: Sequence[PlatformScaleMetrics],
    *,
    platform: PlatformType,
) -> PlatformGateMetrics:
    """Equal-weight the four frozen scale-bucket point estimates."""
    rows = tuple(metrics)
    if not isinstance(platform, PlatformType) or (
        len(rows) != len(_SCALE_ORDER)
        or any(
            not isinstance(row, PlatformScaleMetrics)
            or row.platform is not platform
            for row in rows
        )
        or {row.scale_bucket for row in rows} != set(_SCALE_ORDER)
    ):
        raise ValueError("Reward V4 platform gate scale grid is incomplete")
    return PlatformGateMetrics(
        success_rate=math.fsum(row.success_rate for row in rows) / len(rows),
        mean_final_coverage=(
            math.fsum(row.mean_final_coverage for row in rows) / len(rows)
        ),
        hard_error_count=sum(row.hard_error_count for row in rows),
    )


def bootstrap_platform_scale_metrics(
    episodes: Sequence[RewardEpisodeMetrics],
    *,
    bootstrap_seed: int,
    resample_count: int,
) -> dict[str, object]:
    rows = tuple(sorted(episodes, key=lambda row: row.evaluation_seed))
    point = summarize_platform_scale_metrics(rows)
    if (
        type(bootstrap_seed) is not int
        or type(resample_count) is not int
        or resample_count <= 0
    ):
        raise ValueError("Reward V4 bootstrap configuration is invalid")
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    indices = rng.integers(
        0,
        len(rows),
        size=(resample_count, len(rows)),
        dtype=np.int64,
    ).astype("<i8", copy=False)
    names = (
        "success_rate",
        "mean_final_coverage",
        "priority_coverage_auc_over_macro_actions",
        "mean_steps_to_success",
        "mean_normalized_executed_path_to_success",
    )
    samples: dict[str, list[float]] = {name: [] for name in names}
    for index_row in indices:
        sample = _summarize_rows(
            tuple(rows[int(index)] for index in index_row),
            evaluation_seeds=point.evaluation_seeds,
        )
        for name in names:
            value = getattr(sample, name)
            if value is not None:
                samples[name].append(float(value))
    intervals: dict[str, dict[str, object]] = {}
    for name in names:
        values = np.asarray(samples[name], dtype=np.float64)
        if values.size == 0:
            intervals[name] = {
                "ci95_low": None,
                "ci95_high": None,
                "valid_resample_count": 0,
            }
            continue
        low, high = np.quantile(
            values,
            (0.025, 0.975),
            method="inverted_cdf",
        )
        intervals[name] = {
            "ci95_low": _encode_infinity(float(low)),
            "ci95_high": _encode_infinity(float(high)),
            "valid_resample_count": int(values.size),
        }
    return {
        "schema_version": REWARD_EVALUATION_BOOTSTRAP_SCHEMA,
        "bootstrap_seed": bootstrap_seed,
        "resample_count": resample_count,
        "episode_order": [row.evaluation_seed for row in rows],
        "sample_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(indices).tobytes(order="C")
        ).hexdigest(),
        "point_estimate": point.to_dict(),
        "metrics": intervals,
    }


def build_checkpoint_score(
    metrics: Sequence[PlatformScaleMetrics],
    *,
    enabled_r2_platforms: Sequence[PlatformType],
    payload_sha256: str,
) -> CheckpointScore:
    rows = tuple(metrics)
    if not rows or any(not isinstance(row, PlatformScaleMetrics) for row in rows):
        raise ValueError("checkpoint platform-scale metrics are invalid")
    keys = tuple(_stratum_key(row.platform, row.scale_bucket) for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError("checkpoint platform-scale metric is duplicated")
    platforms = {row.platform for row in rows}
    if any(
        sum(row.platform is platform for row in rows) != len(_SCALE_ORDER)
        or {row.scale_bucket for row in rows if row.platform is platform}
        != set(_SCALE_ORDER)
        for platform in platforms
    ):
        raise ValueError("checkpoint platform scale grid is incomplete")
    enabled = _ordered_unique_platforms(enabled_r2_platforms, allow_empty=True)
    if not set(enabled) <= platforms:
        raise ValueError("checkpoint enabled R2 platform is absent")
    enabled_rows = tuple(row for row in rows if row.platform in enabled)
    priority_rows = tuple(
        row
        for row in enabled_rows
        if row.priority_auc_sample_count > 0
    )
    minimum_priority = (
        min(
            float(row.priority_coverage_auc_over_macro_actions)
            for row in priority_rows
        )
        if priority_rows
        else None
    )
    return CheckpointScore(
        payload_sha256=payload_sha256,
        compared_strata=tuple(sorted(keys)),
        enabled_r2_platforms=enabled,
        priority_comparison_strata=tuple(
            sorted(
                _stratum_key(row.platform, row.scale_bucket)
                for row in priority_rows
            )
        ),
        hard_error_count=sum(row.hard_error_count for row in rows),
        minimum_success_rate=min(row.success_rate for row in rows),
        minimum_mean_final_coverage=min(
            row.mean_final_coverage for row in rows
        ),
        minimum_priority_auc=minimum_priority,
        maximum_steps_to_success=(
            max(row.mean_steps_to_success for row in enabled_rows)
            if enabled_rows
            else None
        ),
        maximum_normalized_path_to_success=(
            max(
                row.mean_normalized_executed_path_to_success
                for row in enabled_rows
            )
            if enabled_rows
            else None
        ),
    )


def select_checkpoint_lexicographically(
    candidates: Sequence[CheckpointScore],
) -> CheckpointScore:
    values = tuple(candidates)
    if not values or any(not isinstance(item, CheckpointScore) for item in values):
        raise ValueError("checkpoint score candidates are invalid")
    first = values[0]
    for candidate in values[1:]:
        if candidate.compared_strata != first.compared_strata:
            raise ValueError("checkpoint compared strata differ")
        if candidate.enabled_r2_platforms != first.enabled_r2_platforms:
            raise ValueError("checkpoint enabled R2 platforms differ")
        if (
            candidate.priority_comparison_strata
            != first.priority_comparison_strata
        ):
            raise ValueError("checkpoint priority comparison strata differ")
    return min(values, key=_checkpoint_ranking)


def _summarize_rows(
    rows: tuple[RewardEpisodeMetrics, ...],
    *,
    evaluation_seeds: tuple[int, ...] | None = None,
) -> PlatformScaleMetrics:
    platform = rows[0].platform
    bucket = rows[0].scale_bucket
    if any(
        row.platform is not platform or row.scale_bucket is not bucket
        for row in rows
    ):
        raise ValueError("Reward V4 stratum mixes platform or scale")
    priority_values = tuple(
        row.priority_coverage_auc_over_macro_actions
        for row in rows
        if row.priority_coverage_auc_over_macro_actions is not None
    )
    step_values = tuple(
        row.steps_to_success
        for row in rows
        if row.steps_to_success is not None
    )
    path_values = tuple(
        row.normalized_executed_path_to_success
        for row in rows
        if row.normalized_executed_path_to_success is not None
    )
    successful = sum(row.success_at_0_95 for row in rows)
    return PlatformScaleMetrics(
        platform=platform,
        scale_bucket=bucket,
        evaluation_seeds=(
            tuple(sorted(row.evaluation_seed for row in rows))
            if evaluation_seeds is None
            else evaluation_seeds
        ),
        episode_count=len(rows),
        success_rate=successful / len(rows),
        mean_final_coverage=math.fsum(row.final_coverage for row in rows)
        / len(rows),
        priority_coverage_auc_over_macro_actions=(
            math.fsum(priority_values) / len(priority_values)
            if priority_values
            else None
        ),
        priority_eligible_episode_count=sum(
            row.priority_denominator_present for row in rows
        ),
        priority_auc_sample_count=len(priority_values),
        mean_steps_to_success=(
            math.fsum(step_values) / len(step_values)
            if step_values
            else math.inf
        ),
        mean_normalized_executed_path_to_success=(
            math.fsum(path_values) / len(path_values)
            if path_values
            else math.inf
        ),
        success_sample_count=successful,
        hard_error_count=sum(row.hard_error_count for row in rows),
    )


def _checkpoint_ranking(score: CheckpointScore) -> tuple[object, ...]:
    efficiency: tuple[object, ...] = ()
    if score.enabled_r2_platforms:
        efficiency = (
            *(
                (-float(score.minimum_priority_auc),)
                if score.priority_comparison_strata
                else ()
            ),
            float(score.maximum_steps_to_success),
            float(score.maximum_normalized_path_to_success),
        )
    return (
        score.hard_error_count,
        -score.minimum_success_rate,
        -score.minimum_mean_final_coverage,
        *efficiency,
        score.payload_sha256,
    )


def _task_order_key(task: RewardEvaluationTask) -> tuple[int, int, int]:
    return (
        _PLATFORM_ORDER.index(task.platform),
        _SCALE_ORDER.index(task.scale_bucket),
        task.evaluation_seed,
    )


def _stratum_key(platform: PlatformType, bucket: TaskScaleBucket) -> str:
    return f"{platform.value}/{bucket.value}"


def _ordered_unique_platforms(
    values: Sequence[PlatformType], *, allow_empty: bool = False
) -> tuple[PlatformType, ...]:
    items = tuple(values)
    if (
        (not items and not allow_empty)
        or any(not isinstance(item, PlatformType) for item in items)
        or len(set(items)) != len(items)
    ):
        raise ValueError("Reward V4 evaluation platforms are invalid")
    return tuple(platform for platform in _PLATFORM_ORDER if platform in items)


def _ordered_unique_buckets(
    values: Sequence[TaskScaleBucket],
) -> tuple[TaskScaleBucket, ...]:
    items = tuple(values)
    if (
        not items
        or any(not isinstance(item, TaskScaleBucket) for item in items)
        or len(set(items)) != len(items)
    ):
        raise ValueError("Reward V4 evaluation scale buckets are invalid")
    return tuple(bucket for bucket in _SCALE_ORDER if bucket in items)


def _ratio_curve(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"Reward V4 {name} curve is empty")
    for value in result:
        _require_ratio(value, f"Reward V4 {name}")
    if any(right < left for left, right in zip(result, result[1:])):
        raise ValueError(f"Reward V4 {name} curve regresses")
    return result


def _path_curve(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if (
        not result
        or any(not math.isfinite(value) or value < 0.0 for value in result)
        or any(right < left for left, right in zip(result, result[1:]))
    ):
        raise ValueError("Reward V4 executed path curve is invalid")
    return result


def _require_ratio(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} is invalid")
    return float(value)


def _valid_order_metric(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isnan(float(value))
        and float(value) >= 0.0
    )


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _domain_sha256(domain: str, value: str) -> str:
    return hashlib.sha256(
        f"reward-v4-evaluation/v1\0{domain}\0{value}".encode("utf-8")
    ).hexdigest()


def _json_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _encode_infinity(value: float) -> float | str:
    if value == math.inf:
        return _POSITIVE_INFINITY
    if value == -math.inf:
        return _NEGATIVE_INFINITY
    return float(value)


def _encode_optional_infinity(value: float | None) -> float | str | None:
    return None if value is None else _encode_infinity(float(value))


__all__ = [
    "CheckpointScore",
    "PlatformScaleMetrics",
    "REWARD_EVALUATION_BOOTSTRAP_SCHEMA",
    "REWARD_EVALUATION_MANIFEST_SCHEMA",
    "RewardEpisodeMetrics",
    "RewardEvaluationManifest",
    "RewardEvaluationTask",
    "bootstrap_platform_scale_metrics",
    "build_checkpoint_score",
    "build_reward_episode_metrics",
    "build_reward_v4_evaluation_manifest",
    "checkpoint_score_sha256",
    "reward_evaluation_manifest_sha256",
    "select_checkpoint_lexicographically",
    "summarize_platform_gate_metrics",
    "summarize_platform_scale_metrics",
]

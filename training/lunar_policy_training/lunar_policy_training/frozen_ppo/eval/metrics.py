"""Unified Stage 5 episode metrics and fixed-seed episode bootstrap CIs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

from .baselines import ALL_METHODS


EPISODE_FIELDS: Final = (
    "method",
    "scale_profile",
    "scenario_key",
    "scenario_seed",
    "terrain_seed",
    "start_pose_seed",
    "evaluation_seed",
    "success",
    "final_coverage",
    "steps_to_99_success_only",
    "path_length_to_99_success_only",
    "coverage_auc_over_steps",
    "coverage_per_meter",
    "invalid_action_count",
    "planner_failure_count",
    "safety_violation_count",
    "termination_reason",
    "coverage_curve",
)
BOOTSTRAP_METRICS: Final = (
    "success_rate_under_fixed_step_budget",
    "mean_final_coverage",
    "steps_to_99_success_only",
    "path_length_to_99_success_only",
    "coverage_auc_over_steps",
    "coverage_per_meter",
    "invalid_action_count_mean",
    "planner_failure_count_mean",
    "safety_violation_count",
)
ZERO_DISTANCE_POLICY: Final = "zero_when_no_travel/v1"
_TERMINATION_REASONS: Final = frozenset(
    {
        "success_done",
        "failure_done",
        "stagnation_done",
        "no_candidate_done",
        "safety_done",
    }
)


class MetricError(ValueError):
    """Episode or aggregate metrics violate the deterministic Stage 5 contract."""


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    method: str
    scale_profile: str
    scenario_key: str
    scenario_seed: int
    terrain_seed: int
    start_pose_seed: int
    evaluation_seed: int
    success: bool
    final_coverage: float
    steps_to_99_success_only: int | None
    path_length_to_99_success_only: float | None
    coverage_auc_over_steps: float
    coverage_per_meter: float
    invalid_action_count: int
    planner_failure_count: int
    safety_violation_count: int
    termination_reason: str
    coverage_curve: tuple[float, ...]

    def __post_init__(self) -> None:
        validate_episode_result(self)


def build_episode_result(
    *,
    method: str,
    scale_profile: str,
    scenario_key: str,
    scenario_seed: int,
    terrain_seed: int,
    start_pose_seed: int,
    evaluation_seed: int,
    coverage_curve: Sequence[float],
    cumulative_path_length_curve: Sequence[float],
    steps_executed: int,
    invalid_action_count: int,
    planner_failure_count: int,
    safety_violation_count: int,
    termination_reason: str,
    max_steps: int,
    success_threshold: float,
    zero_distance_policy: str,
) -> EpisodeResult:
    """Compute one comparable episode after terminal carry-forward to max_steps."""

    if type(max_steps) is not int or max_steps <= 0:
        raise MetricError("max_steps must be a positive integer")
    if type(steps_executed) is not int or not 0 <= steps_executed <= max_steps:
        raise MetricError("steps_executed is outside the fixed step budget")
    if (
        not isinstance(success_threshold, (int, float))
        or not math.isfinite(float(success_threshold))
        or not 0.0 < float(success_threshold) <= 1.0
    ):
        raise MetricError("success threshold must be finite and in (0, 1]")
    if zero_distance_policy != ZERO_DISTANCE_POLICY:
        raise MetricError("unsupported zero-distance coverage policy")

    coverage = tuple(float(value) for value in coverage_curve)
    path = tuple(float(value) for value in cumulative_path_length_curve)
    if len(coverage) != max_steps + 1 or len(path) != max_steps + 1:
        raise MetricError("coverage and path curves must include step 0 through max_steps")
    if not all(math.isfinite(value) for value in (*coverage, *path)):
        raise MetricError("coverage and path curves must contain finite values")
    if any(not 0.0 <= value <= 1.0 for value in coverage):
        raise MetricError("coverage curve must stay within [0, 1]")
    if any(value < 0.0 for value in path):
        raise MetricError("path curve cannot be negative")
    if any(right < left for left, right in zip(coverage, coverage[1:])):
        raise MetricError("coverage curve must be nondecreasing")
    if any(right < left for left, right in zip(path, path[1:])):
        raise MetricError("path curve must be nondecreasing")
    if any(value != coverage[steps_executed] for value in coverage[steps_executed + 1 :]):
        raise MetricError("terminal coverage must carry forward through max_steps")
    if any(value != path[steps_executed] for value in path[steps_executed + 1 :]):
        raise MetricError("terminal path length must carry forward through max_steps")

    first_success = next(
        (index for index, value in enumerate(coverage) if value >= success_threshold),
        None,
    )
    if first_success is not None and first_success > steps_executed:
        raise MetricError("success cannot occur after the executed terminal step")
    success = first_success is not None
    auc = sum(
        (coverage[index] + coverage[index + 1]) * 0.5
        for index in range(max_steps)
    ) / max_steps
    total_path = path[-1]
    coverage_per_meter = coverage[-1] / total_path if total_path > 0.0 else 0.0
    return EpisodeResult(
        method=method,
        scale_profile=scale_profile,
        scenario_key=scenario_key,
        scenario_seed=scenario_seed,
        terrain_seed=terrain_seed,
        start_pose_seed=start_pose_seed,
        evaluation_seed=evaluation_seed,
        success=success,
        final_coverage=coverage[-1],
        steps_to_99_success_only=first_success if success else None,
        path_length_to_99_success_only=path[first_success] if success else None,
        coverage_auc_over_steps=float(auc),
        coverage_per_meter=float(coverage_per_meter),
        invalid_action_count=invalid_action_count,
        planner_failure_count=planner_failure_count,
        safety_violation_count=safety_violation_count,
        termination_reason=termination_reason,
        coverage_curve=coverage,
    )


def validate_episode_result(result: EpisodeResult) -> None:
    if not isinstance(result, EpisodeResult):
        raise MetricError("episode result must use the shared EpisodeResult schema")
    if result.method not in ALL_METHODS:
        raise MetricError("episode method is not a Stage 5 method")
    if not result.scale_profile or not result.scenario_key:
        raise MetricError("episode scale profile and scenario key must be nonempty")
    for name in ("scenario_seed", "terrain_seed", "start_pose_seed", "evaluation_seed"):
        if type(getattr(result, name)) is not int:
            raise MetricError(f"{name} must be an integer")
    if type(result.success) is not bool:
        raise MetricError("episode success must be boolean")
    finite_values = (
        result.final_coverage,
        result.coverage_auc_over_steps,
        result.coverage_per_meter,
    )
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in finite_values):
        raise MetricError("episode scalar metrics must be finite")
    if not 0.0 <= result.final_coverage <= 1.0:
        raise MetricError("final coverage must stay within [0, 1]")
    if not 0.0 <= result.coverage_auc_over_steps <= 1.0:
        raise MetricError("coverage AUC must stay within [0, 1]")
    if result.coverage_per_meter < 0.0:
        raise MetricError("coverage per meter cannot be negative")
    for name in (
        "invalid_action_count",
        "planner_failure_count",
        "safety_violation_count",
    ):
        value = getattr(result, name)
        if type(value) is not int or value < 0:
            raise MetricError(f"{name} must be a nonnegative integer")
    if result.termination_reason not in _TERMINATION_REASONS:
        raise MetricError("episode termination reason is invalid")
    if not isinstance(result.coverage_curve, tuple) or not result.coverage_curve:
        raise MetricError("coverage curve must be a nonempty tuple")
    if not all(math.isfinite(value) for value in result.coverage_curve):
        raise MetricError("coverage curve values must be finite")
    if any(right < left for left, right in zip(result.coverage_curve, result.coverage_curve[1:])):
        raise MetricError("coverage curve must be nondecreasing")
    if result.coverage_curve[-1] != result.final_coverage:
        raise MetricError("final coverage must equal the carried coverage curve endpoint")
    if result.success:
        if (
            type(result.steps_to_99_success_only) is not int
            or result.steps_to_99_success_only < 0
            or not isinstance(result.path_length_to_99_success_only, (int, float))
            or not math.isfinite(float(result.path_length_to_99_success_only))
            or result.path_length_to_99_success_only < 0.0
            or result.final_coverage < 0.99
        ):
            raise MetricError("success-only metrics are invalid for a successful episode")
    elif (
        result.steps_to_99_success_only is not None
        or result.path_length_to_99_success_only is not None
    ):
        raise MetricError("success-only metrics must be null for an unsuccessful episode")


def episode_record(result: EpisodeResult) -> dict[str, object]:
    validate_episode_result(result)
    return {
        name: (
            list(result.coverage_curve)
            if name == "coverage_curve"
            else getattr(result, name)
        )
        for name in EPISODE_FIELDS
    }


def bootstrap_episode_indices(
    episode_count: int,
    resample_count: int,
    bootstrap_seed: int,
) -> np.ndarray:
    if type(episode_count) is not int or episode_count <= 0:
        raise MetricError("bootstrap episode_count must be positive")
    if type(resample_count) is not int or resample_count <= 0:
        raise MetricError("bootstrap resample_count must be positive")
    if type(bootstrap_seed) is not int:
        raise MetricError("bootstrap seed must be an integer")
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    return rng.integers(
        0,
        episode_count,
        size=(resample_count, episode_count),
        dtype=np.int64,
    ).astype("<i8", copy=False)


def bootstrap_indices_sha256(indices: np.ndarray) -> str:
    array = np.asarray(indices)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise MetricError("bootstrap indices must be a two-dimensional integer matrix")
    canonical = np.ascontiguousarray(array, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def summarize_episodes(
    episodes: Sequence[EpisodeResult],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    rows = tuple(episodes)
    if not rows:
        raise MetricError("at least one episode is required")
    for row in rows:
        validate_episode_result(row)
    methods = {row.method for row in rows}
    scales = {row.scale_profile for row in rows}
    if len(methods) != 1 or len(scales) != 1:
        raise MetricError("one summary cannot mix methods or scale profiles")
    ordered = tuple(sorted(rows, key=_episode_order_key))
    keys = tuple(_episode_order_key(row) for row in ordered)
    if len(set(keys)) != len(keys):
        raise MetricError("episode schedule identities must be unique")

    values = _summary_values(ordered)
    summary: dict[str, object] = {
        "method": ordered[0].method,
        "scale_profile": ordered[0].scale_profile,
        "episode_count": len(ordered),
        **values,
    }
    indices = bootstrap_episode_indices(
        len(ordered),
        bootstrap_resamples,
        bootstrap_seed,
    )
    samples: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRICS}
    for index_row in indices:
        sampled = tuple(ordered[int(index)] for index in index_row)
        sampled_values = _summary_values(sampled)
        for name in BOOTSTRAP_METRICS:
            value = sampled_values[name]
            if value is not None:
                samples[name].append(float(value))

    intervals: dict[str, dict[str, object]] = {}
    for name in BOOTSTRAP_METRICS:
        estimate = values[name]
        metric_samples = np.asarray(samples[name], dtype=np.float64)
        if estimate is None:
            intervals[name] = {
                "estimate": None,
                "ci95_low": None,
                "ci95_high": None,
                "valid_resample_count": 0,
            }
            continue
        if metric_samples.size == 0 or not np.isfinite(metric_samples).all():
            raise MetricError(f"bootstrap metric {name} has no finite resamples")
        low, high = np.quantile(metric_samples, (0.025, 0.975), method="linear")
        intervals[name] = {
            "estimate": estimate,
            "ci95_low": float(low),
            "ci95_high": float(high),
            "valid_resample_count": int(metric_samples.size),
        }

    audit = {
        "schema_version": "fixed_seed_episode_bootstrap_95ci/v1",
        "method": ordered[0].method,
        "scale_profile": ordered[0].scale_profile,
        "sampling_unit": "episode",
        "rng_algorithm": "numpy.random.PCG64",
        "bootstrap_seed": bootstrap_seed,
        "resample_count": bootstrap_resamples,
        "episode_count": len(ordered),
        "sample_index_dtype": "<i8",
        "sample_index_order": "C",
        "sample_index_shape": list(indices.shape),
        "sample_indices_sha256": bootstrap_indices_sha256(indices),
        "episode_order": [
            {
                "scenario_key": row.scenario_key,
                "scenario_seed": row.scenario_seed,
                "terrain_seed": row.terrain_seed,
                "start_pose_seed": row.start_pose_seed,
                "evaluation_seed": row.evaluation_seed,
            }
            for row in ordered
        ],
        "percentiles": [2.5, 97.5],
        "metrics": intervals,
    }
    return summary, audit


def _episode_order_key(result: EpisodeResult) -> tuple[object, ...]:
    return (
        result.scenario_key,
        result.scenario_seed,
        result.terrain_seed,
        result.start_pose_seed,
        result.evaluation_seed,
    )


def _summary_values(episodes: Sequence[EpisodeResult]) -> dict[str, object]:
    success_rows = tuple(row for row in episodes if row.success)
    return {
        "success_rate_under_fixed_step_budget": float(
            np.mean([float(row.success) for row in episodes], dtype=np.float64)
        ),
        "mean_final_coverage": _finite_mean(row.final_coverage for row in episodes),
        "steps_to_99_success_only": (
            _finite_mean(float(row.steps_to_99_success_only) for row in success_rows)
            if success_rows
            else None
        ),
        "path_length_to_99_success_only": (
            _finite_mean(float(row.path_length_to_99_success_only) for row in success_rows)
            if success_rows
            else None
        ),
        "coverage_auc_over_steps": _finite_mean(
            row.coverage_auc_over_steps for row in episodes
        ),
        "coverage_per_meter": _finite_mean(row.coverage_per_meter for row in episodes),
        "invalid_action_count_mean": _finite_mean(
            float(row.invalid_action_count) for row in episodes
        ),
        "planner_failure_count_mean": _finite_mean(
            float(row.planner_failure_count) for row in episodes
        ),
        "safety_violation_count": sum(row.safety_violation_count for row in episodes),
    }


def _finite_mean(values: Sequence[float] | object) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise MetricError("summary inputs must be nonempty and finite")
    return float(np.mean(array, dtype=np.float64))


__all__ = [
    "BOOTSTRAP_METRICS",
    "EPISODE_FIELDS",
    "ZERO_DISTANCE_POLICY",
    "EpisodeResult",
    "MetricError",
    "bootstrap_episode_indices",
    "bootstrap_indices_sha256",
    "build_episode_result",
    "episode_record",
    "summarize_episodes",
    "validate_episode_result",
]

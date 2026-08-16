"""Public evaluation metrics backed by the extracted deterministic core."""

from .metrics_core import (
    BOOTSTRAP_METRICS,
    EPISODE_FIELDS,
    ZERO_DISTANCE_POLICY,
    EpisodeResult,
    MetricError,
    bootstrap_episode_indices,
    bootstrap_indices_sha256,
    build_episode_result,
    episode_record,
    priority_coverage_auc_over_macro_actions,
    summarize_episodes,
    validate_episode_result,
)


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
    "priority_coverage_auc_over_macro_actions",
    "summarize_episodes",
    "validate_episode_result",
]

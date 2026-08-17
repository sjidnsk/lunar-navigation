"""Atomic Reward V4 evaluation decisions at one sealed update boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import math
from pathlib import Path
from types import MappingProxyType

from .checkpoint import (
    TrainingCheckpointV6,
    load_checkpoint,
    save_checkpoint_atomic,
)
from .evaluation.report import (
    RewardV4EvaluationReport,
    read_reward_v4_report,
    reward_v4_report_sha256,
    write_reward_v4_report,
)
from .reward_contract import DEFAULT_REWARD_CONFIG, RewardConfigV4, RewardStage
from .reward_curriculum import (
    AcceptedRewardCheckpoint,
    PlatformType,
    RewardCurriculumState,
    apply_evaluation,
    apply_rollback,
)
from .reward_evaluation import (
    CheckpointScore,
    checkpoint_score_sha256,
    reward_evaluation_manifest_sha256,
    select_checkpoint_lexicographically,
)


BEST_CHECKPOINT_STATE_SCHEMA = "lunar-best-checkpoint-state/v1"
CHECKPOINT_DECISION_SCHEMA = "lunar-reward-checkpoint-decision/v1"
_EVALUATION_FAULTS = frozenset(
    (
        "candidate_before_evaluation_report",
        "evaluation_report_before_checkpoint",
    )
)


class RewardUpdateBoundaryError(RuntimeError):
    """One Reward V4 evaluation boundary is inconsistent or incomplete."""


class InjectedRewardEvaluationFault(RewardUpdateBoundaryError):
    """Test-only crash at a durable candidate/report boundary."""


@dataclass(frozen=True, slots=True)
class RewardUpdateBoundaryDecision:
    curriculum_state: RewardCurriculumState
    curriculum_events: tuple[Mapping[str, object], ...]
    checkpoint_decision: Mapping[str, object] | None
    best_checkpoint_state: Mapping[str, object]
    candidate_checkpoint_gpu_seconds: float
    rollback_checkpoint_path: Path | None

    def __post_init__(self) -> None:
        if not isinstance(self.curriculum_state, RewardCurriculumState):
            raise ValueError("Reward V4 boundary curriculum is invalid")
        if not isinstance(self.curriculum_events, tuple) or any(
            not isinstance(event, Mapping) for event in self.curriculum_events
        ):
            raise ValueError("Reward V4 boundary events are invalid")
        object.__setattr__(
            self,
            "curriculum_events",
            tuple(MappingProxyType(dict(event)) for event in self.curriculum_events),
        )
        if self.checkpoint_decision is not None:
            if not isinstance(self.checkpoint_decision, Mapping):
                raise ValueError("Reward V4 checkpoint decision is invalid")
            object.__setattr__(
                self,
                "checkpoint_decision",
                MappingProxyType(dict(self.checkpoint_decision)),
            )
        if not isinstance(self.best_checkpoint_state, Mapping):
            raise ValueError("Reward V4 best checkpoint state is invalid")
        object.__setattr__(
            self,
            "best_checkpoint_state",
            MappingProxyType(_copy_best_state(self.best_checkpoint_state)),
        )
        if (
            not isinstance(self.candidate_checkpoint_gpu_seconds, (int, float))
            or isinstance(self.candidate_checkpoint_gpu_seconds, bool)
            or not math.isfinite(float(self.candidate_checkpoint_gpu_seconds))
            or float(self.candidate_checkpoint_gpu_seconds) < 0.0
        ):
            raise ValueError("Reward V4 candidate GPU marker is invalid")
        object.__setattr__(
            self,
            "candidate_checkpoint_gpu_seconds",
            float(self.candidate_checkpoint_gpu_seconds),
        )
        if self.rollback_checkpoint_path is not None and (
            not isinstance(self.rollback_checkpoint_path, Path)
            or not self.rollback_checkpoint_path.is_absolute()
        ):
            raise ValueError("Reward V4 rollback checkpoint path is invalid")


def advance_reward_update_boundary(
    state: RewardCurriculumState,
    *,
    update_id: int,
    evaluation_report: RewardV4EvaluationReport | None,
    candidate_checkpoint_path: Path | None,
    candidate_checkpoint_gpu_seconds: float,
    best_checkpoint_state: Mapping[str, object],
    config: RewardConfigV4 = DEFAULT_REWARD_CONFIG,
) -> RewardUpdateBoundaryDecision:
    """Purely derive curriculum, checkpoint selection, and rollback intent."""
    if not isinstance(state, RewardCurriculumState):
        raise TypeError("Reward V4 boundary state is invalid")
    if type(update_id) is not int or update_id <= state.current_update_id:
        raise ValueError("Reward V4 boundary update must be monotonic")
    if not isinstance(config, RewardConfigV4):
        raise TypeError("Reward V4 boundary config is invalid")
    marker = _candidate_gpu_marker(candidate_checkpoint_gpu_seconds)
    best = _validate_best_state(best_checkpoint_state)
    if evaluation_report is None:
        if candidate_checkpoint_path is not None:
            raise ValueError("Reward V4 unevaluated boundary has candidate path")
        return RewardUpdateBoundaryDecision(
            curriculum_state=replace(state, current_update_id=update_id),
            curriculum_events=(),
            checkpoint_decision=None,
            best_checkpoint_state=best,
            candidate_checkpoint_gpu_seconds=marker,
            rollback_checkpoint_path=None,
        )
    if not isinstance(evaluation_report, RewardV4EvaluationReport):
        raise TypeError("Reward V4 boundary report is invalid")
    if (
        not isinstance(candidate_checkpoint_path, Path)
        or not candidate_checkpoint_path.is_absolute()
    ):
        raise ValueError("Reward V4 candidate checkpoint path is invalid")
    score = evaluation_report.checkpoint_score
    if score.payload_sha256 != evaluation_report.checkpoint_payload_sha256:
        raise ValueError("Reward V4 candidate and report identities differ")
    report_platforms = tuple(evaluation_report.platform_gate_metrics)
    expected_r2 = tuple(
        platform
        for platform in report_platforms
        if state.platforms[platform].stage is RewardStage.R2
    )
    if score.enabled_r2_platforms != expected_r2:
        raise ValueError("Reward V4 report R2 platform set differs")

    evaluated = apply_evaluation(
        state,
        metrics_by_platform=evaluation_report.platform_gate_metrics,
        update_id=update_id,
        config=config,
    )
    report_sha = reward_v4_report_sha256(evaluation_report)
    manifest_sha = reward_evaluation_manifest_sha256(
        evaluation_report.manifest
    )
    candidate = {
        "candidate_checkpoint_path": str(candidate_checkpoint_path),
        "payload_sha256": score.payload_sha256,
        "checkpoint_score": score.to_dict(),
        "checkpoint_score_sha256": checkpoint_score_sha256(score),
        "evaluation_report_sha256": report_sha,
        "evaluation_manifest_sha256": manifest_sha,
        "accepted_update_id": update_id,
        "curriculum_state": evaluated.to_dict(),
    }
    accepted = list(best["accepted"])
    compatible_indexes = tuple(
        index
        for index, item in enumerate(accepted)
        if _score_identity(_score_from_mapping(item["checkpoint_score"]))
        == _score_identity(score)
    )
    accepted_candidate = False
    if not evaluated.pending_rollback_platforms:
        if compatible_indexes:
            current_index = max(
                compatible_indexes,
                key=lambda index: accepted[index]["accepted_update_id"],
            )
            selected = select_checkpoint_lexicographically(
                (_score_from_mapping(accepted[current_index]["checkpoint_score"]), score)
            )
            if selected.payload_sha256 == score.payload_sha256:
                accepted[current_index] = candidate
                accepted_candidate = True
            else:
                accepted[current_index]["accepted_update_id"] = update_id
                accepted[current_index]["curriculum_state"] = (
                    evaluated.to_dict()
                )
        else:
            accepted.append(candidate)
            accepted_candidate = True

    rollback_path: Path | None = None
    final_curriculum = evaluated
    if evaluated.pending_rollback_platforms:
        if not accepted:
            raise RewardUpdateBoundaryError(
                "Reward V4 rollback has no accepted checkpoint"
            )
        rollback = max(
            accepted,
            key=lambda item: item["accepted_update_id"],
        )
        rollback_path = Path(rollback["candidate_checkpoint_path"])
        accepted_curriculum = RewardCurriculumState.from_mapping(
            rollback["curriculum_state"]
        )
        final_curriculum = apply_rollback(
            evaluated,
            triggers=evaluated.pending_rollback_platforms,
            checkpoint=AcceptedRewardCheckpoint(
                update_id=accepted_curriculum.current_update_id,
                payload_sha256=rollback["payload_sha256"],
                curriculum_state=accepted_curriculum,
            ),
            current_update_id=update_id,
            config=config,
        )

    events: list[dict[str, object]] = [
        {
            "event": "EVALUATION_APPLIED",
            "update_id": update_id,
            "report_sha256": report_sha,
            "manifest_sha256": manifest_sha,
        },
        {
            "event": (
                "CHECKPOINT_ACCEPTED" if accepted_candidate else "CHECKPOINT_RETAINED"
            ),
            "update_id": update_id,
            "candidate_payload_sha256": score.payload_sha256,
        },
    ]
    if evaluated.stage is not state.stage:
        events.append(
            {
                "event": "STAGE_CHANGED",
                "update_id": update_id,
                "from_stage": state.stage.value,
                "to_stage": evaluated.stage.value,
            }
        )
    if rollback_path is not None:
        events.append(
            {
                "event": "ROLLBACK_APPLIED",
                "update_id": update_id,
                "checkpoint_path": str(rollback_path),
                "triggers": [
                    platform.value
                    for platform in evaluated.pending_rollback_platforms
                ],
            }
        )
    best_result = {
        "schema_version": BEST_CHECKPOINT_STATE_SCHEMA,
        "accepted": sorted(
            accepted,
            key=lambda item: (
                item["accepted_update_id"], item["payload_sha256"]
            ),
        ),
    }
    checkpoint_decision = {
        "schema_version": CHECKPOINT_DECISION_SCHEMA,
        "update_id": update_id,
        "candidate_checkpoint_path": str(candidate_checkpoint_path),
        "candidate_payload_sha256": score.payload_sha256,
        "candidate_score_sha256": checkpoint_score_sha256(score),
        "evaluation_report_sha256": report_sha,
        "evaluation_manifest_sha256": manifest_sha,
        "accepted": accepted_candidate,
        "rollback_checkpoint_path": (
            None if rollback_path is None else str(rollback_path)
        ),
    }
    return RewardUpdateBoundaryDecision(
        curriculum_state=final_curriculum,
        curriculum_events=tuple(events),
        checkpoint_decision=checkpoint_decision,
        best_checkpoint_state=best_result,
        candidate_checkpoint_gpu_seconds=marker,
        rollback_checkpoint_path=rollback_path,
    )


def advance_reward_sentinel_update_boundary(
    state: RewardCurriculumState,
    *,
    update_id: int,
    evaluation_report: RewardV4EvaluationReport,
    candidate_checkpoint_path: Path,
    candidate_checkpoint_gpu_seconds: float,
    best_checkpoint_state: Mapping[str, object],
    config: RewardConfigV4 = DEFAULT_REWARD_CONFIG,
) -> RewardUpdateBoundaryDecision:
    """Apply a hard-error sentinel without changing curriculum or ranking."""
    if not isinstance(evaluation_report, RewardV4EvaluationReport):
        raise TypeError("Reward V4 sentinel report is invalid")
    if (
        not isinstance(candidate_checkpoint_path, Path)
        or not candidate_checkpoint_path.is_absolute()
    ):
        raise ValueError("Reward V4 sentinel candidate path is invalid")
    score = evaluation_report.checkpoint_score
    if score.payload_sha256 != evaluation_report.checkpoint_payload_sha256:
        raise ValueError("Reward V4 sentinel candidate identity differs")
    report_platforms = tuple(evaluation_report.platform_gate_metrics)
    expected_r2 = tuple(
        platform
        for platform in report_platforms
        if state.platforms[platform].stage is RewardStage.R2
    )
    if score.enabled_r2_platforms != expected_r2:
        raise ValueError("Reward V4 sentinel R2 platform set differs")
    if score.hard_error_count != 0:
        raise RewardUpdateBoundaryError(
            "Reward V4 sentinel evaluation contains a hard error"
        )
    boundary = advance_reward_update_boundary(
        state,
        update_id=update_id,
        evaluation_report=None,
        candidate_checkpoint_path=None,
        candidate_checkpoint_gpu_seconds=candidate_checkpoint_gpu_seconds,
        best_checkpoint_state=best_checkpoint_state,
        config=config,
    )
    return RewardUpdateBoundaryDecision(
        curriculum_state=boundary.curriculum_state,
        curriculum_events=(
            {
                "event": "SENTINEL_EVALUATION_PASSED",
                "update_id": update_id,
                "candidate_checkpoint_path": str(
                    candidate_checkpoint_path
                ),
                "candidate_payload_sha256": score.payload_sha256,
                "report_sha256": reward_v4_report_sha256(
                    evaluation_report
                ),
                "manifest_sha256": reward_evaluation_manifest_sha256(
                    evaluation_report.manifest
                ),
            },
        ),
        checkpoint_decision=None,
        best_checkpoint_state=boundary.best_checkpoint_state,
        candidate_checkpoint_gpu_seconds=(
            boundary.candidate_checkpoint_gpu_seconds
        ),
        rollback_checkpoint_path=None,
    )


def materialize_reward_evaluation_artifacts(
    *,
    candidate_checkpoint_path: Path,
    evaluation_report_path: Path,
    build_candidate_checkpoint: Callable[[], TrainingCheckpointV6],
    evaluate_candidate: Callable[[TrainingCheckpointV6], RewardV4EvaluationReport],
    fault_injection: str | None = None,
) -> tuple[TrainingCheckpointV6, RewardV4EvaluationReport]:
    """Create or verify one immutable candidate/report pair idempotently."""
    for path, name in (
        (candidate_checkpoint_path, "candidate checkpoint"),
        (evaluation_report_path, "evaluation report"),
    ):
        if not isinstance(path, Path) or not path.is_absolute():
            raise RewardUpdateBoundaryError(f"Reward V4 {name} path is invalid")
    if candidate_checkpoint_path == evaluation_report_path:
        raise RewardUpdateBoundaryError("Reward V4 artifact paths collide")
    if not callable(build_candidate_checkpoint) or not callable(evaluate_candidate):
        raise RewardUpdateBoundaryError("Reward V4 artifact callback is invalid")
    if fault_injection is not None and fault_injection not in _EVALUATION_FAULTS:
        raise RewardUpdateBoundaryError("Reward V4 artifact fault is invalid")

    if candidate_checkpoint_path.exists():
        candidate = load_checkpoint(candidate_checkpoint_path)
        if not isinstance(candidate, TrainingCheckpointV6):
            raise RewardUpdateBoundaryError(
                "Reward V4 candidate checkpoint is not resumable"
            )
    else:
        candidate = build_candidate_checkpoint()
        if not isinstance(candidate, TrainingCheckpointV6):
            raise RewardUpdateBoundaryError(
                "Reward V4 candidate builder returned invalid state"
            )
        candidate_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint_atomic(
            candidate_checkpoint_path,
            candidate,
            overwrite=False,
        )
    if fault_injection == "candidate_before_evaluation_report":
        raise InjectedRewardEvaluationFault(fault_injection)

    if evaluation_report_path.exists():
        report = read_reward_v4_report(evaluation_report_path)
    else:
        report = evaluate_candidate(candidate)
        if not isinstance(report, RewardV4EvaluationReport):
            raise RewardUpdateBoundaryError(
                "Reward V4 evaluator returned invalid report"
            )
        write_reward_v4_report(evaluation_report_path, report)
    if report.checkpoint_payload_sha256 != candidate.payload_sha256:
        raise RewardUpdateBoundaryError(
            "Reward V4 report checkpoint identity differs"
        )
    if fault_injection == "evaluation_report_before_checkpoint":
        raise InjectedRewardEvaluationFault(fault_injection)
    return candidate, report


def _candidate_gpu_marker(value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("Reward V4 candidate GPU marker is invalid")
    return float(value)


def _validate_best_state(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "accepted",
    }:
        raise ValueError("Reward V4 best checkpoint structure is invalid")
    if value["schema_version"] != BEST_CHECKPOINT_STATE_SCHEMA or not isinstance(
        value["accepted"], list
    ):
        raise ValueError("Reward V4 best checkpoint state is invalid")
    result = _copy_best_state(value)
    seen: set[str] = set()
    for item in result["accepted"]:
        required = {
            "candidate_checkpoint_path",
            "payload_sha256",
            "checkpoint_score",
            "checkpoint_score_sha256",
            "evaluation_report_sha256",
            "evaluation_manifest_sha256",
            "accepted_update_id",
            "curriculum_state",
        }
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("Reward V4 accepted checkpoint structure is invalid")
        path = Path(item["candidate_checkpoint_path"])
        score = _score_from_mapping(item["checkpoint_score"])
        curriculum = RewardCurriculumState.from_mapping(item["curriculum_state"])
        if (
            not path.is_absolute()
            or score.payload_sha256 != item["payload_sha256"]
            or checkpoint_score_sha256(score) != item["checkpoint_score_sha256"]
            or not _is_sha256(item["evaluation_report_sha256"])
            or not _is_sha256(item["evaluation_manifest_sha256"])
            or type(item["accepted_update_id"]) is not int
            or item["accepted_update_id"] <= 0
            or curriculum.current_update_id != item["accepted_update_id"]
            or item["payload_sha256"] in seen
        ):
            raise ValueError("Reward V4 accepted checkpoint identity is invalid")
        seen.add(item["payload_sha256"])
    return result


def _copy_best_state(value: Mapping[str, object]) -> dict[str, object]:
    accepted = value.get("accepted")
    if not isinstance(accepted, list):
        raise ValueError("Reward V4 accepted checkpoint list is invalid")
    return {
        "schema_version": value.get("schema_version"),
        "accepted": [
            {
                **dict(item),
                "checkpoint_score": dict(item["checkpoint_score"]),
                "curriculum_state": {
                    **dict(item["curriculum_state"]),
                    "platforms": {
                        key: dict(platform)
                        for key, platform in item["curriculum_state"][
                            "platforms"
                        ].items()
                    },
                    "pending_rollback_platforms": list(
                        item["curriculum_state"]["pending_rollback_platforms"]
                    ),
                },
            }
            for item in accepted
        ],
    }


def _score_from_mapping(value: object) -> CheckpointScore:
    if not isinstance(value, Mapping) or set(value) != {
        "payload_sha256",
        "compared_strata",
        "enabled_r2_platforms",
        "priority_comparison_strata",
        "hard_error_count",
        "minimum_success_rate",
        "minimum_mean_final_coverage",
        "minimum_priority_auc",
        "maximum_steps_to_success",
        "maximum_normalized_path_to_success",
    }:
        raise ValueError("Reward V4 checkpoint score structure is invalid")
    return CheckpointScore(
        payload_sha256=value["payload_sha256"],
        compared_strata=tuple(value["compared_strata"]),
        enabled_r2_platforms=tuple(
            PlatformType(item) for item in value["enabled_r2_platforms"]
        ),
        priority_comparison_strata=tuple(value["priority_comparison_strata"]),
        hard_error_count=value["hard_error_count"],
        minimum_success_rate=value["minimum_success_rate"],
        minimum_mean_final_coverage=value["minimum_mean_final_coverage"],
        minimum_priority_auc=_decode_metric(value["minimum_priority_auc"]),
        maximum_steps_to_success=_decode_metric(
            value["maximum_steps_to_success"]
        ),
        maximum_normalized_path_to_success=_decode_metric(
            value["maximum_normalized_path_to_success"]
        ),
    )


def _decode_metric(value: object) -> float | None:
    if value is None:
        return None
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    return value


def _score_identity(score: CheckpointScore) -> tuple[object, ...]:
    return (
        score.compared_strata,
        score.enabled_r2_platforms,
        score.priority_comparison_strata,
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "BEST_CHECKPOINT_STATE_SCHEMA",
    "CHECKPOINT_DECISION_SCHEMA",
    "InjectedRewardEvaluationFault",
    "RewardUpdateBoundaryDecision",
    "RewardUpdateBoundaryError",
    "advance_reward_update_boundary",
    "materialize_reward_evaluation_artifacts",
]

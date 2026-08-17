"""Durable two-tier scheduling for Reward V4 checkpoint evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping


REWARD_V4_FULL_EVALUATION_INTERVAL_GPU_SECONDS = 43_200.0
REWARD_V4_SENTINEL_MACRO_ACTIONS_PER_TASK = 1
REWARD_V4_EVALUATION_MODE_SCHEMA = "lunar-reward-v4-evaluation-mode/v1"


class RewardV4EvaluationTier(str, Enum):
    SENTINEL = "sentinel"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class RewardV4EvaluationMode:
    tier: RewardV4EvaluationTier
    checkpoint_payload_sha256: str
    previous_candidate_gpu_seconds: float
    candidate_gpu_seconds: float
    schema_version: str = REWARD_V4_EVALUATION_MODE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REWARD_V4_EVALUATION_MODE_SCHEMA:
            raise ValueError("Reward V4 evaluation mode schema differs")
        _require_sha256(self.checkpoint_payload_sha256)
        previous = _require_gpu_seconds(
            self.previous_candidate_gpu_seconds,
            "previous candidate",
        )
        candidate = _require_gpu_seconds(
            self.candidate_gpu_seconds,
            "candidate",
        )
        if candidate < previous:
            raise ValueError("Reward V4 evaluation GPU boundary regressed")
        object.__setattr__(self, "previous_candidate_gpu_seconds", previous)
        object.__setattr__(self, "candidate_gpu_seconds", candidate)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tier": self.tier.value,
            "checkpoint_payload_sha256": self.checkpoint_payload_sha256,
            "previous_candidate_gpu_seconds": (
                self.previous_candidate_gpu_seconds
            ),
            "candidate_gpu_seconds": self.candidate_gpu_seconds,
        }


def select_reward_v4_evaluation_tier(
    *,
    previous_candidate_gpu_seconds: float,
    candidate_gpu_seconds: float,
    full_interval_gpu_seconds: float = (
        REWARD_V4_FULL_EVALUATION_INTERVAL_GPU_SECONDS
    ),
) -> RewardV4EvaluationTier:
    """Choose full only when this candidate crosses an absolute full interval."""
    previous = _require_gpu_seconds(
        previous_candidate_gpu_seconds,
        "previous candidate",
    )
    candidate = _require_gpu_seconds(candidate_gpu_seconds, "candidate")
    interval = _require_gpu_seconds(full_interval_gpu_seconds, "full interval")
    if interval <= 0.0:
        raise ValueError("Reward V4 full evaluation interval must be positive")
    if candidate < previous:
        raise ValueError("Reward V4 evaluation GPU boundary regressed")
    if math.floor(candidate / interval) > math.floor(previous / interval):
        return RewardV4EvaluationTier.FULL
    return RewardV4EvaluationTier.SENTINEL


def materialize_reward_v4_evaluation_mode(
    path: Path,
    *,
    checkpoint_payload_sha256: str,
    previous_candidate_gpu_seconds: float,
    candidate_gpu_seconds: float,
) -> RewardV4EvaluationMode:
    """Create or verify one checkpoint-bound evaluation-mode artifact."""
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("Reward V4 evaluation mode path is invalid")
    expected = RewardV4EvaluationMode(
        tier=select_reward_v4_evaluation_tier(
            previous_candidate_gpu_seconds=previous_candidate_gpu_seconds,
            candidate_gpu_seconds=candidate_gpu_seconds,
        ),
        checkpoint_payload_sha256=checkpoint_payload_sha256,
        previous_candidate_gpu_seconds=previous_candidate_gpu_seconds,
        candidate_gpu_seconds=candidate_gpu_seconds,
    )
    if path.exists():
        actual = _read_reward_v4_evaluation_mode(path)
        if actual.checkpoint_payload_sha256 != checkpoint_payload_sha256:
            raise ValueError(
                "Reward V4 evaluation mode checkpoint identity differs"
            )
        if actual != expected:
            raise ValueError("Reward V4 evaluation mode boundary differs")
        return actual
    _write_reward_v4_evaluation_mode(path, expected)
    return expected


def _write_reward_v4_evaluation_mode(
    path: Path, mode: RewardV4EvaluationMode
) -> None:
    body = mode.to_dict()
    payload = {**body, "mode_sha256": _payload_sha256(body)}
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


def _read_reward_v4_evaluation_mode(path: Path) -> RewardV4EvaluationMode:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Reward V4 evaluation mode is unreadable") from error
    fields = {
        "schema_version",
        "tier",
        "checkpoint_payload_sha256",
        "previous_candidate_gpu_seconds",
        "candidate_gpu_seconds",
        "mode_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("Reward V4 evaluation mode structure differs")
    body = {key: payload[key] for key in fields if key != "mode_sha256"}
    if payload["mode_sha256"] != _payload_sha256(body):
        raise ValueError("Reward V4 evaluation mode hash differs")
    try:
        return RewardV4EvaluationMode(
            tier=RewardV4EvaluationTier(body["tier"]),
            checkpoint_payload_sha256=body[
                "checkpoint_payload_sha256"
            ],
            previous_candidate_gpu_seconds=body[
                "previous_candidate_gpu_seconds"
            ],
            candidate_gpu_seconds=body["candidate_gpu_seconds"],
            schema_version=body["schema_version"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Reward V4 evaluation mode structure differs"
        ) from error


def _payload_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("Reward V4 evaluation checkpoint identity is invalid")
    return value


def _require_gpu_seconds(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"Reward V4 {name} GPU seconds are invalid")
    return float(value)


__all__ = (
    "REWARD_V4_FULL_EVALUATION_INTERVAL_GPU_SECONDS",
    "REWARD_V4_SENTINEL_MACRO_ACTIONS_PER_TASK",
    "RewardV4EvaluationMode",
    "RewardV4EvaluationTier",
    "materialize_reward_v4_evaluation_mode",
    "select_reward_v4_evaluation_tier",
)

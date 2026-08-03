"""Stage 4 rollout snapshot、transition、buffer 与 GAE 合同。"""

from __future__ import annotations
from typing import TYPE_CHECKING

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

import numpy as np

from ..policy.observation import PolicyObservation
if TYPE_CHECKING:
    from typing import Any as ArtifactStore
if TYPE_CHECKING:
    from typing import Any as CellXY


SNAPSHOT_SCHEMA_VERSION: Final = "rollout_candidate_snapshot/v1"
_SNAPSHOT_MAGIC: Final = b"LPPSNAP1"
_ARRAY_FIELDS: Final = (
    "prior_channels",
    "coverage_summary",
    "local_crop",
    "frontier_features",
    "pose_features",
    "candidate_mask",
)
_FORBIDDEN_SNAPSHOT_KEYS: Final = frozenset(
    {"truth", "coverable_mask", "dense_highres_truth"}
)
_DONE_REASONS: Final = frozenset(
    {
        "success_done",
        "failure_done",
        "stagnation_done",
        "no_candidate_done",
        "safety_done",
        "none",
    }
)
GAMMA: Final = np.float32(0.995)
GAE_LAMBDA: Final = np.float32(0.95)
ROLLOUT_STEPS_PER_ENV: Final = 128
ROLLOUT_ENV_COUNT: Final = 8
ROLLOUT_BATCH_SIZE: Final = ROLLOUT_STEPS_PER_ENV * ROLLOUT_ENV_COUNT
_TRANSITION_RECORD_FIELDS: Final = frozenset(
    {
        "candidate_snapshot_sha256",
        "policy_state_sha256",
        "config_sha256",
        "lineage",
        "selected_frontier_index",
        "selected_cell_xy",
        "selected_theta",
        "old_log_prob_frontier",
        "old_log_prob_theta",
        "old_log_prob_total",
        "old_value",
        "reward",
        "done",
        "done_reason",
        "planned_path_cells",
        "path_length_m",
        "path_observation_step_m",
        "coverage_gain_cells",
        "coverage_gain_per_meter",
        "observation_sample_count",
        "ray_count_per_sample",
        "ray_cell_visit_count",
        "newly_observed_cell_count",
        "planner_validation_summary",
        "execution_observation_summary",
        "frontier_extractor_version",
        "observation_schema_version",
        "action_space_version",
        "reward_version",
        "planner_version",
    }
)


class RolloutContractError(ValueError):
    """Rollout bytes or metadata violate the frozen Stage 4 contract."""


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    prior_channels: np.ndarray
    coverage_summary: np.ndarray
    local_crop: np.ndarray
    frontier_features: np.ndarray
    pose_features: np.ndarray
    candidate_mask: np.ndarray
    frontier_cells: tuple[CellXY, ...]
    top_m_selection_summary: Mapping[str, object]

    def __post_init__(self) -> None:
        arrays = self.array_fields()
        for array in arrays:
            array.setflags(write=False)
        if self.candidate_mask.dtype != np.bool_:
            raise RolloutContractError("candidate snapshot mask must be boolean")
        float_arrays = arrays[:-1]
        if any(array.dtype != np.float32 for array in float_arrays):
            raise RolloutContractError("candidate snapshot float arrays must be FP32")
        if any(not np.isfinite(array).all() for array in float_arrays):
            raise RolloutContractError("candidate snapshot arrays must be finite")
        if (
            self.frontier_features.ndim != 2
            or self.frontier_features.shape[1:] != (22,)
            or self.candidate_mask.shape != self.frontier_features.shape[:1]
        ):
            raise RolloutContractError("candidate snapshot frontier shape mismatch")
        valid_count = int(np.count_nonzero(self.candidate_mask))
        if valid_count == 0 or len(self.frontier_cells) != valid_count:
            raise RolloutContractError("candidate snapshot cell/mask count mismatch")
        _reject_forbidden_snapshot_content(self.top_m_selection_summary)

    @classmethod
    def capture(
        cls,
        *,
        observation: PolicyObservation,
        frontier_cells: tuple[CellXY, ...],
        top_m_selection_summary: Mapping[str, object],
    ) -> "CandidateSnapshot":
        if not isinstance(observation, PolicyObservation):
            raise RolloutContractError("snapshot requires a PolicyObservation")
        if not isinstance(top_m_selection_summary, Mapping):
            raise RolloutContractError("top-M selection summary must be a mapping")
        summary = _copy_json_mapping(top_m_selection_summary)
        return cls(
            prior_channels=np.asarray(observation.prior_channels, dtype="<f4").copy(),
            coverage_summary=np.asarray(
                observation.coverage_summary, dtype="<f4"
            ).copy(),
            local_crop=np.asarray(observation.local_crop, dtype="<f4").copy(),
            frontier_features=np.asarray(
                observation.frontier_features, dtype="<f4"
            ).copy(),
            pose_features=np.asarray(observation.pose_features, dtype="<f4").copy(),
            candidate_mask=np.asarray(observation.candidate_mask, dtype=np.bool_).copy(),
            frontier_cells=tuple(frontier_cells),
            top_m_selection_summary=MappingProxyType(summary),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def array_fields(self) -> tuple[np.ndarray, ...]:
        return tuple(getattr(self, name) for name in _ARRAY_FIELDS)

    def to_policy_observation(self) -> PolicyObservation:
        return PolicyObservation(
            prior_channels=self.prior_channels.copy(),
            coverage_summary=self.coverage_summary.copy(),
            local_crop=self.local_crop.copy(),
            frontier_features=self.frontier_features.copy(),
            pose_features=self.pose_features.copy(),
            candidate_mask=self.candidate_mask.copy(),
        )

    def to_bytes(self) -> bytes:
        array_records: list[dict[str, object]] = []
        payloads: list[bytes] = []
        for name, array in zip(_ARRAY_FIELDS, self.array_fields(), strict=True):
            canonical = np.ascontiguousarray(array)
            payload = canonical.tobytes(order="C")
            payloads.append(payload)
            array_records.append(
                {
                    "name": name,
                    "dtype": canonical.dtype.str,
                    "shape": list(canonical.shape),
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        header = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "arrays": array_records,
            "frontier_cells": [[cell.x, cell.y] for cell in self.frontier_cells],
            "top_m_selection_summary": dict(self.top_m_selection_summary),
        }
        header_bytes = ArtifactStore.canonical_json_bytes(header)
        return b"".join(
            (
                _SNAPSHOT_MAGIC,
                struct.pack("<Q", len(header_bytes)),
                header_bytes,
                *payloads,
            )
        )

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        expected_sha256: str | None = None,
    ) -> "CandidateSnapshot":
        if not isinstance(payload, bytes) or len(payload) < len(_SNAPSHOT_MAGIC) + 8:
            raise RolloutContractError("candidate snapshot payload is truncated")
        actual_hash = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and actual_hash != expected_sha256:
            raise RolloutContractError("candidate snapshot hash mismatch")
        if not payload.startswith(_SNAPSHOT_MAGIC):
            raise RolloutContractError("candidate snapshot payload magic mismatch")
        header_start = len(_SNAPSHOT_MAGIC) + 8
        header_size = struct.unpack(
            "<Q", payload[len(_SNAPSHOT_MAGIC) : header_start]
        )[0]
        header_end = header_start + header_size
        if header_end > len(payload):
            raise RolloutContractError("candidate snapshot payload header is truncated")
        header_bytes = payload[header_start:header_end]
        try:
            header = json.loads(header_bytes.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RolloutContractError("candidate snapshot payload header is invalid") from exc
        if not isinstance(header, dict) or header_bytes != ArtifactStore.canonical_json_bytes(
            header
        ):
            raise RolloutContractError("candidate snapshot payload header is not canonical")
        if set(header) != {
            "schema_version",
            "arrays",
            "frontier_cells",
            "top_m_selection_summary",
        } or header.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise RolloutContractError("candidate snapshot payload schema drift")
        records = header.get("arrays")
        if not isinstance(records, list) or [
            record.get("name") if isinstance(record, dict) else None for record in records
        ] != list(_ARRAY_FIELDS):
            raise RolloutContractError("candidate snapshot payload array set drift")
        offset = header_end
        arrays: dict[str, np.ndarray] = {}
        for record in records:
            if not isinstance(record, dict) or set(record) != {
                "name",
                "dtype",
                "shape",
                "size_bytes",
                "sha256",
            }:
                raise RolloutContractError("candidate snapshot payload array schema drift")
            name = record["name"]
            expected_dtype = "|b1" if name == "candidate_mask" else "<f4"
            shape = record["shape"]
            size = record["size_bytes"]
            if (
                record["dtype"] != expected_dtype
                or not isinstance(shape, list)
                or not shape
                or any(type(value) is not int or value <= 0 for value in shape)
                or type(size) is not int
                or size < 0
            ):
                raise RolloutContractError("candidate snapshot payload array metadata drift")
            end = offset + size
            raw = payload[offset:end]
            if (
                len(raw) != size
                or record["sha256"] != hashlib.sha256(raw).hexdigest()
            ):
                raise RolloutContractError("candidate snapshot payload array hash mismatch")
            dtype = np.dtype(expected_dtype)
            expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if expected_size != size:
                raise RolloutContractError("candidate snapshot payload array size mismatch")
            arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy()
            offset = end
        if offset != len(payload):
            raise RolloutContractError("candidate snapshot payload has trailing bytes")
        cells = header.get("frontier_cells")
        if not isinstance(cells, list) or any(
            not isinstance(cell, list)
            or len(cell) != 2
            or any(type(value) is not int for value in cell)
            for cell in cells
        ):
            raise RolloutContractError("candidate snapshot payload cells are invalid")
        summary = header.get("top_m_selection_summary")
        if not isinstance(summary, dict):
            raise RolloutContractError("candidate snapshot top-M summary is invalid")
        return cls(
            **arrays,
            frontier_cells=tuple(CellXY(*cell) for cell in cells),
            top_m_selection_summary=MappingProxyType(_copy_json_mapping(summary)),
        )


@dataclass(frozen=True, slots=True)
class RolloutTransition:
    snapshot_bytes: bytes
    candidate_snapshot_sha256: str
    policy_state_sha256: str
    config_sha256: str
    lineage: Mapping[str, object]
    selected_frontier_index: int
    selected_cell_xy: CellXY
    selected_theta: np.float32
    old_log_prob_frontier: np.float32
    old_log_prob_theta: np.float32
    old_log_prob_total: np.float32
    old_value: np.float32
    reward: np.float32
    done: bool
    done_reason: str
    planned_path_cells: tuple[CellXY, ...]
    path_length_m: np.float32
    path_observation_step_m: np.float32
    coverage_gain_cells: int
    coverage_gain_per_meter: np.float32
    observation_sample_count: int
    ray_count_per_sample: int
    ray_cell_visit_count: int
    newly_observed_cell_count: int
    planner_validation_summary: Mapping[str, object]
    execution_observation_summary: Mapping[str, object]
    frontier_extractor_version: str
    observation_schema_version: str
    action_space_version: str
    reward_version: str
    planner_version: str

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def create(
        cls,
        *,
        snapshot: CandidateSnapshot,
        policy_state_sha256: str,
        config_sha256: str,
        lineage: Mapping[str, object],
        selected_frontier_index: int,
        selected_cell_xy: CellXY,
        selected_theta: float,
        old_log_prob_frontier: float,
        old_log_prob_theta: float,
        old_log_prob_total: float,
        old_value: float,
        reward: float,
        done: bool,
        done_reason: str,
        planned_path_cells: tuple[CellXY, ...],
        path_length_m: float,
        path_observation_step_m: float,
        coverage_gain_cells: int,
        coverage_gain_per_meter: float,
        observation_sample_count: int,
        ray_count_per_sample: int,
        ray_cell_visit_count: int,
        newly_observed_cell_count: int,
        planner_validation_summary: Mapping[str, object],
        execution_observation_summary: Mapping[str, object],
        frontier_extractor_version: str,
        observation_schema_version: str,
        action_space_version: str,
        reward_version: str,
        planner_version: str,
    ) -> "RolloutTransition":
        if not isinstance(snapshot, CandidateSnapshot):
            raise RolloutContractError("transition requires a candidate snapshot")
        snapshot_bytes = snapshot.to_bytes()
        return cls(
            snapshot_bytes=snapshot_bytes,
            candidate_snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
            policy_state_sha256=policy_state_sha256,
            config_sha256=config_sha256,
            lineage=MappingProxyType(_copy_json_mapping(lineage)),
            selected_frontier_index=selected_frontier_index,
            selected_cell_xy=selected_cell_xy,
            selected_theta=np.float32(selected_theta),
            old_log_prob_frontier=np.float32(old_log_prob_frontier),
            old_log_prob_theta=np.float32(old_log_prob_theta),
            old_log_prob_total=np.float32(old_log_prob_total),
            old_value=np.float32(old_value),
            reward=np.float32(reward),
            done=done,
            done_reason=done_reason,
            planned_path_cells=tuple(planned_path_cells),
            path_length_m=np.float32(path_length_m),
            path_observation_step_m=np.float32(path_observation_step_m),
            coverage_gain_cells=coverage_gain_cells,
            coverage_gain_per_meter=np.float32(coverage_gain_per_meter),
            observation_sample_count=observation_sample_count,
            ray_count_per_sample=ray_count_per_sample,
            ray_cell_visit_count=ray_cell_visit_count,
            newly_observed_cell_count=newly_observed_cell_count,
            planner_validation_summary=MappingProxyType(
                _copy_json_mapping(planner_validation_summary)
            ),
            execution_observation_summary=MappingProxyType(
                _copy_json_mapping(execution_observation_summary)
            ),
            frontier_extractor_version=frontier_extractor_version,
            observation_schema_version=observation_schema_version,
            action_space_version=action_space_version,
            reward_version=reward_version,
            planner_version=planner_version,
        )

    @property
    def snapshot(self) -> CandidateSnapshot:
        return CandidateSnapshot.from_bytes(
            self.snapshot_bytes,
            expected_sha256=self.candidate_snapshot_sha256,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_snapshot_sha256": self.candidate_snapshot_sha256,
            "policy_state_sha256": self.policy_state_sha256,
            "config_sha256": self.config_sha256,
            "lineage": dict(self.lineage),
            "selected_frontier_index": self.selected_frontier_index,
            "selected_cell_xy": [self.selected_cell_xy.x, self.selected_cell_xy.y],
            "selected_theta": float(self.selected_theta),
            "old_log_prob_frontier": float(self.old_log_prob_frontier),
            "old_log_prob_theta": float(self.old_log_prob_theta),
            "old_log_prob_total": float(self.old_log_prob_total),
            "old_value": float(self.old_value),
            "reward": float(self.reward),
            "done": self.done,
            "done_reason": self.done_reason,
            "planned_path_cells": [[cell.x, cell.y] for cell in self.planned_path_cells],
            "path_length_m": float(self.path_length_m),
            "path_observation_step_m": float(self.path_observation_step_m),
            "coverage_gain_cells": self.coverage_gain_cells,
            "coverage_gain_per_meter": float(self.coverage_gain_per_meter),
            "observation_sample_count": self.observation_sample_count,
            "ray_count_per_sample": self.ray_count_per_sample,
            "ray_cell_visit_count": self.ray_cell_visit_count,
            "newly_observed_cell_count": self.newly_observed_cell_count,
            "planner_validation_summary": dict(self.planner_validation_summary),
            "execution_observation_summary": dict(self.execution_observation_summary),
            "frontier_extractor_version": self.frontier_extractor_version,
            "observation_schema_version": self.observation_schema_version,
            "action_space_version": self.action_space_version,
            "reward_version": self.reward_version,
            "planner_version": self.planner_version,
        }

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        snapshot_bytes: bytes,
    ) -> "RolloutTransition":
        if not isinstance(record, Mapping) or set(record) != _TRANSITION_RECORD_FIELDS:
            raise RolloutContractError("transition record field set drift")
        selected_cell = _cell_from_json(record["selected_cell_xy"], "selected cell")
        path_value = record["planned_path_cells"]
        if not isinstance(path_value, list):
            raise RolloutContractError("planned path cells must be a list")
        path_cells = tuple(
            _cell_from_json(value, "planned path cell") for value in path_value
        )
        lineage = record["lineage"]
        planner = record["planner_validation_summary"]
        execution = record["execution_observation_summary"]
        if not all(isinstance(value, Mapping) for value in (lineage, planner, execution)):
            raise RolloutContractError("transition diagnostic mappings are invalid")
        return cls(
            snapshot_bytes=bytes(snapshot_bytes),
            candidate_snapshot_sha256=record["candidate_snapshot_sha256"],
            policy_state_sha256=record["policy_state_sha256"],
            config_sha256=record["config_sha256"],
            lineage=MappingProxyType(_copy_json_mapping(lineage)),
            selected_frontier_index=record["selected_frontier_index"],
            selected_cell_xy=selected_cell,
            selected_theta=np.float32(record["selected_theta"]),
            old_log_prob_frontier=np.float32(record["old_log_prob_frontier"]),
            old_log_prob_theta=np.float32(record["old_log_prob_theta"]),
            old_log_prob_total=np.float32(record["old_log_prob_total"]),
            old_value=np.float32(record["old_value"]),
            reward=np.float32(record["reward"]),
            done=record["done"],
            done_reason=record["done_reason"],
            planned_path_cells=path_cells,
            path_length_m=np.float32(record["path_length_m"]),
            path_observation_step_m=np.float32(record["path_observation_step_m"]),
            coverage_gain_cells=record["coverage_gain_cells"],
            coverage_gain_per_meter=np.float32(record["coverage_gain_per_meter"]),
            observation_sample_count=record["observation_sample_count"],
            ray_count_per_sample=record["ray_count_per_sample"],
            ray_cell_visit_count=record["ray_cell_visit_count"],
            newly_observed_cell_count=record["newly_observed_cell_count"],
            planner_validation_summary=MappingProxyType(_copy_json_mapping(planner)),
            execution_observation_summary=MappingProxyType(
                _copy_json_mapping(execution)
            ),
            frontier_extractor_version=record["frontier_extractor_version"],
            observation_schema_version=record["observation_schema_version"],
            action_space_version=record["action_space_version"],
            reward_version=record["reward_version"],
            planner_version=record["planner_version"],
        )

    def _validate(self) -> None:
        snapshot = CandidateSnapshot.from_bytes(
            self.snapshot_bytes,
            expected_sha256=self.candidate_snapshot_sha256,
        )
        for name, value in (
            ("candidate snapshot", self.candidate_snapshot_sha256),
            ("policy", self.policy_state_sha256),
            ("config", self.config_sha256),
        ):
            if not _is_sha256(value):
                raise RolloutContractError(f"{name} hash is invalid")
        if type(self.selected_frontier_index) is not int:
            raise RolloutContractError("selected frontier index must be an integer")
        if not 0 <= self.selected_frontier_index < snapshot.candidate_mask.size:
            raise RolloutContractError("selected frontier index is out of range")
        if not bool(snapshot.candidate_mask[self.selected_frontier_index]):
            raise RolloutContractError("selected frontier index is masked")
        valid_indices = tuple(int(value) for value in np.flatnonzero(snapshot.candidate_mask))
        selected_rank = valid_indices.index(self.selected_frontier_index)
        if snapshot.frontier_cells[selected_rank] != self.selected_cell_xy:
            raise RolloutContractError("selected cell does not match the saved snapshot")
        float_values = (
            self.selected_theta,
            self.old_log_prob_frontier,
            self.old_log_prob_theta,
            self.old_log_prob_total,
            self.old_value,
            self.reward,
            self.path_length_m,
            self.path_observation_step_m,
            self.coverage_gain_per_meter,
        )
        if any(not isinstance(value, np.float32) for value in float_values):
            raise RolloutContractError("transition scalar math fields must be FP32")
        if any(not math.isfinite(float(value)) for value in float_values):
            raise RolloutContractError("transition scalar fields must be finite")
        if not -math.pi <= float(self.selected_theta) < math.pi:
            raise RolloutContractError("selected theta must use the half-open interval")
        expected_total = np.float32(
            self.old_log_prob_frontier + self.old_log_prob_theta
        )
        if self.old_log_prob_total.tobytes() != expected_total.tobytes():
            raise RolloutContractError("old log-prob factorization mismatch")
        if type(self.done) is not bool or self.done_reason not in _DONE_REASONS:
            raise RolloutContractError("transition terminal fields are invalid")
        if self.done != (self.done_reason != "none"):
            raise RolloutContractError("transition done reason does not match done")
        if self.path_length_m < 0.0 or self.path_observation_step_m <= 0.0:
            raise RolloutContractError("transition path diagnostics are invalid")
        if any(not isinstance(cell, CellXY) for cell in self.planned_path_cells):
            raise RolloutContractError("planned path cells are invalid")
        if self.path_length_m > 0.0 and not self.planned_path_cells:
            raise RolloutContractError("planned path cells are missing")
        integer_fields = (
            self.coverage_gain_cells,
            self.observation_sample_count,
            self.ray_count_per_sample,
            self.ray_cell_visit_count,
            self.newly_observed_cell_count,
        )
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise RolloutContractError("transition count diagnostics are invalid")
        expected_gain = (
            np.float32(self.coverage_gain_cells / self.path_length_m)
            if self.path_length_m > 0.0
            else np.float32(0.0)
        )
        if not np.isclose(
            self.coverage_gain_per_meter,
            expected_gain,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise RolloutContractError("coverage gain per meter mismatch")
        for name in (
            "frontier_extractor_version",
            "observation_schema_version",
            "action_space_version",
            "reward_version",
            "planner_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise RolloutContractError(f"{name} is invalid")


@dataclass(frozen=True, slots=True)
class GAEResult:
    raw_advantages: np.ndarray
    normalized_advantages: np.ndarray
    returns: np.ndarray

    def __post_init__(self) -> None:
        for value in (
            self.raw_advantages,
            self.normalized_advantages,
            self.returns,
        ):
            value.setflags(write=False)


def compute_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
) -> GAEResult:
    """计算固定 gamma/lambda 的 `[T,E]` FP32 GAE。"""

    arrays = (rewards, values, last_values)
    if any(not isinstance(value, np.ndarray) or value.dtype != np.float32 for value in arrays):
        raise RolloutContractError("GAE rewards, values, and last_values must be FP32")
    if not isinstance(dones, np.ndarray) or dones.dtype != np.bool_:
        raise RolloutContractError("GAE dones must use boolean dtype")
    if (
        rewards.ndim != 2
        or rewards.shape[0] == 0
        or rewards.shape[1] == 0
        or values.shape != rewards.shape
        or dones.shape != rewards.shape
        or last_values.shape != (rewards.shape[1],)
    ):
        raise RolloutContractError("GAE array shape mismatch")
    if any(not np.isfinite(value).all() for value in arrays):
        raise RolloutContractError("GAE inputs must be finite")

    time_steps, env_count = rewards.shape
    raw = np.zeros((time_steps, env_count), dtype=np.float32)
    next_advantage = np.zeros((env_count,), dtype=np.float32)
    for time_index in range(time_steps - 1, -1, -1):
        next_value = last_values if time_index == time_steps - 1 else values[time_index + 1]
        not_done = np.logical_not(dones[time_index]).astype(np.float32)
        delta = np.float32(
            rewards[time_index]
            + GAMMA * next_value * not_done
            - values[time_index]
        )
        next_advantage = np.float32(
            delta + GAMMA * GAE_LAMBDA * not_done * next_advantage
        )
        raw[time_index] = next_advantage
    returns = np.float32(raw + values)
    mean = np.mean(raw, dtype=np.float32)
    centered = np.float32(raw - mean)
    variance = np.mean(np.float32(centered * centered), dtype=np.float32)
    standard_deviation = np.sqrt(variance, dtype=np.float32)
    if not math.isfinite(float(standard_deviation)) or standard_deviation <= np.float32(1e-8):
        raise RolloutContractError("GAE normalization is degenerate")
    normalized = np.float32(centered / standard_deviation)
    if any(not np.isfinite(value).all() for value in (raw, normalized, returns)):
        raise RolloutContractError("GAE outputs must be finite")
    return GAEResult(
        raw_advantages=raw,
        normalized_advantages=normalized,
        returns=returns,
    )


class RolloutBatch:
    """固定 `[T=128,E=8]`、只能被一个 PPO update 消费的 batch。"""

    def __init__(
        self,
        *,
        transitions: tuple[RolloutTransition, ...],
        rewards: np.ndarray,
        dones: np.ndarray,
        gae: GAEResult,
        policy_state_sha256: str,
        config_sha256: str,
        lineage: Mapping[str, object],
    ) -> None:
        if len(transitions) != ROLLOUT_BATCH_SIZE:
            raise RolloutContractError("rollout batch transition count drift")
        expected_shape = (ROLLOUT_STEPS_PER_ENV, ROLLOUT_ENV_COUNT)
        if rewards.shape != expected_shape or dones.shape != expected_shape:
            raise RolloutContractError("rollout batch layout shape drift")
        if any(
            value.shape != expected_shape
            for value in (
                gae.raw_advantages,
                gae.normalized_advantages,
                gae.returns,
            )
        ):
            raise RolloutContractError("rollout batch GAE shape drift")
        self._transitions = transitions
        self._size = len(transitions)
        self.rewards = np.asarray(rewards, dtype=np.float32).copy()
        self.dones = np.asarray(dones, dtype=np.bool_).copy()
        self.raw_advantages = gae.raw_advantages.copy()
        self.normalized_advantages = gae.normalized_advantages.copy()
        self.returns = gae.returns.copy()
        self.old_log_prob_frontier = _transition_matrix(
            transitions, "old_log_prob_frontier", np.float32
        )
        self.old_log_prob_theta = _transition_matrix(
            transitions, "old_log_prob_theta", np.float32
        )
        self.old_log_prob_total = _transition_matrix(
            transitions, "old_log_prob_total", np.float32
        )
        self.old_values = _transition_matrix(transitions, "old_value", np.float32)
        self.selected_frontier_indices = _transition_matrix(
            transitions, "selected_frontier_index", np.int64
        )
        self.selected_thetas = _transition_matrix(
            transitions, "selected_theta", np.float32
        )
        self.policy_state_sha256 = policy_state_sha256
        self.config_sha256 = config_sha256
        self.lineage = MappingProxyType(_copy_json_mapping(lineage))
        self.snapshot_hashes = tuple(
            transition.candidate_snapshot_sha256 for transition in transitions
        )
        self._status = "fresh"
        for array in (
            self.rewards,
            self.dones,
            self.raw_advantages,
            self.normalized_advantages,
            self.returns,
            self.old_log_prob_frontier,
            self.old_log_prob_theta,
            self.old_log_prob_total,
            self.old_values,
            self.selected_frontier_indices,
            self.selected_thetas,
        ):
            array.setflags(write=False)

    @property
    def layout_shape(self) -> tuple[int, int]:
        return ROLLOUT_STEPS_PER_ENV, ROLLOUT_ENV_COUNT

    @property
    def size(self) -> int:
        return self._size

    @property
    def transitions(self) -> tuple[RolloutTransition, ...]:
        if self._status == "consumed":
            raise RolloutContractError("rollout batch transitions were consumed")
        return self._transitions

    @property
    def consumed(self) -> bool:
        return self._status == "consumed"

    @property
    def claimed(self) -> bool:
        return self._status == "claimed"

    def claim_for_update(self, current_policy_state_sha256: str) -> None:
        if self._status == "consumed":
            raise RolloutContractError("rollout batch is already consumed")
        if self._status == "claimed":
            raise RolloutContractError("rollout batch is already claimed")
        if self._status == "invalid":
            raise RolloutContractError("rollout batch is invalid")
        if current_policy_state_sha256 != self.policy_state_sha256:
            raise RolloutContractError("rollout batch has a stale policy hash")
        self._status = "claimed"

    def mark_consumed(self) -> None:
        if self._status != "claimed":
            raise RolloutContractError("rollout batch was not claimed for update")
        self._status = "consumed"
        self._transitions = ()

    def invalidate(self) -> None:
        if self._status == "consumed":
            raise RolloutContractError("consumed rollout batch cannot be invalidated")
        self._status = "invalid"
        self._transitions = ()

    def policy_observations(self, flat_indices: np.ndarray) -> tuple[PolicyObservation, ...]:
        if self._status != "claimed":
            raise RolloutContractError("rollout batch must be claimed before snapshot access")
        if (
            not isinstance(flat_indices, np.ndarray)
            or flat_indices.dtype != np.int64
            or flat_indices.ndim != 1
            or bool((flat_indices < 0).any())
            or bool((flat_indices >= self._size).any())
        ):
            raise RolloutContractError("rollout batch indices are invalid")
        return tuple(
            self._transitions[int(index)].snapshot.to_policy_observation()
            for index in flat_indices
        )


class RolloutBuffer:
    """每个环境只计 trainable transition 的固定 Stage 4 buffer。"""

    def __init__(self) -> None:
        self._per_env: list[list[RolloutTransition]] = [
            [] for _ in range(ROLLOUT_ENV_COUNT)
        ]
        self._policy_state_sha256: str | None = None
        self._config_sha256: str | None = None
        self._lineage: dict[str, object] | None = None
        self._finalized = False

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def per_env_counts(self) -> tuple[int, ...]:
        return tuple(len(values) for values in self._per_env)

    @property
    def transition_count(self) -> int:
        return sum(self.per_env_counts)

    def add(self, env_index: int, transition: RolloutTransition) -> None:
        if self._finalized:
            raise RolloutContractError("rollout buffer was finalized")
        if type(env_index) is not int or not 0 <= env_index < ROLLOUT_ENV_COUNT:
            raise RolloutContractError("rollout buffer env index is invalid")
        if not isinstance(transition, RolloutTransition):
            raise RolloutContractError("rollout buffer accepts RolloutTransition only")
        if len(self._per_env[env_index]) >= ROLLOUT_STEPS_PER_ENV:
            raise RolloutContractError("rollout buffer env quota exceeded")
        transition._validate()
        if self._policy_state_sha256 is None:
            self._policy_state_sha256 = transition.policy_state_sha256
            self._config_sha256 = transition.config_sha256
            self._lineage = dict(transition.lineage)
        if transition.policy_state_sha256 != self._policy_state_sha256:
            raise RolloutContractError("rollout buffer contains mixed policy hashes")
        if transition.config_sha256 != self._config_sha256:
            raise RolloutContractError("rollout buffer contains mixed config hashes")
        if dict(transition.lineage) != self._lineage:
            raise RolloutContractError("rollout buffer contains mixed lineage")
        self._per_env[env_index].append(transition)

    def finalize(self, last_values: np.ndarray) -> RolloutBatch:
        if self._finalized:
            raise RolloutContractError("rollout buffer was already finalized")
        if self.per_env_counts != (ROLLOUT_STEPS_PER_ENV,) * ROLLOUT_ENV_COUNT:
            raise RolloutContractError("rollout buffer env quota is incomplete")
        if (
            not isinstance(last_values, np.ndarray)
            or last_values.dtype != np.float32
            or last_values.shape != (ROLLOUT_ENV_COUNT,)
            or not np.isfinite(last_values).all()
        ):
            raise RolloutContractError("rollout buffer last_values must be finite FP32 [8]")
        transitions = tuple(
            self._per_env[env_index][time_index]
            for time_index in range(ROLLOUT_STEPS_PER_ENV)
            for env_index in range(ROLLOUT_ENV_COUNT)
        )
        rewards = _transition_matrix(transitions, "reward", np.float32)
        values = _transition_matrix(transitions, "old_value", np.float32)
        dones = _transition_matrix(transitions, "done", np.bool_)
        gae = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_values=last_values,
        )
        assert self._policy_state_sha256 is not None
        assert self._config_sha256 is not None
        assert self._lineage is not None
        batch = RolloutBatch(
            transitions=transitions,
            rewards=rewards,
            dones=dones,
            gae=gae,
            policy_state_sha256=self._policy_state_sha256,
            config_sha256=self._config_sha256,
            lineage=self._lineage,
        )
        self._per_env = [[] for _ in range(ROLLOUT_ENV_COUNT)]
        self._policy_state_sha256 = None
        self._config_sha256 = None
        self._lineage = None
        self._finalized = True
        return batch


def _transition_matrix(
    transitions: tuple[RolloutTransition, ...],
    field: str,
    dtype: np.dtype | type,
) -> np.ndarray:
    values = np.asarray([getattr(value, field) for value in transitions], dtype=dtype)
    return values.reshape(ROLLOUT_STEPS_PER_ENV, ROLLOUT_ENV_COUNT)


def _cell_from_json(value: object, label: str) -> CellXY:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(component) is not int for component in value)
    ):
        raise RolloutContractError(f"{label} is invalid")
    return CellXY(value[0], value[1])


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _copy_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise RolloutContractError("snapshot metadata must be finite JSON") from exc
    if not isinstance(copied, dict):
        raise RolloutContractError("snapshot metadata must be a JSON object")
    _reject_forbidden_snapshot_content(copied)
    return copied


def _reject_forbidden_snapshot_content(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _FORBIDDEN_SNAPSHOT_KEYS:
                raise RolloutContractError(f"forbidden snapshot field: {key}")
            _reject_forbidden_snapshot_content(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_forbidden_snapshot_content(child)

"""Incremental, resumable HOPPER task-coverability closure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from ..capability_freeze import FrozenPlatformCapability
from ..environment.task_area import DETAIL_PER_GLOBAL
from .task_cache import PlatformTaskPayload, TaskCommonArtifact


_JOURNAL_SCHEMA = "lunar-hopper-task-closure-journal/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class HopperEdgeState(str, Enum):
    CERTIFIED = "CERTIFIED"
    STABLE_PHYSICAL_REJECTION = "STABLE_PHYSICAL_REJECTION"
    WAITING_EVIDENCE = "WAITING_EVIDENCE"


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("hopper closure journal is not canonical") from error


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"hopper closure journal {name} is invalid")
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _edge_states(value: object) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(value, Mapping):
        raise ValueError("hopper closure journal edge states are invalid")
    output: dict[str, Mapping[str, object]] = {}
    for key, raw in sorted(value.items()):
        if not isinstance(key, str) or not key or not isinstance(raw, Mapping):
            raise ValueError("hopper closure journal edge state is invalid")
        normalized = json.loads(_canonical_json(dict(raw)).decode("utf-8"))
        if not isinstance(normalized, dict):
            raise ValueError("hopper closure journal edge state differs")
        output[key] = MappingProxyType(
            {name: _freeze_json(item) for name, item in normalized.items()}
        )
    return MappingProxyType(output)


@dataclass(frozen=True, slots=True, eq=False)
class HopperClosureJournal:
    task_key_sha256: str
    common_artifact_sha256: str
    completed_round: int
    reached_positions_um: np.ndarray
    observed_detail_bits: np.ndarray
    pending_sources: tuple[int, ...]
    edge_states: Mapping[str, Mapping[str, object]]
    journal_sha256: str

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, HopperClosureJournal)
            and self.task_key_sha256 == other.task_key_sha256
            and self.common_artifact_sha256 == other.common_artifact_sha256
            and self.completed_round == other.completed_round
            and np.array_equal(
                self.reached_positions_um, other.reached_positions_um
            )
            and np.array_equal(
                self.observed_detail_bits, other.observed_detail_bits
            )
            and self.pending_sources == other.pending_sources
            and {
                key: dict(value) for key, value in self.edge_states.items()
            }
            == {
                key: dict(value) for key, value in other.edge_states.items()
            }
            and self.journal_sha256 == other.journal_sha256
        )

    def __post_init__(self) -> None:
        _require_sha256(self.task_key_sha256, "task key")
        _require_sha256(self.common_artifact_sha256, "common artifact")
        if type(self.completed_round) is not int or self.completed_round < 0:
            raise ValueError("hopper closure journal round is invalid")
        reached = np.ascontiguousarray(
            np.asarray(self.reached_positions_um, dtype=np.int64)
        )
        if reached.ndim != 2 or reached.shape[1:] != (3,):
            raise ValueError("hopper closure journal reached positions differ")
        reached.setflags(write=False)
        observed = np.ascontiguousarray(
            np.asarray(self.observed_detail_bits, dtype=np.uint8)
        )
        if observed.ndim != 1 or observed.size == 0:
            raise ValueError("hopper closure journal observed bits differ")
        observed.setflags(write=False)
        if (
            not isinstance(self.pending_sources, tuple)
            or any(
                type(value) is not int or value < 0
                for value in self.pending_sources
            )
            or tuple(sorted(set(self.pending_sources))) != self.pending_sources
        ):
            raise ValueError("hopper closure journal pending sources differ")
        states = _edge_states(self.edge_states)
        object.__setattr__(self, "reached_positions_um", reached)
        object.__setattr__(self, "observed_detail_bits", observed)
        object.__setattr__(self, "edge_states", states)
        _require_sha256(self.journal_sha256, "identity")
        if self.journal_sha256 != hashlib.sha256(
            _canonical_json(self._body())
        ).hexdigest():
            raise ValueError("hopper closure journal identity differs")

    @classmethod
    def create(
        cls,
        *,
        task_key_sha256: str,
        common_artifact_sha256: str,
        completed_round: int,
        reached_positions_um: np.ndarray,
        observed_detail_bits: np.ndarray,
        pending_sources: tuple[int, ...],
        edge_states: Mapping[str, Mapping[str, object]],
    ) -> "HopperClosureJournal":
        reached = np.ascontiguousarray(reached_positions_um, dtype=np.int64)
        observed = np.ascontiguousarray(observed_detail_bits, dtype=np.uint8)
        body = {
            "schema": _JOURNAL_SCHEMA,
            "task_key_sha256": task_key_sha256,
            "common_artifact_sha256": common_artifact_sha256,
            "completed_round": completed_round,
            "reached_positions_um": reached.tolist(),
            "observed_detail_bits_hex": observed.tobytes().hex(),
            "pending_sources": list(pending_sources),
            "edge_states": dict(edge_states),
        }
        return cls(
            task_key_sha256=task_key_sha256,
            common_artifact_sha256=common_artifact_sha256,
            completed_round=completed_round,
            reached_positions_um=reached,
            observed_detail_bits=observed,
            pending_sources=pending_sources,
            edge_states=edge_states,
            journal_sha256=hashlib.sha256(_canonical_json(body)).hexdigest(),
        )

    def _body(self) -> dict[str, object]:
        return {
            "schema": _JOURNAL_SCHEMA,
            "task_key_sha256": self.task_key_sha256,
            "common_artifact_sha256": self.common_artifact_sha256,
            "completed_round": self.completed_round,
            "reached_positions_um": self.reached_positions_um.tolist(),
            "observed_detail_bits_hex": self.observed_detail_bits.tobytes().hex(),
            "pending_sources": list(self.pending_sources),
            "edge_states": {
                key: _thaw_json(value)
                for key, value in self.edge_states.items()
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "journal_sha256": self.journal_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "HopperClosureJournal":
        fields = {
            "schema",
            "task_key_sha256",
            "common_artifact_sha256",
            "completed_round",
            "reached_positions_um",
            "observed_detail_bits_hex",
            "pending_sources",
            "edge_states",
            "journal_sha256",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value["schema"] != _JOURNAL_SCHEMA
            or not isinstance(value["observed_detail_bits_hex"], str)
        ):
            raise ValueError("hopper closure journal schema differs")
        try:
            observed = np.frombuffer(
                bytes.fromhex(value["observed_detail_bits_hex"]),
                dtype=np.uint8,
            ).copy()
        except ValueError as error:
            raise ValueError("hopper closure journal observed bits differ") from error
        try:
            return cls(
                task_key_sha256=value["task_key_sha256"],
                common_artifact_sha256=value["common_artifact_sha256"],
                completed_round=value["completed_round"],
                reached_positions_um=np.asarray(
                    value["reached_positions_um"], dtype=np.int64
                ),
                observed_detail_bits=observed,
                pending_sources=tuple(value["pending_sources"]),
                edge_states=value["edge_states"],
                journal_sha256=value["journal_sha256"],
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("hopper closure journal payload differs") from error


def _unpack_mask(
    packed: np.ndarray | bytes,
    *,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    raw = np.frombuffer(bytes(packed), dtype=np.uint8)
    bit_count = int(np.prod(shape, dtype=np.int64))
    expected_bytes = (bit_count + 7) // 8
    if raw.size != expected_bytes:
        raise ValueError(f"hopper closure resume {name} geometry differs")
    output = np.unpackbits(
        raw,
        bitorder="little",
        count=bit_count,
    ).astype(np.bool_, copy=False).reshape(shape)
    if not np.array_equal(
        np.packbits(output.reshape(-1), bitorder="little"), raw
    ):
        raise ValueError(f"hopper closure resume {name} padding differs")
    return np.ascontiguousarray(output)


def _state_index(value: object, *, span: int, name: str) -> tuple[int, int]:
    if type(value) is not int or not 0 <= value < span * span:
        raise ValueError(f"hopper closure resume {name} index differs")
    return divmod(value, span)


def _decode_resume_state(
    *,
    journal: HopperClosureJournal,
    common: TaskCommonArtifact,
    platform: FrozenPlatformCapability,
    qualified_start: Mapping[str, object],
) -> Mapping[str, object]:
    from .task_coverability import _task_projected_view

    span = common.geometry.span_cells
    state = journal.edge_states.get("__state__")
    expected_state_fields = {
        "source_identity_sha256",
        "platform_capability_sha256",
        "start_identity_sha256",
        "safe_union_bits_hex",
        "processed_certified_edges",
        "landing_algorithm_id",
        "edge_algorithm_id",
        "first_round_candidate_count",
        "trajectory_edge_patch_count",
        "logical_projection_call_count",
        "outside_roi_support_count",
    }
    if not isinstance(state, Mapping) or set(state) != expected_state_fields:
        raise ValueError("hopper closure resume authority state differs")
    if (
        state["source_identity_sha256"] != common.source_identity_sha256
        or state["platform_capability_sha256"] != platform.content_sha256
        or state["start_identity_sha256"]
        != qualified_start.get("start_identity_sha256")
    ):
        raise ValueError("hopper closure resume authority differs")
    for name in ("landing_algorithm_id", "edge_algorithm_id"):
        if not isinstance(state[name], str) or not state[name]:
            raise ValueError("hopper closure resume algorithm differs")
    for name in (
        "first_round_candidate_count",
        "trajectory_edge_patch_count",
        "logical_projection_call_count",
        "outside_roi_support_count",
    ):
        if type(state[name]) is not int or state[name] < 0:
            raise ValueError("hopper closure resume counter differs")

    detail_shape = (
        span * DETAIL_PER_GLOBAL,
        span * DETAIL_PER_GLOBAL,
    )
    observed_detail = _unpack_mask(
        journal.observed_detail_bits,
        shape=detail_shape,
        name="observed detail",
    )
    safe_hex = state["safe_union_bits_hex"]
    if not isinstance(safe_hex, str):
        raise ValueError("hopper closure resume safe union differs")
    try:
        safe_bytes = bytes.fromhex(safe_hex)
    except ValueError as error:
        raise ValueError("hopper closure resume safe union differs") from error
    safe_union = _unpack_mask(
        safe_bytes,
        shape=(span, span),
        name="safe union",
    )

    coarse = _task_projected_view(common, detail=False)
    reached_positions: dict[
        tuple[int, int], tuple[float, float, float]
    ] = {}
    reached_cells: list[tuple[int, int]] = []
    for raw_position in journal.reached_positions_um:
        position = tuple(
            float(value) / 1_000_000.0 for value in raw_position
        )
        try:
            cell = coarse.canvas.world_to_grid(*position[:2])
        except ValueError as error:
            raise ValueError(
                "hopper closure resume reached position leaves task"
            ) from error
        if cell in reached_positions:
            raise ValueError("hopper closure resume reached cells differ")
        reached_positions[cell] = position
        reached_cells.append(cell)
    if reached_cells != sorted(reached_cells):
        raise ValueError("hopper closure resume reached order differs")

    pending_sources = {
        _state_index(value, span=span, name="pending source")
        for value in journal.pending_sources
    }
    if len(pending_sources) != len(journal.pending_sources):
        raise ValueError("hopper closure resume pending sources differ")

    edge_states: dict[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[str, str, tuple[int, ...]],
    ] = {}
    for key, raw in journal.edge_states.items():
        if key == "__state__":
            continue
        parts = key.split(":")
        if len(parts) != 2 or any(not part.isdecimal() for part in parts):
            raise ValueError("hopper closure resume edge key differs")
        source = _state_index(int(parts[0]), span=span, name="edge source")
        target = _state_index(int(parts[1]), span=span, name="edge target")
        if set(raw) != {
            "disposition",
            "reason_code",
            "dependency_tile_indices",
        }:
            raise ValueError("hopper closure resume edge state differs")
        disposition = raw["disposition"]
        reason = raw["reason_code"]
        dependencies = raw["dependency_tile_indices"]
        if (
            disposition not in {item.value for item in HopperEdgeState}
            or not isinstance(reason, str)
            or not isinstance(dependencies, tuple)
            or any(type(value) is not int for value in dependencies)
            or any(not 0 <= value < span * span for value in dependencies)
            or dependencies != tuple(sorted(set(dependencies)))
        ):
            raise ValueError("hopper closure resume edge state differs")
        edge_states[(source, target)] = (
            disposition,
            reason,
            tuple(dependencies),
        )

    raw_processed = state["processed_certified_edges"]
    if not isinstance(raw_processed, tuple):
        raise ValueError("hopper closure resume processed edges differ")
    processed: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for raw_edge in raw_processed:
        if (
            not isinstance(raw_edge, tuple)
            or len(raw_edge) != 2
        ):
            raise ValueError("hopper closure resume processed edge differs")
        edge = (
            _state_index(raw_edge[0], span=span, name="processed source"),
            _state_index(raw_edge[1], span=span, name="processed target"),
        )
        processed.add(edge)
    if (
        len(processed) != len(raw_processed)
        or not pending_sources.issubset(reached_positions)
        or any(
            source not in reached_positions or target not in reached_positions
            for source, target in processed
        )
        or any(
            edge_states.get(edge, (None,))[0]
            != HopperEdgeState.CERTIFIED.value
            for edge in processed
        )
    ):
        raise ValueError("hopper closure resume graph differs")

    return MappingProxyType(
        {
            "completed_round": journal.completed_round,
            "reached_positions": reached_positions,
            "observed_detail": observed_detail,
            "pending_sources": pending_sources,
            "edge_states": edge_states,
            "safe_union": safe_union,
            "processed_certified_edges": processed,
            "landing_algorithm_id": state["landing_algorithm_id"],
            "edge_algorithm_id": state["edge_algorithm_id"],
            "first_round_candidate_count": state[
                "first_round_candidate_count"
            ],
            "trajectory_edge_patch_count": state[
                "trajectory_edge_patch_count"
            ],
            "logical_projection_call_count": state[
                "logical_projection_call_count"
            ],
            "outside_roi_support_count": state[
                "outside_roi_support_count"
            ],
        }
    )


class HopperTaskClosureBuilder:
    """Build the optimized closure; the reference remains the authority."""

    def __init__(
        self,
        *,
        common: TaskCommonArtifact,
        platform: FrozenPlatformCapability,
        qualified_start: Mapping[str, object],
        bridge: object,
        task_key_sha256: str | None = None,
        journal_sink: Callable[[HopperClosureJournal], None] | None = None,
    ) -> None:
        if not isinstance(common, TaskCommonArtifact):
            raise TypeError("hopper closure builder requires TaskCommonArtifact")
        if not isinstance(platform, FrozenPlatformCapability):
            raise TypeError("hopper closure builder requires frozen capability")
        self.common = common
        self.platform = platform
        self.qualified_start = qualified_start
        self.bridge = bridge
        self.task_key_sha256 = (
            task_key_sha256
            if task_key_sha256 is not None
            else hashlib.sha256(
                (
                    common.artifact_sha256
                    + platform.content_sha256
                    + str(qualified_start.get("start_identity_sha256", ""))
                ).encode("ascii")
            ).hexdigest()
        )
        _require_sha256(self.task_key_sha256, "task key")
        self.journal_sink = journal_sink
        self.metrics: Mapping[str, int] = MappingProxyType({})
        self.journals: tuple[HopperClosureJournal, ...] = ()

    def build(
        self, *, resume: HopperClosureJournal | None = None
    ) -> PlatformTaskPayload:
        if resume is not None and (
            not isinstance(resume, HopperClosureJournal)
            or resume.task_key_sha256 != self.task_key_sha256
            or resume.common_artifact_sha256 != self.common.artifact_sha256
        ):
            raise ValueError("hopper closure resume authority differs")
        from .task_coverability import _build_hopper_task_coverability

        journals: list[HopperClosureJournal] = []

        def commit_round(state: Mapping[str, object]) -> None:
            reached = state["reached_positions"]
            assert isinstance(reached, Mapping)
            reached_um = np.ascontiguousarray(
                [
                    np.rint(
                        np.asarray(reached[cell], dtype=np.float64)
                        * 1_000_000.0
                    ).astype(np.int64)
                    for cell in sorted(reached)
                ],
                dtype=np.int64,
            ).reshape((-1, 3))
            observed = np.asarray(state["observed_detail"], dtype=np.bool_)
            packed = np.packbits(
                np.ascontiguousarray(observed).reshape(-1),
                bitorder="little",
            )
            serialized_edges = {
                f"{source[0] * self.common.geometry.span_cells + source[1]}:"
                f"{target[0] * self.common.geometry.span_cells + target[1]}": {
                    "disposition": value[0],
                    "reason_code": value[1],
                    "dependency_tile_indices": list(value[2]),
                }
                for (source, target), value in sorted(
                    state["edge_states"].items()
                )
            }
            serialized_edges["__state__"] = {
                "source_identity_sha256": self.common.source_identity_sha256,
                "platform_capability_sha256": self.platform.content_sha256,
                "start_identity_sha256": self.qualified_start[
                    "start_identity_sha256"
                ],
                "safe_union_bits_hex": np.packbits(
                    np.asarray(state["safe_union"], dtype=np.bool_).reshape(-1),
                    bitorder="little",
                ).tobytes().hex(),
                "processed_certified_edges": [
                    [
                        source[0] * self.common.geometry.span_cells + source[1],
                        target[0] * self.common.geometry.span_cells + target[1],
                    ]
                    for source, target in state[
                        "processed_certified_edges"
                    ]
                ],
                "landing_algorithm_id": state["landing_algorithm_id"],
                "edge_algorithm_id": state["edge_algorithm_id"],
                "first_round_candidate_count": state[
                    "first_round_candidate_count"
                ],
                "trajectory_edge_patch_count": state[
                    "trajectory_edge_patch_count"
                ],
                "logical_projection_call_count": state[
                    "logical_projection_call_count"
                ],
                "outside_roi_support_count": state[
                    "outside_roi_support_count"
                ],
            }
            journal = HopperClosureJournal.create(
                task_key_sha256=self.task_key_sha256,
                common_artifact_sha256=self.common.artifact_sha256,
                completed_round=int(state["completed_round"]),
                reached_positions_um=reached_um,
                observed_detail_bits=np.ascontiguousarray(packed),
                pending_sources=tuple(
                    cell[0] * self.common.geometry.span_cells + cell[1]
                    for cell in state["pending_sources"]
                ),
                edge_states=serialized_edges,
            )
            journals.append(journal)
            if self.journal_sink is not None:
                self.journal_sink(journal)

        metrics: dict[str, int] = {}
        resume_state = (
            None
            if resume is None
            else _decode_resume_state(
                journal=resume,
                common=self.common,
                platform=self.platform,
                qualified_start=self.qualified_start,
            )
        )
        payload = _build_hopper_task_coverability(
            common=self.common,
            platform=self.platform,
            qualified_start=self.qualified_start,
            bridge=self.bridge,
            incremental=True,
            round_callback=commit_round,
            metrics_output=metrics,
            resume_state=resume_state,
        )
        self.metrics = MappingProxyType(dict(metrics))
        self.journals = tuple(journals)
        return payload


__all__ = [
    "HopperClosureJournal",
    "HopperEdgeState",
    "HopperTaskClosureBuilder",
]

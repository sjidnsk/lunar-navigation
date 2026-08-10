"""复用 fed9 旧候选和 Observation V3 公式的运行时薄边界。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np

# 这些模块自身只包含观测/候选数学，不导入 reward、collector、optimizer 或 PPO。
from lunar_policy_training.environment.candidate_builder import (
    CandidateBatch,
    CandidateBuilderV2,
)
from lunar_policy_training.environment.observation_builder import (
    MissionRaster,
    ObservationBuilderV2,
    ObservedWorld,
    PlatformProjection,
    Pose2,
)
from lunar_policy_training.polar_data.raster import MapCanvas

from .identity import DecisionIdentity, IdentityMismatch


def _candidate_set_id(candidates: CandidateBatch) -> str:
    digest = hashlib.sha256()
    digest.update((candidates.canvas_id or "").encode("utf-8"))
    digest.update(candidates.features.tobytes(order="C"))
    digest.update(candidates.mask.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ObservationSnapshot:
    identity: DecisionIdentity
    arrays: Mapping[str, np.ndarray]
    candidates: CandidateBatch
    candidate_set_id: str
    canvas: MapCanvas
    elevation_m: np.ndarray
    platform_type: str

    def require_identity(self, expected: DecisionIdentity) -> None:
        if not isinstance(expected, DecisionIdentity):
            raise TypeError("expected identity must use DecisionIdentity")
        for name in (
            "mission_revision",
            "map_snapshot_id",
            "robot_state_id",
            "state_time_ns",
            "execution_state",
            "candidate_set_id",
        ):
            if getattr(expected, name) != getattr(self.identity, name):
                raise IdentityMismatch(f"identity {name} mismatch")


class Fed9ObservationRuntime:
    """以原始 fed9 builder 生成同一候选快照和七输入。"""

    def __init__(
        self,
        candidate_builder: CandidateBuilderV2,
        observation_builder: ObservationBuilderV2 | None = None,
    ) -> None:
        if not isinstance(candidate_builder, CandidateBuilderV2):
            raise TypeError("candidate_builder must use fed9 CandidateBuilderV2")
        self._candidate_builder = candidate_builder
        self._observation_builder = observation_builder or ObservationBuilderV2()

    def build(
        self,
        world: ObservedWorld,
        mission: MissionRaster,
        pose_map: Pose2,
        projection: PlatformProjection,
        platform_type: str,
        identity: DecisionIdentity,
    ) -> ObservationSnapshot:
        if not isinstance(identity, DecisionIdentity):
            raise TypeError("identity must use DecisionIdentity")
        candidates = self._candidate_builder.build(
            world,
            mission,
            pose_map,
            projection,
            platform_type=platform_type,
        )
        arrays = self._observation_builder.build(
            world,
            mission,
            pose_map,
            projection,
            candidates,
            platform_type,
        )
        candidate_set_id = _candidate_set_id(candidates)
        if identity.candidate_set_id and identity.candidate_set_id != candidate_set_id:
            raise IdentityMismatch("identity candidate_set_id mismatch")
        frozen_arrays: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            copied = np.ascontiguousarray(value.copy())
            copied.setflags(write=False)
            frozen_arrays[name] = copied
        frozen_elevation = np.ascontiguousarray(world.elevation_m.copy())
        frozen_elevation.setflags(write=False)
        full_identity = identity.with_candidate_set(candidate_set_id)
        return ObservationSnapshot(
            identity=full_identity,
            arrays=frozen_arrays,
            candidates=candidates,
            candidate_set_id=candidate_set_id,
            canvas=world.canvas,
            elevation_m=frozen_elevation,
            platform_type=platform_type,
        )


__all__ = ["Fed9ObservationRuntime", "ObservationSnapshot"]

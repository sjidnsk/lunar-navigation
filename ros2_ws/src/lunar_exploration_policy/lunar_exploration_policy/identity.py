"""一次策略决策边界的不可变生产者身份。"""

from __future__ import annotations

from dataclasses import dataclass, replace


class IdentityMismatch(ValueError):
    """动作或规划请求不再对应生成它的观测快照。"""


@dataclass(frozen=True, slots=True)
class DecisionIdentity:
    mission_revision: int
    map_snapshot_id: str
    robot_state_id: str
    state_time_ns: int
    execution_state: str
    candidate_set_id: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.mission_revision) is not int
            or self.mission_revision < 0
            or type(self.state_time_ns) is not int
            or self.state_time_ns < 0
        ):
            raise ValueError("identity revision/time must be non-negative integers")
        for name in (
            "map_snapshot_id",
            "robot_state_id",
            "execution_state",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"identity {name} must be non-empty")
        if not isinstance(self.candidate_set_id, str):
            raise ValueError("identity candidate_set_id must be a string")

    def with_candidate_set(self, candidate_set_id: str) -> "DecisionIdentity":
        if not candidate_set_id:
            raise ValueError("candidate_set_id must be non-empty")
        return replace(self, candidate_set_id=candidate_set_id)


__all__ = ["DecisionIdentity", "IdentityMismatch"]

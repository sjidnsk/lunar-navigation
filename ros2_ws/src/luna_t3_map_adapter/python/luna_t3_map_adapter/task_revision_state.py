"""Small, ROS-free active-task/map-revision state machine."""

from __future__ import annotations

from dataclasses import dataclass

from .global_map_cache import TaskRoi


ACTIVE = 1


@dataclass
class TaskRevisionState:
    mission_id: str | None = None
    mission_revision: int | None = None
    roi: TaskRoi | None = None
    map_revision: int | None = None
    active: bool = False

    def activate(self, mission_id: str, mission_revision: int, desired_state: int, roi: TaskRoi) -> bool:
        changed = (mission_id, mission_revision, roi) != (
            self.mission_id, self.mission_revision, self.roi
        )
        self.mission_id = mission_id
        self.mission_revision = mission_revision
        self.roi = roi
        self.active = desired_state == ACTIVE
        return self.active and changed and self.map_revision is not None

    def pause(self) -> bool:
        was_active = self.active
        self.active = False
        return was_active

    def observe_map_revision(self, revision: int) -> bool:
        if revision < 0 or (self.map_revision is not None and revision <= self.map_revision):
            return False
        self.map_revision = revision
        return self.active

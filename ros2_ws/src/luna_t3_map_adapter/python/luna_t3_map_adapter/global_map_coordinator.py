"""Coordinates task/revision events into complete global-map publications."""

from __future__ import annotations

from grid_map_msgs.msg import GridMap

from .global_grid_conversion import to_canonical_global_grid, to_grid_map_message
from .global_map_cache import GlobalMapCache, TaskRoi
from .task_revision_state import TaskRevisionState


class GlobalMapCoordinator:
    """Pure coordinator: an event yields a message only when a snapshot is valid."""

    def __init__(self, cache: GlobalMapCache) -> None:
        self._cache = cache
        self._state = TaskRevisionState()

    def accept_task(
        self,
        mission_id: str,
        mission_revision: int,
        desired_state: int,
        roi: TaskRoi,
    ) -> GridMap | None:
        if not self._state.activate(mission_id, mission_revision, desired_state, roi):
            return None
        return self._build()

    def accept_revision(self, map_revision: int) -> GridMap | None:
        if not self._state.observe_map_revision(map_revision):
            return None
        return self._build()

    def _build(self) -> GridMap:
        assert self._state.mission_id is not None
        assert self._state.mission_revision is not None
        assert self._state.roi is not None
        assert self._state.map_revision is not None
        snapshot = self._cache.refresh(
            self._state.mission_id,
            self._state.mission_revision,
            self._state.roi,
            self._state.map_revision,
        )
        return to_grid_map_message(to_canonical_global_grid(snapshot))

"""Deployment-safe task graph decision inputs; no ROS, torch or truth imports."""

import math
from dataclasses import replace
from .config import GraphConfig
from .graph import GraphBuilder
from .history import DirectionHistory
from .task_analysis import TaskAnalyzer


class DecisionCore:
    def __init__(self, task, sensor, config=None):
        self.config = config or GraphConfig()
        self.task, self.sensor = task, sensor
        self.analyzer = TaskAnalyzer(task, sensor)
        self.builder = GraphBuilder(self.config)
        self.history = DirectionHistory(self.config.history_tolerance_m)
        self._epoch = None

    @property
    def coverage(self):return self.analyzer.coverage

    def consume(self,snapshot):
        identity=(snapshot.epoch,snapshot.resolution_m,tuple(snapshot.origin))
        if self._epoch is not None and self._epoch != identity:
            self.history.clear()
        self._epoch = identity
        self.coverage.consume(snapshot)

    def observe(self, snapshot, velocity=(0.0, 0.0)):
        self.consume(snapshot)
        if (
            math.isfinite(snapshot.goal_position_tolerance_m)
            and snapshot.goal_position_tolerance_m > 0
        ):
            self.history.tolerance_m = snapshot.goal_position_tolerance_m
        report = self.analyzer.update(snapshot)
        self.builder.workspace = self.analyzer.workspace
        observation = self.builder.build(
            snapshot, report, self.history, self.task, self.sensor, velocity
        )
        if len(self.builder.unrepresented_interfaces) and not report.completed:
            report = replace(report, available=False, exhausted=False,
                             reason_code="UNREPRESENTED_TASK_INTERFACES")
        return observation, report

    def record_observation(self, pose, observed_cells):
        """Call after a real sensor update at its executed physical pose."""
        self.history.record(pose, observed_cells, self.sensor)

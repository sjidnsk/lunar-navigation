"""Current task observations never create obligations through unknown transit."""
from dataclasses import replace
import numpy as np
from lunar_drl_exploration.contracts import Pose, SensorSpec
from lunar_drl_exploration.decision import DecisionCore
from lunar_drl_exploration.task_analysis import TaskAnalyzer
from test_task_analysis import snapshot, task


def test_unknown_outside_bypass_does_not_prevent_task_exhaustion():
    m = np.full((12, 30), 2, np.uint8)
    m[5:8, 2:5] = 1
    m[2:5, 4:6] = 0
    m[2:4, 4:26] = 0
    m[3:8, 23:26] = 0
    s = snapshot(m, m, pose=Pose(3.5, 6.5, 0))
    report = TaskAnalyzer(task(1, 5, 27, 8), SensorSpec(range_m=3)).update(s)
    assert report.known_area_m2 > 0
    assert report.available and report.exhausted and report.completed
    assert len(report.frontier_cells) == 0


def test_far_outside_start_has_actions_without_invented_task_frontiers():
    m = np.zeros((16, 45), np.uint8)
    m[5:11, 2:9] = 1
    s = snapshot(m, pose=Pose(4.5, 7.5, 0))
    observation, report = DecisionCore(task(30, 5, 40, 12), SensorSpec(range_m=3)).observe(s)
    assert report.available and not report.completed
    assert report.reason_code == 'TASK_NOT_OBSERVED'
    assert report.known_area_m2 == 0 and len(report.frontier_cells) == 0
    assert len(observation.goals) > 0


def test_current_outside_stance_can_observe_task_across_nontraversable_band():
    m = np.full((12, 15), 2, np.uint8)
    b = m.copy()
    m[3:9, 2:5] = b[3:9, 2:5] = 1
    b[5:7, 5:8] = 1
    b[5:7, 8:10] = 0
    s = snapshot(m, b, pose=Pose(3.5, 5.5, 0))
    observation, report = DecisionCore(task(8, 5, 10, 7), SensorSpec(range_m=6)).observe(s)
    assert report.available and not report.completed
    assert len(report.frontier_cells) > 0
    assert np.all(report.witnesses[:, 0] < 8)
    assert observation.features[:, 3:11].max() > 0


def test_observation_consumes_frontier_without_changing_measured_area():
    m = np.ones((10, 10), np.uint8)
    m[5, 5] = 0
    analyzer = TaskAnalyzer(task(0, 0, 10, 10), SensorSpec(range_m=3))
    first = analyzer.update(snapshot(m))
    assert not first.completed and first.known_area_m2 == 99
    m[5, 5] = 2
    final = analyzer.update(replace(snapshot(m, m), revision=2))
    assert final.completed and final.known_area_m2 == 100 and final.new_area_m2 == 1


def test_small_task_hole_needs_observation_even_after_99_percent():
    m = np.ones((10, 10), np.uint8)
    m[5, 5] = 0
    report = TaskAnalyzer(task(0, 0, 10, 10), SensorSpec(range_m=3)).update(snapshot(m))
    assert report.known_area_m2 == 99
    assert report.available and not report.completed and len(report.frontier_cells)

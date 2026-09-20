"""Online exhaustion and truth coverage are independent task results."""
from dataclasses import replace
import numpy as np
from lunar_drl_exploration.contracts import Pose, SensorSpec
from lunar_drl_exploration.task_analysis import TaskAnalyzer
from test_task_analysis import snapshot, task


def test_sealed_unobserved_chamber_does_not_extend_observed_task():
    m = np.ones((15,15), np.uint8)
    m[4:11,4:11] = 2
    m[6:9,6:9] = 0
    r = TaskAnalyzer(task(0,0,15,15), SensorSpec(range_m=4)).update(snapshot(m,m))
    assert r.available and r.exhausted and r.completed
    assert r.known_area_m2 == 216 and len(r.frontier_cells) == 0


def test_missing_native_start_cannot_finish_even_fully_observed_task():
    s = replace(snapshot(np.ones((5,5), np.uint8)),
                start_connections=np.empty((0,2),int), start_connection_status='INPUT_UNAVAILABLE')
    r = TaskAnalyzer(task(0,0,5,5), SensorSpec()).update(s)
    assert not r.available and not r.completed


def test_no_task_observations_cannot_claim_completed_task():
    m = np.ones((15,15),np.uint8)
    m[4:11,4:11] = 2
    m[6:9,6:9] = 0
    r = TaskAnalyzer(task(6,6,9,9), SensorSpec(range_m=4)).update(snapshot(m,m))
    assert r.available and not r.completed and r.known_area_m2 == 0
    assert r.reason_code == 'TASK_NOT_OBSERVED'


def test_runtime_finishes_observed_task_without_external_frontier_obligation():
    from lunar_drl_exploration.runtime import InferenceRuntime
    from test_runtime import Adapter
    adapter = Adapter()
    adapter.snapshot = snapshot(np.ones((10,10),np.uint8), pose=Pose(2.5,2.5,0))
    def unexpected_policy(_):
        raise AssertionError('exhausted task must not dispatch a goal')
    runtime = InferenceRuntime(adapter, unexpected_policy, SensorSpec(range_m=3))
    runtime.start(task(0,0,10,10))
    runtime.tick()
    assert runtime.state == 'EXHAUSTED' and not adapter.inflight
    assert runtime.report.completed and runtime.report.exhausted
    assert runtime.status()['known_area_m2'] == 100
    assert not runtime.status()['reference_available']


def test_runtime_continues_after_99_percent_when_a_current_view_exists():
    from lunar_drl_exploration.runtime import InferenceRuntime
    from test_runtime import Adapter
    m = np.ones((10,10),np.uint8)
    m[5,5] = 0
    adapter = Adapter()
    adapter.snapshot = snapshot(m,pose=Pose(2.5,2.5,0))
    runtime = InferenceRuntime(adapter,lambda _:0,SensorSpec(range_m=3))
    runtime.start(task(0,0,10,10))
    runtime.tick()
    assert runtime.state == 'RUNNING' and adapter.inflight
    assert runtime.report.known_area_m2 == 99 and not runtime.report.completed

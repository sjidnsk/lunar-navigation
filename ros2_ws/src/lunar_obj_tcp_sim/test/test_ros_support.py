"""Behavior checks for the GridMap interface and observation evidence."""
from types import SimpleNamespace
import numpy as np
import pytest


def test_grid_map_rectangular_axes_and_unknown_are_preserved():
    from builtin_interfaces.msg import Time
    from lunar_obj_tcp_sim.ros_support import observation_grid_map
    observation = SimpleNamespace(elevation=np.array([[1., np.nan, 3.], [4., 5., 6.]]),
                                  origin_xy=(-2.4, 8.2), resolution=.2)
    msg = observation_grid_map(observation, Time(sec=7, nanosec=23))
    assert msg.header.frame_id == 'odom'
    assert msg.header.stamp.sec == 7 and msg.header.stamp.nanosec == 23
    assert msg.info.pose.position.x == pytest.approx(-2.1)
    assert msg.info.pose.position.y == pytest.approx(8.4)
    assert msg.layers == ['elevation']
    assert [d.size for d in msg.data[0].layout.dim] == [2, 3]
    assert [d.stride for d in msg.data[0].layout.dim] == [6, 3]
    np.testing.assert_allclose(msg.data[0].data, [6., 5., 4., 3., np.nan, 1.], equal_nan=True)


def test_grid_map_rejects_infinite_height_but_keeps_unknown():
    from builtin_interfaces.msg import Time
    from lunar_obj_tcp_sim.ros_support import observation_grid_map
    with pytest.raises(ValueError, match='elevation'):
        observation_grid_map(SimpleNamespace(elevation=np.array([[np.inf]]), origin_xy=(0,0), resolution=.2), Time())


def test_task_uses_map_bounds_not_zero_center():
    from builtin_interfaces.msg import Time
    from lunar_obj_tcp_sim.ros_support import exploration_task
    msg = exploration_task([10., -8., 30., 12.], Time(sec=5))
    assert [(p.x,p.y) for p in msg.boundary.points] == [(10.,-8.),(30.,-8.),(30.,12.),(10.,12.)]
    assert msg.header.frame_id == 'map' and msg.command == msg.START


def test_evidence_count_is_unique_and_excludes_task_halo():
    from lunar_obj_tcp_sim.observation_state import EvidenceCoverage
    evidence = EvidenceCoverage({'origin_xy':[-1.,-1.], 'resolution':1., 'shape':[4,4],
                                 'task_bounds_xy':[0.,0.,2.,2.]})
    obs=SimpleNamespace(elevation=np.ones((3,3)), origin_xy=(-1.,-1.), resolution=1.)
    assert evidence.add(obs) == 4
    assert evidence.add(obs) == 0
    assert evidence.observed_cells == 4 and evidence.total_cells == 4


def test_evidence_does_not_fill_nan_or_count_outside_map():
    from lunar_obj_tcp_sim.observation_state import EvidenceCoverage
    evidence=EvidenceCoverage({'origin_xy':[0.,0.], 'resolution':1., 'shape':[2,2],
                               'task_bounds_xy':[0.,0.,2.,2.]})
    obs=SimpleNamespace(elevation=np.array([[2.,np.nan],[3.,4.]]), origin_xy=(-1.,0.), resolution=1.)
    assert evidence.add(obs) == 1
    assert evidence.observed_cells == 1


def test_known_chunk_receipt_waits_for_new_mapper_input_count():
    from lunar_obj_tcp_sim.observation_state import MapReceipt
    receipt=MapReceipt()
    receipt.update(8,0,3,3)
    assert receipt.ready
    receipt.sent()
    assert not receipt.ready
    receipt.update(8,0,3,3)
    assert not receipt.ready
    receipt.update(9,1,4,3)
    assert not receipt.ready
    receipt.update(9,0,4,4)
    assert receipt.ready


def test_cancel_before_goal_acceptance_cancels_late_accepted_goal():
    from concurrent.futures import Future
    from lunar_obj_tcp_sim.observation_state import DeferredGoalCancellation
    acceptance=Future(); cancellation=Future()
    class Handle:
        accepted=True
        def cancel_goal_async(self):
            cancellation.set_result('cancellation sent')
            return cancellation
    request=DeferredGoalCancellation(acceptance)
    request.cancel()
    assert not cancellation.done()
    acceptance.set_result(Handle())
    assert cancellation.result()=='cancellation sent'
    request.cancel()  # Must not issue the same cancellation twice.


def test_cancel_after_rejected_goal_does_not_create_another_request():
    from concurrent.futures import Future
    from lunar_obj_tcp_sim.observation_state import DeferredGoalCancellation
    acceptance=Future(); request=DeferredGoalCancellation(acceptance)
    acceptance.set_result(SimpleNamespace(accepted=False))
    request.cancel()
    assert request.cancel_future is None

import math
from types import SimpleNamespace
import numpy as np
import pytest


def test_display_uses_rotated_sensor_offset_and_configured_window():
    from nav_msgs.msg import Odometry
    from lunar_obj_tcp_sim.visualization import scene_markers
    odom=Odometry(); odom.pose.pose.position.x=10.;odom.pose.pose.position.y=20.;odom.pose.pose.position.z=3.
    odom.pose.pose.orientation.z=math.sin(math.pi/4);odom.pose.pose.orientation.w=math.cos(math.pi/4)
    settings={'sensor_offset_xyz_m':[1.,0.,1.5],'sensor_range_m':12.,'sensor_fov_deg':120.,'local_window_size_m':64.}
    markers=scene_markers(odom,settings,[0.,0.,100.,100.]).markers
    origin=next(m for m in markers if m.ns=='sensor_origin')
    assert (origin.pose.position.x,origin.pose.position.y,origin.pose.position.z)==pytest.approx((10.,21.,4.5))
    window=next(m for m in markers if m.ns=='planning_window')
    assert max(p.x for p in window.points)-min(p.x for p in window.points)==64.
    assert min(p.y for p in window.points)==-12.


def test_truth_preview_omits_holes_and_does_not_mutate_map():
    from builtin_interfaces.msg import Time
    from lunar_obj_tcp_sim.visualization import truth_preview
    values=np.array([[1.,np.nan],[2.,3.]])
    terrain=SimpleNamespace(elevation=values,origin_xy=np.array([10.,20.]),resolution=.2)
    result=truth_preview(terrain,Time())
    assert len(result.markers[0].points)==3
    assert result.markers[0].ns=='terrain_truth'
    assert np.isnan(values[0,1])

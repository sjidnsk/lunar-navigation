"""Collector-owned true candidate descriptors, independent of learned policy."""
from dataclasses import replace
import numpy as np
import pytest
from lunar_drl_exploration.contracts import Pose, SensorSpec, TaskSpec, DecisionObservation
from lunar_drl_exploration.config import GraphConfig, load_platform_config
from lunar_drl_exploration.scene import TerrainGrid
from lunar_drl_exploration.reference import CoverageReference
from lunar_drl_exploration.sensor import SensorModel


def observation_at(points, yaws):
    points=np.asarray(points,float);yaws=np.asarray(yaws,float)
    nodes=np.repeat(np.arange(len(points)),len(yaws));headings=np.tile(yaws,len(points))
    f=np.zeros((len(points),19),np.float32);f[:,:2]=(points-points[0])/10
    return DecisionObservation(np.arange(len(points)),points,f,np.empty((0,2),int),np.empty(0),0,
        np.array([[0,0],[1,0],[1,1],[0,1]]),np.zeros(8),nodes,headings,
        np.column_stack((points[nodes],headings)),'fixture',1)


def fixture():
    terrain=TerrainGrid.from_heights(np.zeros((26,26),np.float32),.5,(0.,0.),load_platform_config())
    task=TaskSpec('p','map',[[0,0],[13,0],[13,13],[0,13]])
    sensor=SensorSpec(range_m=2.,fov_deg=90.)
    ref=CoverageReference.build(terrain,Pose(5.25,5.25,0),task,sensor)
    return terrain,task,sensor,ref


def test_truth_skeleton_and_candidate_gain_match_actual_static_sensor():
    from lunar_drl_exploration.privileged import build_truth, PrivilegedBuilder
    terrain,task,sensor,ref=fixture();scene=build_truth(terrain,ref)
    builder=PrivilegedBuilder(terrain,scene,task,sensor)
    obs=observation_at([[5.25,5.25],[6.25,5.25]],np.arange(8)*np.pi/4)
    known=np.zeros(terrain.shape,bool);state=builder.build(obs,np.packbits(known.ravel(),bitorder='little'))
    assert state.scene_id==ref.reference_id
    assert len(state.actions.positions)==2
    assert len(state.actions.gains)==16
    for i,goal in enumerate(obs.goals):
        measurement=SensorModel.observe(terrain,Pose(*goal),sensor)
        assert state.actions.gains[i]==pytest.approx(len(measurement.indices)*.25)
    assert not state.actions.gains.flags.writeable
    assert np.all(np.diff(state.actions.support_offsets)>0)
    assert np.all(state.actions.support_distances<=2.+1e-6)


def test_candidate_descriptor_uses_frozen_before_bits_and_survives_reordering():
    from lunar_drl_exploration.privileged import build_truth,PrivilegedBuilder
    terrain,task,sensor,ref=fixture();builder=PrivilegedBuilder(terrain,build_truth(terrain,ref),task,sensor)
    obs=observation_at([[5.25,5.25],[6.25,5.25]],[0.,np.pi])
    bits=np.packbits(np.zeros(terrain.shape,bool).ravel(),bitorder='little');before=builder.build(obs,bits)
    measured=SensorModel.observe(terrain,Pose(*obs.goals[0]),sensor)
    known=np.zeros(terrain.shape,bool);known.ravel()[measured.indices]=True
    after=builder.build(obs,np.packbits(known.ravel(),bitorder='little'))
    assert after.actions.gains[0]==0 and before.actions.gains[0]>0
    assert np.all(after.actions.gains<=before.actions.gains)
    order=np.array([3,1,0,2]);shuffled=replace(obs,goals=obs.goals[order],action_nodes=obs.action_nodes[order],action_yaws=obs.action_yaws[order])
    again=builder.build(shuffled,bits)
    np.testing.assert_array_equal(again.actions.gains,before.actions.gains[order])
    np.testing.assert_allclose(again.actions.positions[again.actions.action_positions],shuffled.goals[:,:2])
    bits[:]=255
    assert np.any(before.observed==0)


def test_continuous_query_includes_native_entry_cost_in_support_radius():
    from lunar_drl_exploration.privileged import build_truth,PrivilegedBuilder
    terrain,task,sensor,ref=fixture();scene=build_truth(terrain,ref)
    # A controlled skeleton node lies exactly 2m along the legal row from the
    # containing center; actual point is 0.24m before it. Old strict r loses it.
    scene=replace(scene,positions=np.array([[7.25,5.25]]),edges=np.empty((0,2),int),edge_lengths=np.empty(0),reference_offsets=np.array([0,len(scene.reference_indices)]))
    obs=observation_at([[5.01,5.25]],[0.])
    state=PrivilegedBuilder(terrain,scene,task,sensor).build(obs,np.packbits(np.zeros(terrain.shape,bool).ravel(),bitorder='little'))
    np.testing.assert_allclose(state.actions.support_distances,[2.24],atol=1e-6)
    np.testing.assert_array_equal(state.actions.support_indices,[0])


def test_empty_terminal_action_descriptor_has_no_invented_candidate():
    from lunar_drl_exploration.privileged import build_truth,PrivilegedBuilder
    terrain,task,sensor,ref=fixture();obs=observation_at([[5.25,5.25]],[0.])
    obs=replace(obs,action_nodes=np.empty(0,int),action_yaws=np.empty(0),goals=np.empty((0,3)))
    state=PrivilegedBuilder(terrain,build_truth(terrain,ref),task,sensor).build(obs,np.packbits(np.ones(terrain.shape,bool).ravel(),bitorder='little'))
    assert state.actions.positions.shape==(0,2) and len(state.actions.gains)==0
    np.testing.assert_array_equal(state.actions.support_offsets,[0])


def test_truth_zero_gain_is_feature_and_does_not_remove_transit_action():
    from lunar_drl_exploration.privileged import build_truth,PrivilegedBuilder
    terrain,task,sensor,ref=fixture();obs=observation_at([[5.25,5.25],[6.25,5.25]],[0.,np.pi])
    state=PrivilegedBuilder(terrain,build_truth(terrain,ref),task,sensor).build(obs,np.packbits(np.ones(terrain.shape,bool).ravel(),bitorder='little'))
    assert len(obs.goals)==4 and len(state.actions.gains)==4
    assert not np.any(state.actions.gains)

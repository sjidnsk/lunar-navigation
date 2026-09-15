import math
import numpy as np
from lunar_drl_exploration.contracts import Pose, SensorSpec
from lunar_drl_exploration.config import load_platform_config
from lunar_drl_exploration.scene import TerrainGrid
from lunar_drl_exploration.sensor import SensorModel
from lunar_drl_exploration.ros_messages import MeasurementBuffer, grid_message


def test_coalesced_frames_retain_only_actual_hits_until_native_ack():
    t=TerrainGrid.from_heights(np.zeros((100,100),np.float32),.2,(-10.,-10.),load_platform_config())
    a=SensorModel.observe(t,Pose(.1,.1,0),SensorSpec(range_m=2))
    b=SensorModel.observe(t,Pose(.1,.1,math.pi),SensorSpec(range_m=2))
    buf=MeasurementBuffer(t.origin,t.resolution_m)
    buf.add(a,10);buf.add(b,20)
    origin,planes=buf.payload()
    assert np.isfinite(planes[0]).sum()==len(np.union1d(a.indices,b.indices))
    assert planes.shape[1]<100 and planes.shape[2]<100
    # Ack for older publication cannot erase newly included measurements.
    buf.acknowledge(10)
    _,remaining=buf.payload()
    assert np.isfinite(remaining[0]).sum()==len(b.indices)
    buf.acknowledge(20)
    assert buf.payload() is None


def test_grid_wire_preserves_center_lattice_and_nan_holes():
    import pytest
    pytest.importorskip('grid_map_msgs')
    planes=np.full((5,3,4),np.nan,np.float32);planes[:,1,2]=[2,.1,.2,.3,1]
    msg=grid_message(planes,(-.4,.8),.2,1234)
    assert msg.info.pose.position.x==0
    assert msg.info.pose.position.y==1.1
    assert np.array_equal(np.asarray(msg.data[0].data).reshape(3,4)[::-1,::-1],planes[0],equal_nan=True)
    assert msg.header.stamp.nanosec==1234
    assert [d.size for d in msg.data[0].layout.dim]==[3,4]
    assert [d.stride for d in msg.data[0].layout.dim]==[12,4]


def test_transition_whole_action_baseline_and_final_budget_state():
    from lunar_drl_exploration.ros_env import make_transition
    from lunar_drl_exploration.contracts import PrivilegedState
    from test_task_analysis import snapshot
    from lunar_drl_exploration.contracts import TaskSpec
    from lunar_drl_exploration.decision import DecisionCore
    # Reuse actual graph contracts; progress polling is irrelevant to reward.
    snap=snapshot(np.ones((9,9),np.uint8))
    task=TaskSpec('t','map',np.array([[0,0],[2,0],[2,2],[0,2]]))
    obs,report=DecisionCore(task,SensorSpec()).observe(snap)
    priv=PrivilegedState('scene',np.array([3],np.uint8))
    t=make_transition(obs,0,obs,priv,priv,initial=(1.,2.,3.),final=(5.,12.,3.+math.pi),
        exhausted=False,budget_hit=True,episode_id='e',actor_version=7)
    assert t.reward == .04-.02-.005-.001
    assert t.parts.distance_m==10
    assert t.truncated and not t.terminated
    assert t.next_observation is obs and t.actor_version==7


def test_native_no_path_and_timeout_with_final_state_remain_transitions(monkeypatch):
    from lunar_drl_exploration.ros_env import RosExplorationEnv,InfrastructureError
    from lunar_drl_exploration.ros_messages import NavigationResult
    from lunar_drl_exploration.config import TrainingConfig
    from lunar_drl_exploration.reference import CoverageReference
    from lunar_drl_exploration.decision import DecisionCore
    from lunar_drl_exploration.plant import KinematicPlant
    from test_task_analysis import snapshot,task
    from types import SimpleNamespace
    cfg=TrainingConfig();env=RosExplorationEnv(cfg,0)
    t=TerrainGrid.from_heights(np.zeros((12,12),np.float32),1.,(0.,0.),cfg.platform)
    pose=Pose(3.5,3.5,0);s=snapshot(np.pad(np.ones((6,6),np.uint8),3),pose=pose)
    env.terrain=t;env.task=task(0,0,12,12);env.core=DecisionCore(env.task,SensorSpec(range_m=3))
    env.reference=CoverageReference.build(t,pose,env.task,SensorSpec(range_m=3))
    env.plant=KinematicPlant(t,pose,cfg.platform);env._apply_known(s)
    env.observation,env.report=env.core.observe(s);env.privileged=env._privileged()
    env._closed=False;env.steps=0;env.episode_id='failure-fixture'
    env.adapter=SimpleNamespace(velocity=(0.,0.),processed_stamp_ns=123)
    monkeypatch.setattr(env,'_snapshot',lambda:s)
    for outcome,reason in ((1,'NO_PATH'),(4,'TIMEOUT')):
        monkeypatch.setattr(env,'_execute_goal',lambda goal,o=outcome,r=reason:NavigationResult(o,r,0))
        before=env.observation
        result=env.step(0,actor_version=12)
        assert result.observation is before and result.next_observation.revision==s.revision
        assert result.reward==-.001 and not result.terminated and not result.truncated
        assert env.last_execution.reason_code==reason
    monkeypatch.setattr(env,'_snapshot',lambda:(_ for _ in ()).throw(InfrastructureError('lost map transport')))
    import pytest
    with pytest.raises(InfrastructureError,match='lost map'):env.step(0)


def test_deployment_history_pose_uses_actual_map_from_odom_transform():
    import pytest
    pytest.importorskip('geometry_msgs')
    from geometry_msgs.msg import TransformStamped
    from lunar_drl_exploration.ros_messages import odometry_message,pose_in_map
    msg=odometry_message(Pose(1.,2.,.25),.1,.2,100)
    tf=TransformStamped();tf.header.frame_id='map';tf.child_frame_id='odom'
    tf.transform.translation.x=10.;tf.transform.translation.y=-3.
    tf.transform.rotation.w=math.cos(math.pi/4);tf.transform.rotation.z=math.sin(math.pi/4)
    pose=pose_in_map(msg,tf)
    assert pose.x==pytest.approx(8.) and pose.y==pytest.approx(-2.)
    assert pose.yaw==pytest.approx(.25+math.pi/2)

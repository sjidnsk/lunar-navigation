"""Current decision input is distinct from retained measured evidence."""
import time
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from test_runtime import Adapter
from test_task_analysis import task


@pytest.fixture
def adapter():
    rclpy=pytest.importorskip('rclpy')
    from rclpy.context import Context
    from lunar_drl_exploration.ros_messages import RosNavigationAdapter
    context=Context();rclpy.init(context=context,domain_id=205)
    node=rclpy.create_node('adapter_readiness',context=context,enable_rosout=False)
    a=RosNavigationAdapter(node,action_name='/test/navigate',policy_map_service='/test/map',
        odometry_topic='/test/odom',diagnostics_topic='/test/diag',path_reference_topic='/test/ref',tf_topic='/test/tf')
    original=a.client
    yield a
    a.client=original;a.close();node.destroy_node();context.shutdown()


class Future:
    def __init__(self):self.response=None
    def done(self):return self.response is not None
    def result(self):return self.response


class Client:
    def __init__(self):self.ready=True;self.future=None;self.requests=[]
    def service_is_ready(self):return self.ready
    def call_async(self,request):
        self.requests.append(request);self.future=Future();return self.future


def reply(snap,*,ready=True,invalid=False):
    return SimpleNamespace(ready=ready,reason_code='READY' if ready else 'INPUT_UNAVAILABLE',
        epoch="replacement" if invalid else snap.epoch,fine_revision=snap.revision,full_snapshot=not invalid,
        resolution_m=snap.resolution_m,origin=snap.origin,
        pose=snap.pose,tiles=[dict(tile_x=k[0],tile_y=k[1],states=t.states,
            intrinsic_states=t.intrinsic_states,observed=t.observed,costs=t.costs,elevation_m=t.elevation_m)
            for k,t in snap.tiles.items()],start_connections=snap.start_connections,
        local_bounds=snap.local_bounds,profile_hash=snap.profile_hash,start_connection_status='READY',
        goal_position_tolerance_m=snap.goal_position_tolerance_m,
        goal_yaw_tolerance_rad=snap.goal_yaw_tolerance_rad,processed_stamp=SimpleNamespace(sec=1,nanosec=0))


@pytest.mark.parametrize('invalid',[False,True])
def test_unavailable_or_invalid_response_retains_cache_without_authorizing_it(adapter,invalid):
    a=adapter;c=Client();a.client=c;s=Adapter().snapshot
    assert a.poll_snapshot() is None
    c.future.response=reply(s)
    assert a.poll_snapshot() is not None
    saved=a.snapshot
    c.future.response=reply(s,ready=invalid,invalid=invalid)
    assert a.poll_snapshot() is None
    assert a.snapshot is saved
    c.ready=False
    assert a.poll_snapshot() is None
    c.ready=True;c.future.response=reply(s)
    assert a.poll_snapshot() is None
    c.future.response=reply(s)
    assert a.poll_snapshot() is not None


@pytest.mark.parametrize('boundary',['start','resume','completion'])
def test_decision_boundary_waits_for_response_requested_after_boundary(adapter,boundary,monkeypatch):
    from lunar_drl_exploration.runtime import InferenceRuntime
    from lunar_drl_exploration.ros_messages import NavigationResult
    from lunar_drl_exploration.contracts import SensorSpec
    a=adapter;c=Client();a.client=c;s=Adapter().snapshot
    a.poll_snapshot();c.future.response=reply(s);a.poll_snapshot()
    # A response issued before the boundary must not supply its current anchor.
    previous=c.future
    state=Adapter();a.velocity=(0.,0.)
    monkeypatch.setattr(a,'begin_goal',state.begin_goal)
    monkeypatch.setattr(a,'cancel_goal',state.cancel_goal)
    monkeypatch.setattr(a,'poll_goal',state.poll_goal)
    monkeypatch.setattr(type(a),'inflight',property(lambda self:state.inflight))
    runtime=InferenceRuntime(a,lambda obs:0,SensorSpec(range_m=3))
    if boundary=='start':runtime.start(task(0,0,9,9))
    else:
        runtime.start(task(0,0,9,9));runtime.core.consume(s)
        runtime.core.record_observation(s.pose,(1,))
        if boundary=='resume':runtime.state='PAUSED';runtime.resume()
        else:state.inflight=True;state.result=NavigationResult(0,'GOAL_REACHED',1)
    runtime.tick();assert not state.inflight
    previous.response=reply(s);runtime.tick()
    assert not state.inflight and runtime.state=='WAITING_FOR_INPUT'
    if boundary!='start':
        assert runtime.core.coverage.known_area_m2>0
        assert runtime.core.history.bits([[s.pose.x,s.pose.y]])[0,0]==1
    c.future.response=reply(s,ready=False);runtime.tick();assert not state.inflight
    c.ready=False;runtime.tick();assert not state.inflight
    c.ready=True;runtime.tick();assert not state.inflight
    c.future.response=reply(s);runtime.tick();assert not state.inflight
    from lunar_drl_exploration.contracts import Pose
    current=replace(s,pose=Pose(3.5,2.5,0.))
    c.future.response=reply(current);runtime.tick()
    assert a.snapshot.revision==s.revision and a.snapshot.pose==current.pose
    assert state.inflight and runtime.state=='RUNNING'
    if boundary!='start':assert runtime.core.history.bits([[s.pose.x,s.pose.y]])[0,0]==1


def test_best_effort_state_is_received_and_allows_measured_stop(adapter):
    import rclpy
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from nav_msgs.msg import Odometry
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import TransformStamped
    from lunar_drl_exploration.contracts import Pose
    from lunar_drl_exploration.ros_messages import odometry_message
    from lunar_drl_exploration.runtime import InferenceRuntime
    a=adapter;qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
    for sub in (a.subscriptions[0],a.subscriptions[3]):
        assert sub.qos_profile.reliability==qos.reliability
        assert sub.qos_profile.durability==qos.durability and sub.qos_profile.depth==10
    assert a.subscriptions[2].qos_profile.durability==DurabilityPolicy.TRANSIENT_LOCAL
    odom=a.node.create_publisher(Odometry,'/test/odom',qos)
    tf=a.node.create_publisher(TFMessage,'/test/tf',qos)
    transform=TransformStamped();transform.header.frame_id='map';transform.child_frame_id='odom'
    transform.transform.rotation.w=1.;transform.transform.translation.x=10.
    msg=odometry_message(Pose(1.,2.,0.),.005,.02,1_000_000_000);msg.header.frame_id='odom'
    from rclpy.executors import SingleThreadedExecutor
    executor=SingleThreadedExecutor(context=a.node.context);executor.add_node(a.node)
    try:
        deadline=time.monotonic()+3.
        while a.pose is None and time.monotonic()<deadline:
            tf.publish(TFMessage(transforms=[transform]));odom.publish(msg)
            executor.spin_once(timeout_sec=.02)
    finally:executor.shutdown()
    assert a.pose is not None and a.pose.x==11. and a.velocity==(.005,.02)
    runtime=InferenceRuntime(a,lambda obs:0);runtime.start(task(0,0,9,9));runtime.pause();runtime.tick()
    assert runtime.state=='PAUSED'
    runtime.cancel();runtime.tick();assert runtime.state=='CANCELED'

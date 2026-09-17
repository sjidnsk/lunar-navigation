"""Native Humble TCP + two-domain ROS test. Never connects to Unreal.
Run with candidate interface and P4 overlays sourced. Uses domains 181/182.
"""
import concurrent.futures
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import rclpy
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer, CancelResponse
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from std_msgs.msg import UInt8MultiArray, MultiArrayDimension
from grid_map_msgs.msg import GridMap
from lunar_car_ctrl.msg import VehicleTelemetry
from lunar_planning_msgs.action import NavigateToPose
from lunar_planning_msgs.msg import PathReference
from lunar_pure_exploration_msgs.msg import PureExplorationStatus
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'joint'))
from ue_path import encode_response

MAGIC=0x4C494441
ROOT=Path(__file__).resolve().parents[1]
def packet(command,payload):return struct.pack('<4I',MAGIC,command,len(payload),MAGIC^command^len(payload))+payload
server=socket.socket();server.bind(('127.0.0.1',0));server.listen(1);server.settimeout(15)
port=server.getsockname()[1];received=[];errors=[];running=True;wire=None;send_lock=threading.Lock()
def send(data):
    with send_lock:wire.sendall(data)
def serve():
    global wire
    try:
        wire,_=server.accept();wire.settimeout(.02);buf=b'';last=0
        while running:
            if time.monotonic()-last>.05:
                raw=struct.pack('<I21f',0xEB93EB93,*([0.]*8+[100.,200.,1.,0.,-math.sqrt(.5),0.,math.sqrt(.5)]+[0.]*6))
                send(packet(4,raw));last=time.monotonic()
            try:data=wire.recv(8192)
            except socket.timeout:continue
            if not data:break
            buf+=data
            while len(buf)>=16:
                magic,cmd,size,checksum=struct.unpack('<4I',buf[:16]);assert magic==MAGIC and checksum==magic^cmd^size
                if len(buf)<16+size:break
                payload=buf[16:16+size];buf=buf[16+size:];received.append((cmd,payload))
    except Exception as e:
        if running:errors.append(str(e))
thread=threading.Thread(target=serve);thread.start()
children=[];contexts=[];executors=[];threads=[];logs=[]
def spawn(cmd,env,name):
    f=open('/tmp/p4-ue-test-'+name+'.log','w');logs.append(f)
    p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(p);return p

def wait(condition,seconds=12):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        assert not errors,errors
        if condition():return
        for p in children:assert p.poll() is None,('child exited',p.pid,p.returncode)
        time.sleep(.03)
    raise AssertionError('Timed out waiting for condition')
def responses(req):return [p for c,p in received if c==8 and struct.unpack_from('<I',p)[0]==req]
try:
    env=dict(os.environ,ROS_DOMAIN_ID='181',ROS_LOCALHOST_ONLY='0',RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',PYTHONNOUSERSITE='1')
    binary='/home/yanfa/P4/vehicle_interface/install/lunar_car_ctrl/lib/lunar_car_ctrl/lunar_car_node'
    spawn([binary,'--ros-args','-p','tcp_host:=127.0.0.1','-p',f'tcp_port:={port}','-p','max_linear_speed:=0.2'],env,'interface')
    nodes=[]
    for domain in (181,182):
        c=Context();rclpy.init(context=c,domain_id=domain);contexts.append(c)
        n=rclpy.create_node('p4_ue_native_test_'+str(domain),context=c);nodes.append(n)
        e=MultiThreadedExecutor(num_threads=4,context=c);e.add_node(n);executors.append(e)
        t=threading.Thread(target=e.spin);t.start();threads.append(t)
    old,p4=nodes
    requests=[];telemetry=[]
    old.create_subscription(UInt8MultiArray,'/car/ue_path_request',lambda m:requests.append(bytes(m.data)),10)
    old.create_subscription(VehicleTelemetry,'/car/telemetry',lambda m:telemetry.append(m),10)
    pub=old.create_publisher(UInt8MultiArray,'/car/ue_path_response',10)
    wait(lambda:wire is not None and len(telemetry)>2)
    data=packet(7,struct.pack('<I3f',17,10200.,20000.,100.))
    with send_lock:
        for lo,hi in [(0,2),(2,9),(9,20),(20,len(data))]:
            wire.sendall(data[lo:hi]);time.sleep(.02)
    wait(lambda:requests)
    assert requests[-1]==data[16:]
    payload=encode_response(17,0,[(10000.,20000.,100.),(10200.,20000.,100.)])
    pub.publish(UInt8MultiArray(data=list(payload)));wait(lambda:responses(17))
    assert responses(17)[-1]==payload
    # Invalid response shape and wrong ID must never enter wire.
    count=len(responses(17));pub.publish(UInt8MultiArray(data=list(payload[:-1])))
    pub.publish(UInt8MultiArray(data=list(encode_response(99,2,[]))))
    time.sleep(.3);assert len(responses(17))==count and not responses(99)
    # Coalesced packets remain separate and do not interrupt telemetry.
    send(packet(7,struct.pack('<I3f',18,10000,20000,100))+packet(7,struct.pack('<I3f',19,10000,20000,100)))
    wait(lambda:len(requests)>=3)
    assert [struct.unpack_from('<I',x)[0] for x in requests[-2:]]==[18,19]
    print('PASS native interface: fragmented/coalesced request, multiplexed telemetry, exact response, invalid response rejection',flush=True)
    pose_pub=p4.create_publisher(Odometry,'/Car/T3/localization/odometry',10)
    tf_pub=p4.create_publisher(TFMessage,'/P4/input/map_to_odom',10)
    grid_pub=p4.create_publisher(GridMap,'/Car/T3/mapping/grid_map',10)
    state_pub=p4.create_publisher(PureExplorationStatus,'/Car/T4/exploration/status',10)
    ref_pub=p4.create_publisher(PathReference,'/Car/T4/planning/path_reference',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    grid=GridMap();grid.header.frame_id='map';grid.info.resolution=1.;grid.info.length_x=10.;grid.info.length_y=10.;grid.info.pose.orientation.w=1.;grid.layers=['elevation']
    from std_msgs.msg import Float32MultiArray
    arr=Float32MultiArray(data=[0.]*100);arr.layout.dim=[MultiArrayDimension(label='column_index',size=10,stride=100),MultiArrayDimension(label='row_index',size=10,stride=10)];grid.data=[arr]
    exploration_state=[PureExplorationStatus.EXECUTING]
    pauses=[]
    from lunar_pure_exploration_msgs.msg import PureExplorationTask
    def on_task(m):
        if m.command==m.PAUSE:
            pauses.append(m.task_id);exploration_state[0]=PureExplorationStatus.PAUSED
    p4.create_subscription(PureExplorationTask,'/Car/T4/exploration/task',on_task,10)
    def publish_inputs():
        m=Odometry();m.header.frame_id='odom';m.child_frame_id='base_link';m.header.stamp=p4.get_clock().now().to_msg();m.pose.pose.orientation.w=1.;pose_pub.publish(m)
        tf=TransformStamped();tf.header.frame_id='map';tf.child_frame_id='odom';tf.transform.rotation.w=1.;tf_pub.publish(TFMessage(transforms=[tf]));grid_pub.publish(grid)
        state_pub.publish(PureExplorationStatus(task_id='loopback-exploration',state=exploration_state[0]))
    p4.create_timer(.1,publish_inputs)
    goals=[];handles=[]
    def execute(handle):
        goals.append(handle.request);handles.append(handle)
        end=time.monotonic()+20
        while running and time.monotonic()<end:
            if handle.is_cancel_requested:
                handle.canceled();return NavigateToPose.Result(outcome=5,reason_code='CANCELED')
            time.sleep(.03)
        handle.abort();return NavigateToPose.Result(outcome=4,reason_code='TEST_TIMEOUT')
    action=ActionServer(p4,NavigateToPose,'/Car/T4/navigation/navigate_to_pose',execute_callback=execute,cancel_callback=lambda _:CancelResponse.ACCEPT,
                        callback_group=__import__('rclpy.callback_groups',fromlist=['ReentrantCallbackGroup']).ReentrantCallbackGroup())
    # Use only staged joint files; do not touch the deployed adapter during tests.
    spawn(['/usr/bin/python3',str(ROOT/'joint/legacy_feedback_input.py')],dict(env,ROS_DOMAIN_ID='182',P4_LEGACY_DOMAIN='181'),'adapter')
    time.sleep(8)
    send(packet(7,struct.pack('<I3f',200,10200.,20000.,100.)))
    wait(lambda:len(goals)==1,20)
    assert pauses==['loopback-exploration']
    assert abs(goals[0].target_x_m-2)<1e-5 and abs(goals[0].target_y_m)<1e-5 and not goals[0].has_target_yaw
    def reference(handle,rev=1,state=0,wrong=False):
        ref=PathReference();ref.session_id.uuid=list(bytes(handle.goal_id.uuid));ref.segment_revision=rev;ref.state=state;ref.reaches_final_goal=True;ref.path.header.frame_id='map'
        if wrong:ref.session_id.uuid=[0]*16
        for x in (0.,2.):
            p=PoseStamped();p.header.frame_id='map';p.pose.position.x=x;p.pose.orientation.w=1.;ref.path.poses.append(p)
        ref_pub.publish(ref)
    reference(handles[0],wrong=True);time.sleep(.3);assert not responses(200)
    reference(handles[0]);wait(lambda:responses(200))
    output=responses(200)[-1];v=struct.unpack('<IB3xI6f',output)
    assert v[:3]==(200,0,2) and all(abs(a-b)<.02 for a,b in zip(v[3:],(10000,20000,100,10200,20000,100))),v
    count=len(responses(200));reference(handles[0],rev=0);time.sleep(.3);assert len(responses(200))==count
    reference(handles[0],state=1);wait(lambda:len(responses(200))==count+1)
    assert responses(200)[-1][4]==2
    reference(handles[0],rev=2);wait(lambda:responses(200)[-1][4]==0)
    send(packet(7,struct.pack('<I3f',201,10200.,20000.,100.)));wait(lambda:len(goals)==2,10)
    reference(handles[0],rev=3);time.sleep(.3);assert not responses(201)
    reference(handles[1]);wait(lambda:responses(201))
    assert responses(201)[-1][4]==0 and len(telemetry)>80
    send(packet(7,struct.pack('<I3f',202,float('nan'),20000.,100.)))
    wait(lambda:responses(202))
    assert responses(202)[-1][4]==2
    time.sleep(.3);assert len(goals)==2
    print('PASS native adapter: shared alignment, centimetre roundtrip, no yaw, session/revision filter, invalidation, new goal cancels old',flush=True)
    print(json.dumps({'telemetry_samples':len(telemetry),'requests':len(requests),'goals':len(goals),'responses':sum(c==8 for c,p in received)}),flush=True)
finally:
    running=False
    time.sleep(.2)
    # Stop adapter first so action handles can cancel before the fake server exits.
    for p in reversed(children):
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGINT)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
    running=False
    if wire:wire.close()
    server.close();thread.join(timeout=3)
    for e in executors:e.shutdown(timeout_sec=2)
    for n in locals().get('nodes',[]):n.destroy_node()
    for c in contexts:c.shutdown()
    for t in threads:t.join(timeout=2)
    for f in logs:f.close()

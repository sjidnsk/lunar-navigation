"""Synchronous isolated native-navigation/public-controller training environment.

Task8 owns workers, asynchronous dispatch, retained completions/ACK and recovery.
This object never imports Torch and never resets inside step().
"""
from dataclasses import dataclass
import math
import os
import threading
import time
import uuid
import numpy as np
from .contracts import Pose, PrivilegedState, RewardParts, Transition
from .config import GraphConfig
from .decision import DecisionCore
from .plant import KinematicPlant, GeometryFailure
from .processes import OwnedProcesses
from .ros_messages import (InfrastructureError, MeasurementBuffer, RosNavigationAdapter,
                           NavigationResult, grid_message, odometry_message, ros_time, velocity_is_stopped)


class InputUnavailable(RuntimeError): pass
class TaskCanceled(RuntimeError): pass


@dataclass(frozen=True)
class ExecutionMetadata:
    goal: tuple
    outcome: int
    reason_code: str
    last_segment_revision: int
    goal_cell: tuple
    actual_cell: tuple
    actual_pose: tuple
    distance_m: float
    turn_rad: float
    simulation_s: float
    wall_s: float
    map_revision: int
    map_processed_stamp_ns: int
    geometry_failure: bool = False


def make_transition(observation,action,next_observation,privileged,next_privileged,*,
                    initial,final,completed,budget_hit,episode_id,actor_version):
    area,distance,turn=(float(b-a) for a,b in zip(initial,final))
    if area < -1e-8 or distance < -1e-8 or turn < -1e-8:
        raise InfrastructureError('nonmonotonic whole-action accounting')
    parts=RewardParts(area,distance,turn)
    reward=area/100.-.02*distance/10.-.005*turn/math.pi-.001
    return Transition(observation,action,reward,next_observation,privileged,next_privileged,
        parts,bool(completed),bool(budget_hit and not completed),episode_id,int(actor_version))


class RosExplorationEnv:
    def __init__(self,config,env_id,*,domain_base=210,wall_timeout_s=120.):
        self.config=config;self.env_id=int(env_id)
        self.namespace=f'/lunar_training/env_{self.env_id}'
        self.domain_id=domain_base+self.env_id
        if not 0 <= self.domain_id <= 232: raise ValueError('invalid isolated DDS domain')
        self.wall_timeout_s=float(wall_timeout_s)
        self.owned=OwnedProcesses();self.node=None;self.context=None;self.adapter=None;self.executor=None;self._spin_error=None;self._spin_thread=None
        self._cancel=threading.Event();self._closed=True;self.actor_version=0
        self.episode_budget=config.curriculum_budgets[0]
        self.last_execution=None;self.episode_metadata={};self.static_scenes={}
        self._step_active=False

    @property
    def owned_pids(self): return self.owned.pids

    def reset(self,seed,family,extent,*,episode_budget=None):
        self.close()
        from .scene import Scene,TerrainGrid
        scene=Scene(seed,family,extent,platform=self.config.platform)
        terrain=TerrainGrid.from_scene(scene)
        self.episode_budget=int(episode_budget) if episode_budget is not None else self.config.curriculum_budgets[0 if extent<=80 else 1 if extent<=150 else 2]
        if self.episode_budget<=0:raise ValueError("positive episode budget required")
        return self.reset_scene(terrain,scene.initial_pose(terrain),scene.task,
                                descriptor={'seed':seed,'family':family,'extent_m':extent})

    def reset_scene(self,terrain,pose,task,*,descriptor=None):
        """Explicit deterministic fixture entry; identical sensor/navigation path."""
        self.close()
        from .reference import CoverageReference
        from .privileged import build_truth, PrivilegedBuilder
        self.terrain=terrain;self.task=task
        self.plant=KinematicPlant(terrain,pose,self.config.platform)
        self.reference=CoverageReference.build(terrain,pose,task,self.config.sensor)
        static=build_truth(terrain,self.reference,GraphConfig(platform=self.config.platform))
        self.privileged_builder=PrivilegedBuilder(terrain,static,task,self.config.sensor,
            GraphConfig(platform=self.config.platform))
        self.static_scenes={static.scene_id:static}
        self.core=DecisionCore(task,self.config.sensor,GraphConfig(platform=self.config.platform),
                               coverage_target=self.config.coverage_target)
        self.core.history.tolerance_m=self.config.goal_position_tolerance_m
        self.buffer=MeasurementBuffer(terrain.origin,terrain.resolution_m)
        self._cancel.clear();self._closed=False;self.steps=0;self.last_execution=None
        self.report=None
        self.episode_id=uuid.uuid4().hex
        self._sim_ns=1_000_000_000;self._last_sensor_ns=0;self._latest_sensor_ns=0
        self._command=(0.,0.);self._next_tick=time.monotonic()
        self._start_wall=time.monotonic();self._max_command_speed=0.;self._sensor_count=0
        try:
            self._spin_error=None
            self._start_ros()
            self._wait(lambda:self.adapter.ready,'native/controller discovery')
            # Wait for actual observed measurements, with continuing physical ticks.
            snap=self._snapshot()
            initial_turn=self.plant.turn_rad;initial_time=self.plant.simulation_s
            # A finite initialization scan uses exactly the same native action,
            # public controller and real sensor path as a policy action.
            self.observation,self.report=self.core.observe(snap,self.adapter.velocity)
            for quarter in range(4):
                if len(snap.start_connections) and self.report.available: break
                goal=(self.plant.pose.x,self.plant.pose.y,pose.yaw+(quarter+1)*math.pi/2)
                result=self._execute_goal(goal)
                snap=self._snapshot()
                if result.outcome!=0:
                    raise InputUnavailable(f'initialization {result.reason_code}')
                self.observation,self.report=self.core.observe(snap,self.adapter.velocity)
            if not len(snap.start_connections):
                raise InputUnavailable('bounded initialization produced no native start connection')
            if not self.report.available: raise InputUnavailable(self.report.reason_code)
            self.privileged=self._privileged()
            self.episode_metadata=dict(descriptor or {},episode_id=self.episode_id,
                scene_id=static.scene_id,reference_area_m2=self.reference.area_m2,
                initial_known_area_m2=self._known_area,
                initialization_turn_rad=self.plant.turn_rad-initial_turn,
                initialization_simulation_s=self.plant.simulation_s-initial_time,
                initialization_wall_s=time.monotonic()-self._start_wall,
                budget=self.episode_budget,domain_id=self.domain_id,namespace=self.namespace)
            return self.observation,self.privileged
        except BaseException:
            self.close();raise

    def _start_ros(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.parameter import Parameter
        from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from tf2_msgs.msg import TFMessage
        from grid_map_msgs.msg import GridMap
        from rosgraph_msgs.msg import Clock
        self.context=Context();rclpy.init(context=self.context,domain_id=self.domain_id)
        self.node=rclpy.create_node('drl_environment',namespace=self.namespace,context=self.context,
            enable_rosout=False,parameter_overrides=[Parameter('use_sim_time',value=True)])
        from rclpy.executors import SingleThreadedExecutor
        self.executor=SingleThreadedExecutor(context=self.context);self.executor.add_node(self.node)
        ns=self.namespace
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._clock_pub=self.node.create_publisher(Clock,'/clock',10)
        self._odom_pub=self.node.create_publisher(Odometry,ns+'/odom',10)
        self._tf_pub=self.node.create_publisher(TFMessage,ns+'/tf',10)
        self._map_pub=self.node.create_publisher(GridMap,ns+'/grid',qos)
        self._cmd_sub=self.node.create_subscription(Twist,ns+'/cmd_vel',self._on_command,10)
        self.adapter=RosNavigationAdapter(self.node,action_name=ns+'/navigate',policy_map_service=ns+'/get_map',
            odometry_topic=ns+'/odom',diagnostics_topic=ns+'/diagnostics',path_reference_topic=ns+'/reference',tf_topic=ns+'/tf')
        def spin():
            try:self.executor.spin()
            except Exception as exc:self._spin_error=exc
        self._spin_thread=threading.Thread(target=spin,daemon=True);self._spin_thread.start()
        env=dict(os.environ,ROS_DOMAIN_ID=str(self.domain_id),ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')
        common={'platform_config':self.config.platform.capability_path,'use_sim_time':True,
                'goal_position_tolerance_m':self.config.goal_position_tolerance_m,
                'goal_yaw_tolerance_rad':math.pi/12,'odometry_topic':ns+'/odom','tf_topic':ns+'/tf'}
        self.owned.start('lunar_incremental_navigation_ros','lunar_incremental_navigation_node',ns,
            dict(common,local_map_topic=ns+'/grid',action_name=ns+'/navigate',path_reference_topic=ns+'/reference',
                local_path_topic=ns+'/path',global_route_topic=ns+'/route',diagnostics_topic=ns+'/diagnostics',
                exploration_map_topic=ns+'/map',policy_map_service=ns+'/get_map',tracking_status_topic=ns+'/tracking',
                enable_tracking_feedback=True),env)
        self.owned.start('lunar_pure_wheeled_controller','lunar_pure_wheeled_controller_node.py',ns,
            dict(common,input_mode='incremental_reference',path_reference_topic=ns+'/reference',
                tracking_status_topic=ns+'/tracking',reference_topic=ns+'/wheeled_reference',path_topic=ns+'/path',
                command_topic=ns+'/cmd_vel',execution_cancel_topic=ns+'/cancel',
                max_linear_mps=.2,max_angular_radps=1.,control_rate_hz=20.),env)

    def _on_command(self,msg):
        self._command=(msg.linear.x,msg.angular.z)
        self._max_command_speed=max(self._max_command_speed,abs(msg.linear.x))
        if abs(msg.linear.y)>1e-12: raise InfrastructureError('public controller issued lateral motion')

    def _pump(self):
        import rclpy
        from rosgraph_msgs.msg import Clock
        from geometry_msgs.msg import TransformStamped
        from tf2_msgs.msg import TFMessage
        from .sensor import SensorModel
        try:self.owned.check()
        except RuntimeError as exc:raise InfrastructureError(str(exc)) from exc
        # Drain incoming commands before each physical integration. No simulated
        # time catch-up or omitted integration/sensor ticks when computation lags.
        if self._spin_error is not None:raise InfrastructureError(str(self._spin_error))
        if not self.plant.collision:
            try:self.plant.advance(*self._command,self.config.integration_step_s)
            except GeometryFailure:pass
        self._sim_ns+=round(self.config.integration_step_s*1e9)
        stamp=ros_time(self._sim_ns)
        self._clock_pub.publish(Clock(clock=stamp))
        p=self.plant.pose
        x,y=self.terrain.world_to_cell(p.x,p.y)
        self._odom_pub.publish(odometry_message(p,self.plant.linear_mps,self.plant.angular_radps,
            self._sim_ns,self.terrain.heights[y,x]))
        tf=TransformStamped();tf.header.stamp=stamp;tf.header.frame_id='map';tf.child_frame_id='odom';tf.transform.rotation.w=1.
        self._tf_pub.publish(TFMessage(transforms=[tf]))
        if self._sim_ns-self._last_sensor_ns>=round(1e9/self.config.observation_hz):
            hits=SensorModel.observe(self.terrain,p,self.config.sensor)
            self.buffer.add(hits,self._sim_ns)
            self.core.record_observation(p,hits.indices)
            payload=self.buffer.payload()
            if payload is not None:
                origin,planes=payload
                self._map_pub.publish(grid_message(planes,origin,self.terrain.resolution_m,self._sim_ns))
            self._last_sensor_ns=self._latest_sensor_ns=self._sim_ns;self._sensor_count+=1
        snap=self.adapter.poll_snapshot()
        if snap is not None:
            self.buffer.acknowledge(self.adapter.processed_stamp_ns)
            self._apply_known(snap)
        self._pace()

    def _pace(self):
        # Total wall-step budget includes computation. Overload resets the
        # deadline without omitting physical steps or accumulating catch-up.
        self._next_tick+=self.config.integration_step_s/self.config.target_rtf
        now=time.monotonic()
        remaining=self._next_tick-now
        if remaining>0:time.sleep(remaining)
        else:self._next_tick=now

    @property
    def _known_area(self):return self.core.coverage.known_area_m2

    def _apply_known(self,snap):
        self.core.consume(snap)

    def _wait(self,predicate,reason,*,timeout=None):
        deadline=time.monotonic()+(self.wall_timeout_s if timeout is None else timeout)
        while True:
            result=predicate()
            if result:return result
            if time.monotonic()>deadline:raise InfrastructureError(f'wall watchdog: {reason}; map={self.adapter.map_reason}; '+self.owned.error_tail())
            self._pump()

    def _snapshot(self):
        minimum=self._latest_sensor_ns
        snap=self._wait(lambda:self.adapter.poll_snapshot(minimum),'processed final policy map')
        self.buffer.acknowledge(self.adapter.processed_stamp_ns);self._apply_known(snap)
        return snap

    def _execute_goal(self,goal):
        self.adapter.begin_goal(goal)
        deadline=time.monotonic()+self.wall_timeout_s
        result=None
        while result is None:
            if self._cancel.is_set() or self.plant.collision:self.adapter.cancel_goal()
            result=self.adapter.poll_goal()
            if result is not None:break
            if time.monotonic()>deadline:
                self.adapter.cancel_goal()
                raise InfrastructureError('navigation transport/execution wall watchdog; '+self.owned.error_tail())
            self._pump()
        # Result completion can precede the controller's last braking command.
        self._wait(lambda:velocity_is_stopped((self.plant.linear_mps,self.plant.angular_radps)),
                   'cancel/goal stop',timeout=10.)
        if self._cancel.is_set():raise TaskCanceled('task canceled after native cancel-and-stop')
        if self.plant.collision:return NavigationResult(6,'COLLISION',result.last_segment_revision)
        return result

    def _privileged(self, observation=None):
        identity=self.core.coverage.identity
        if identity is None:raise InfrastructureError('coverage is unavailable')
        resolution,origin=identity[1:]
        if abs(resolution-self.terrain.resolution_m)>1e-12:
            raise InfrastructureError('native/reference resolution mismatch')
        offset=(np.asarray(self.terrain.origin[:2])-origin[:2])/resolution
        rounded=np.rint(offset).astype(int)
        if np.any(np.abs(offset-rounded)>1e-6):raise InfrastructureError('native/reference lattice mismatch')
        x,y=map(int,rounded);h,w=self.terrain.shape
        mask=self.core.coverage.mask((x,y,x+w,y+h))
        return self.privileged_builder.build(self.observation if observation is None else observation,
                                             self.reference.pack(mask))

    def step(self,action_index,*,actor_version=None):
        if self._closed or self._step_active:raise RuntimeError('environment is closed or already stepping')
        if self.report.completed or self.steps>=self.episode_budget:raise RuntimeError('episode requires reset')
        if self.plant.collision:raise GeometryFailure('collision episode requires reset')
        if not 0<=int(action_index)<len(self.observation.goals):raise ValueError('action outside frozen observation')
        action_index=int(action_index);goal=tuple(self.observation.goals[action_index])
        before=(self._known_area,self.plant.distance_m,self.plant.turn_rad)
        sim=self.plant.simulation_s;wall=time.monotonic();version=self.actor_version if actor_version is None else int(actor_version)
        if version<0:raise ValueError("actor_version must be nonnegative")
        observation,privileged=self.observation,self.privileged
        self._step_active=True
        try:
            result=self._execute_goal(goal)
            snap=self._snapshot()
            next_observation,report=self.core.observe(snap,self.adapter.velocity)
            if not report.available:raise InfrastructureError('unconstructable final state: '+report.reason_code)
            next_privileged=self._privileged(next_observation);self.steps+=1
            final=(self._known_area,self.plant.distance_m,self.plant.turn_rad)
            transition=make_transition(observation,action_index,next_observation,privileged,next_privileged,
                initial=before,final=final,completed=report.completed,budget_hit=self.steps>=self.episode_budget,
                episode_id=self.episode_id,actor_version=version)
            p=self.plant.pose
            self.last_execution=ExecutionMetadata(goal,result.outcome,result.reason_code,result.last_segment_revision,
                self.terrain.world_to_cell(*goal[:2]),self.terrain.world_to_cell(p.x,p.y),(p.x,p.y,p.yaw),
                final[1]-before[1],final[2]-before[2],self.plant.simulation_s-sim,time.monotonic()-wall,
                snap.revision,self.adapter.processed_stamp_ns,self.plant.collision)
            self.observation,self.report,self.privileged=next_observation,report,next_privileged
            return transition
        finally:self._step_active=False

    def progress(self):
        """Cheap snapshot only: never advances TaskAnalyzer's reward baseline."""
        if self._closed:return {'state':'CLOSED','owned_pids':self.owned_pids}
        report=getattr(self,'report',None)
        return {'state':'EXECUTING' if self._step_active else 'READY','episode_id':self.episode_id,
            'steps':self.steps,'known_area_m2':self._known_area,'distance_m':self.plant.distance_m,
            'turn_rad':self.plant.turn_rad,'simulation_s':self.plant.simulation_s,
            'wall_s':time.monotonic()-self._start_wall,'sensor_frames':self._sensor_count,
            'max_command_speed_mps':self._max_command_speed,'owned_pids':self.owned_pids,
            'completed':None if report is None else report.completed,
            'exhausted':None if report is None else report.exhausted,
            'remaining_area_upper_m2':None if report is None else report.remaining_area_upper_m2,
            'coverage_lower_bound':None if report is None else report.coverage_lower_bound}

    def cancel(self):
        """Thread-safe request; active step owns ROS cancel-and-stop pumping."""
        self._cancel.set()

    def _release_scene(self):
        # Replay/collector own their explicit immutable static-scene references.
        # A closed worker must not retain terrain or an old native/map graph.
        for name in ('terrain','plant','reference','core','buffer','observation','privileged','privileged_builder'):
            if hasattr(self,name):setattr(self,name,None)
        self.static_scenes={}

    def close(self):
        if self._closed:
            self._release_scene();return
        self._cancel.set()
        if self._step_active:raise RuntimeError('join active synchronous step after cancel before close')
        try:
            if self.adapter is not None and self.adapter.inflight:
                self.adapter.cancel_goal()
                try:self._wait(lambda:self.adapter.poll_goal(),'close cancellation',timeout=5.)
                except Exception:pass
        finally:
            self.owned.close()
            if self.executor is not None:self.executor.shutdown();self.executor=None
            if self._spin_thread is not None:self._spin_thread.join(timeout=3);self._spin_thread=None
            if self.adapter is not None:self.adapter.close();self.adapter=None
            if self.node is not None:self.node.destroy_node();self.node=None
            if self.context is not None and self.context.ok():self.context.shutdown()
            self.context=None;self._closed=True
            self._release_scene()

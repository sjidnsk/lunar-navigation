"""Observed-only deployment lifecycle. Does not spawn navigation or control."""
import numpy as np
from .contracts import TaskSpec, SensorSpec
from .decision import DecisionCore
from .ros_messages import InfrastructureError, RosNavigationAdapter, velocity_is_stopped


class ActorPolicy:
    """Actor-only artifact loader. Full training checkpoints are not deployment inputs."""
    @classmethod
    def load(cls,path):
        import torch
        from .model import Actor
        from .config import ModelConfig
        record=torch.load(path,map_location='cpu',weights_only=True)
        if record.get('schema')!='task_graph_v3' or set(record)!={'schema','model_config','version','state_dict'}:
            raise ValueError('expected task_graph_v3 actor_state Actor-only artifact')
        actor=Actor(ModelConfig(**record['model_config']))
        actor.load_state_dict(record['state_dict'],strict=True);actor.eval()
        return cls(actor)

    def __init__(self,actor):self.actor=actor

    def __call__(self,observation):
        import torch
        with torch.inference_mode():return int(self.actor([observation])[0].probs.argmax())


class InferenceRuntime:
    def __init__(self,adapter,policy,sensor=None,graph_config=None):
        self.adapter=adapter;self.policy=policy;self.sensor=sensor or SensorSpec()
        self.graph_config=graph_config;self.core=None;self.state='IDLE';self.report=None
        self.last_result=None;self._after_stop=None;self._pending_task=None;self._stamp=-1
        self.last_error=None

    def start(self,task):
        if self.adapter.inflight:
            self._pending_task=task;self._stop('RESTARTING');return
        self.core=DecisionCore(task,self.sensor,self.graph_config)
        self.report=None;self._stamp=-1;self.state='RUNNING';self.last_error=None
        self.adapter.require_fresh_snapshot()

    def _stop(self,state):
        self.state=state
        if self.adapter.inflight:self.adapter.cancel_goal()

    def pause(self):
        if self.state in ('RUNNING','WAITING_FOR_INPUT'):self._stop('PAUSING')

    def resume(self):
        if self.state=='PAUSED':
            self.adapter.require_fresh_snapshot();self.state='RUNNING'

    def cancel(self):
        if self.core is not None:self._stop('CANCELING')

    def handle_task(self,msg):
        if msg.command==msg.START:
            if msg.header.frame_id!='map':raise ValueError('task polygon must use native map frame')
            self.start(TaskSpec(msg.task_id,msg.header.frame_id,
                np.asarray([(p.x,p.y) for p in msg.boundary.points],np.float32)))
        elif self.core is not None and msg.task_id==self.core.task.task_id:
            if msg.command==msg.PAUSE:self.pause()
            elif msg.command==msg.RESUME:self.resume()
            elif msg.command==msg.CANCEL:self.cancel()

    def tick(self):
        if self.core is None:return
        if self.adapter.inflight:
            result=self.adapter.poll_goal()
            if result is not None:
                self.last_result=result
                self.adapter.require_fresh_snapshot()
        snap=self.adapter.poll_snapshot()
        if snap is not None:self.core.consume(snap)
        stamp=getattr(self.adapter,'processed_stamp_ns',0)
        if snap is not None and stamp!=self._stamp:
            self._stamp=stamp
            pose=self.adapter.observation_pose(stamp) if hasattr(self.adapter,'observation_pose') else self.adapter.pose
            if pose is not None and any(np.any(t.observed) for t in snap.tiles.values()):
                self.core.record_observation(pose,(1,))
        if self.state in ('PAUSING','CANCELING','RESTARTING'):
            if not self.adapter.inflight and velocity_is_stopped(self.adapter.velocity):
                if self.state=='RESTARTING':
                    task=self._pending_task;self._pending_task=None;self.start(task)
                else:self.state='PAUSED' if self.state=='PAUSING' else 'CANCELED'
            return
        if self.state not in ('RUNNING','WAITING_FOR_INPUT') or self.adapter.inflight:return
        if snap is None:self.state='WAITING_FOR_INPUT';return
        observation,self.report=self.core.observe(snap,self.adapter.velocity)
        if not self.report.available:self.state='WAITING_FOR_INPUT';return
        if self.report.exhausted:self.state='EXHAUSTED';return
        if not len(observation.goals):raise InfrastructureError('available nonexhausted state has no actions')
        action=int(self.policy(observation))
        if not 0<=action<len(observation.goals):raise ValueError('Actor action outside frozen observation')
        self.adapter.begin_goal(observation.goals[action]);self.state='RUNNING'

    def status(self):
        return {'state':self.state,'task_id':None if self.core is None else self.core.task.task_id,
            'known_area_m2':None if self.core is None else self.core.coverage.known_area_m2,
            'exhausted':None if self.report is None or not self.report.available else self.report.exhausted,
            'reference_area_m2':None,'reference_available':False,
            'reason_code':self.last_error or (self.report.reason_code if self.report else 'INPUT_UNAVAILABLE'),
            'navigation_reason':None if self.last_result is None else self.last_result.reason_code}

    @classmethod
    def attach(cls,node,policy,*,sensor=None,graph_config=None,task_topic='/Car/T4/exploration/task',
               status_topic='/Car/T4/exploration/drl_status',
               action_name='/Car/T4/navigation/navigate_to_pose',
               policy_map_service='/Car/T4/mapping/get_policy_map',
               odometry_topic='/Car/T3/localization/odometry',
               diagnostics_topic='/Car/T4/planning/diagnostics',
               path_reference_topic='/Car/T4/planning/path_reference',tf_topic='/tf'):
        """Attach to existing navigator/controller; caller spins supplied ROS node."""
        from lunar_pure_exploration_msgs.msg import PureExplorationTask
        from diagnostic_msgs.msg import DiagnosticArray,DiagnosticStatus,KeyValue
        adapter=RosNavigationAdapter(node,action_name=action_name,policy_map_service=policy_map_service,
            odometry_topic=odometry_topic,diagnostics_topic=diagnostics_topic,path_reference_topic=path_reference_topic,tf_topic=tf_topic)
        runtime=cls(adapter,policy,sensor,graph_config)
        runtime._task_subscription=node.create_subscription(PureExplorationTask,task_topic,runtime.handle_task,10)
        runtime._status_publisher=node.create_publisher(DiagnosticArray,status_topic,10)
        def tick():
            try:runtime.tick()
            except Exception as exc:
                runtime.last_error=str(exc);runtime.cancel()
            msg=DiagnosticArray();msg.header.stamp=node.get_clock().now().to_msg()
            entry=DiagnosticStatus(name='drl_exploration',message=runtime.state)
            entry.values=[KeyValue(key=k,value='unavailable' if v is None else str(v)) for k,v in runtime.status().items()]
            msg.status=[entry];runtime._status_publisher.publish(msg)
        runtime._timer=node.create_timer(.1,tick)
        return runtime
